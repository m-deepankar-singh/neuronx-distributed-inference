"""NKI per-chunk DeltaNet kernel for CTE (context encoding / prefill).

Single-chunk kernel: processes one chunk (128 tokens) with a stable
triangular solve for intra-chunk correction. The caller loops over chunks in
PyTorch, passing state between calls.

Each kernel call:
  - Takes one chunk of data: q, k, v as (128x128)
  - Takes scalar per-token gates/decays as (128x1): beta, g_cumsum, g_last
  - Takes recurrent state_in (128x128)
  - Returns chunk output (128x128) and state_out (128x128)

No sequence-indexed DMA inside the kernel -- all inputs/outputs are full tiles.
This avoids the DMA OOB issue seen with nl.sequential_range + slice indexing
in the NxDI model compilation context.

NKI v3 (SDK 2.29, NKI 0.3.0). Uses nki.* namespace.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128

# Broadcast partition 0 to all partitions in a 32-wide group.
_BROADCAST_MASK = [0] * 32


@nki.jit
def deltanet_chunk_step(
    query,  # (128, 128) float32 -- one chunk, l2-normed+scaled
    key,  # (128, 128) float32 -- one chunk, l2-normed
    value,  # (128, 128) float32 -- one chunk
    beta_scalar,  # (128, 1) float32 -- write gate per token
    g_cumsum,  # (128, 1) float32 -- cumsum of g within chunk
    g_last,  # (128, 1) float32 -- g_cumsum[-1], repeated over rows
    state_in,  # (128, 128) float32 -- recurrent state from previous chunk
    lower_mask,  # (128, 128) float32 -- strict lower triangular
    identity,  # (128, 128) float32 -- identity matrix
    lower_mask_diag,  # (128, 128) float32 -- lower tri with diagonal
):
    """Process one chunk of DeltaNet.

    Returns:
        output:    (128, 128) float32 -- chunk output
        state_out: (128, 128) float32 -- updated recurrent state
    """
    C, dim = query.shape  # C = 128, dim = 128

    # Output tensors in HBM
    output = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load all inputs into SBUF
    q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_c, src=query)

    k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_c, src=key)

    v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_c, src=value)

    beta_p = nl.ndarray((P_MAX, 1), dtype=beta_scalar.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_p, src=beta_scalar)

    gc_p = nl.ndarray((P_MAX, 1), dtype=g_cumsum.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc_p, src=g_cumsum)

    gl_p = nl.ndarray((P_MAX, 1), dtype=g_last.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gl_p, src=g_last)

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=state_in)

    # Load masks
    eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=identity)

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)

    # ============================================================
    # k_beta = K * beta, v_beta = V * beta
    # ============================================================
    k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_beta,
        data=k_c,
        op0=nl.multiply,
        operand0=beta_p,
        engine=nisa.vector_engine,
    )

    v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=v_beta,
        data=v_c,
        op0=nl.multiply,
        operand0=beta_p,
        engine=nisa.vector_engine,
    )

    # ============================================================
    # Stable decay factors from cumulative log-decay
    #
    # The caller passes g_cumsum and g_last as compact (128, 1) tensors.  Build
    # pairwise decays as exp(gc[i] - gc[j]) so no individual exp(-gc[j]) term
    # can overflow.
    # ============================================================
    exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gc_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gc_p[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )

    exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gl_p[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )

    gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=gc_padded, value=0.0)
    nisa.tensor_copy(dst=gc_padded[0:P_MAX, 0:1], src=gc_p[0:P_MAX, 0:1])

    gc_row_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=gc_row_psum, data=gc_padded)

    gc_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gc_row[0:1, 0:P_MAX], src=gc_row_psum[0:1, 0:P_MAX])

    gc_row_broadcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=gc_row[0:1, 0:P_MAX],
            dst=gc_row_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
            shuffle_mask=_BROADCAST_MASK,
        )

    gc_col_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gc_col_strict,
        data=Lmask,
        op0=nl.multiply,
        operand0=gc_p,
        engine=nisa.vector_engine,
    )
    gc_row_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gc_row_strict, data1=gc_row_broadcast, data2=Lmask, op=nl.multiply
    )
    g_diff_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=g_diff_strict,
        data1=gc_col_strict,
        data2=gc_row_strict,
        op=nl.subtract,
    )
    decay_strict_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=decay_strict_raw,
        op=nl.exp,
        data=g_diff_strict,
        bias=None,
        scale=1.0,
    )
    decay_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=decay_strict, data1=decay_strict_raw, data2=Lmask, op=nl.multiply
    )

    gc_col_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gc_col_diag,
        data=Lmask_d,
        op0=nl.multiply,
        operand0=gc_p,
        engine=nisa.vector_engine,
    )
    gc_row_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gc_row_diag, data1=gc_row_broadcast, data2=Lmask_d, op=nl.multiply
    )
    g_diff_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=g_diff_diag,
        data1=gc_col_diag,
        data2=gc_row_diag,
        op=nl.subtract,
    )
    decay_diag_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=decay_diag_raw,
        op=nl.exp,
        data=g_diff_diag,
        bias=None,
        scale=1.0,
    )
    decay_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=decay_diag, data1=decay_diag_raw, data2=Lmask_d, op=nl.multiply
    )

    # ============================================================
    # Phase 1: Build A matrix (intra-chunk correction)
    # QK = k_beta @ k^T  -- contract over features
    # ============================================================
    kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kb_T_psum, stationary=k_beta, moving=eye)
    kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

    k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=k_T_psum, stationary=k_c, moving=eye)
    k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_T, src=k_T_psum)

    QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
    QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QK, src=QK_psum)

    # QK_decay[i,j] = QK[i,j] * exp(gc[i] - gc[j]) for i > j.
    QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=QK_decay, data1=QK, data2=decay_strict, op=nl.multiply)

    # A = -QK_decay * lower_mask
    neg_QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=neg_QK_decay,
        data=QK_decay,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )
    A_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=A_mat, data1=neg_QK_decay, data2=Lmask, op=nl.multiply)

    # ============================================================
    # Build the single RHS needed for v_new.
    #
    # The materialized-inverse path computes:
    #   value_corr = N @ v_beta
    #   k_cumdecay = N @ (k_beta * exp(gc))
    #   v_new = value_corr - k_cumdecay @ state
    #
    # By associativity and linearity:
    #   v_new = N @ (v_beta - (k_beta * exp(gc)) @ state)
    #
    # So solve one triangular RHS directly and avoid materializing the full N.
    # ============================================================
    kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=kb_exp_gc,
        data=k_beta,
        op0=nl.multiply,
        operand0=exp_gc_p,
        engine=nisa.vector_engine,
    )

    kbe_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=kbe_T_psum, data=kb_exp_gc)
    kbe_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kbe_T, src=kbe_T_psum)

    kbe_state_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kbe_state_psum, stationary=kbe_T, moving=state)
    kbe_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kbe_state, src=kbe_state_psum)

    solve_rhs = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=solve_rhs, data1=v_beta, data2=kbe_state, op=nl.subtract)

    # ============================================================
    # Direct stable triangular solve for:
    #   v_new = solve((I - A_mat), solve_rhs)
    #
    # A_mat is strictly lower triangular, so each row depends only on previous
    # rows. This keeps the compiler-safe "full matmul + row select" structure
    # used by the validated inverse solve, but applies it to the one RHS the
    # model actually needs instead of materializing N = inv(I - A_mat).
    # ============================================================
    v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=v_new, value=0.0)

    A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=A_T_psum, data=A_mat)
    A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=A_T, src=A_T_psum)

    for solve_i in nl.static_range(P_MAX):
        row_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=row_psum, stationary=A_T, moving=v_new)
        row_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row_prod, src=row_psum)

        row_with_rhs = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=row_with_rhs,
            data1=row_prod,
            data2=solve_rhs,
            op=nl.add,
        )

        row_mask = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=row_mask[0:P_MAX, 0:1],
            src=eye[0:P_MAX, solve_i : solve_i + 1],
        )

        row_update = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=row_update,
            data=row_with_rhs,
            op0=nl.multiply,
            operand0=row_mask,
            engine=nisa.vector_engine,
        )

        v_next = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_next, data1=v_new, data2=row_update, op=nl.add)
        nisa.tensor_copy(dst=v_new, src=v_next)

    # ============================================================
    # Phase 2: Inter-chunk state propagation
    # attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
    # ============================================================
    q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=q_T_psum, stationary=q_c, moving=eye)
    q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_T, src=q_T_psum)

    qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
    qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_raw, src=qk_psum)

    attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=attn_intra, data1=qk_raw, data2=decay_diag, op=nl.multiply)

    # attn_inter = (q * exp(g_cumsum)) @ state
    # ============================================================
    q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_exp,
        data=q_c,
        op0=nl.multiply,
        operand0=exp_gc_p,
        engine=nisa.vector_engine,
    )

    qe_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qe_T_psum, stationary=q_exp, moving=eye)
    qe_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qe_T, src=qe_T_psum)

    ai_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ai_psum, stationary=qe_T, moving=state)
    attn_inter = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_inter, src=ai_psum)

    # ============================================================
    # attn_intra @ v_new
    # ============================================================
    ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ai_T_psum, stationary=attn_intra, moving=eye)
    ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

    intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=intra_psum, stationary=ai_T, moving=v_new)
    intra_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=intra_out, src=intra_psum)

    # ============================================================
    # chunk_output = attn_inter + intra_out
    # ============================================================
    chunk_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add)

    nisa.dma_copy(dst=output, src=chunk_out)

    # ============================================================
    # State update: state_new = state * exp(g_last)
    #                         + (k * exp(g_last - gc))^T @ v_new
    # ============================================================
    gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gl_minus_gc_p,
        data1=gl_p,
        data2=gc_p,
        op=nl.subtract,
    )
    exp_gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_minus_gc_p,
        op=nl.exp,
        data=gl_minus_gc_p,
        bias=None,
        scale=1.0,
    )

    k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_raw_decay,
        data=k_c,
        op0=nl.multiply,
        operand0=exp_gl_minus_gc_p,
        engine=nisa.vector_engine,
    )

    kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kv_psum, stationary=k_raw_decay, moving=v_new)
    kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kv_outer, src=kv_psum)

    state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=state_decayed,
        data=state,
        op0=nl.multiply,
        operand0=exp_gl_p,
        engine=nisa.vector_engine,
    )

    state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=state_new, data1=state_decayed, data2=kv_outer, op=nl.add)

    nisa.dma_copy(dst=state_out, src=state_new)

    return output, state_out
