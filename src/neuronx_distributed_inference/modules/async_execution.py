from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import torch

if TYPE_CHECKING:
    from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM
    from neuronx_distributed_inference.models.model_wrapper import ModelWrapper


def _is_hybrid_apc_enabled(neuron_base_instance: "NeuronBaseForCausalLM") -> bool:
    return bool(getattr(neuron_base_instance.config, "use_hybrid_apc_manager", False))


def _batch_vector(
    input_dict: Dict[str, Any],
    key: str,
    *,
    batch_size: int,
    default: int = 0,
) -> torch.Tensor:
    value = input_dict.get(key)
    if value is None:
        return torch.full((batch_size,), default, dtype=torch.int32)
    value = value.reshape(-1).to(torch.int32)
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] > batch_size:
        return value[:batch_size]
    pad = torch.full((batch_size - value.shape[0],), default, dtype=value.dtype)
    return torch.cat([value, pad], dim=0)


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _to_python_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _get_hybrid_apc_bridge(
    neuron_base_instance: "NeuronBaseForCausalLM",
    input_dict: Dict[str, Any],
):
    return _first_present(
        input_dict.get("hybrid_apc_bridge"),
        getattr(neuron_base_instance, "hybrid_apc_bridge", None),
    )


def prepare_hybrid_apc_request_for_execution(
    neuron_base_instance: "NeuronBaseForCausalLM",
    input_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Run scheduler-side hybrid APC request preparation when metadata exists.

    The concrete bridge lives with the Qwen contrib model. This function is
    intentionally duck-typed so the core async path does not import contrib
    modules. vLLM/NxDI request prep can opt in by attaching a bridge object plus
    attention-hit metadata to ``input_dict``.
    """

    if not _is_hybrid_apc_enabled(neuron_base_instance):
        return input_dict

    bridge = _get_hybrid_apc_bridge(neuron_base_instance, input_dict)
    if bridge is None:
        return input_dict

    request_id = _first_present(
        input_dict.get("hybrid_request_id"),
        input_dict.get("request_id"),
    )
    attention_hit_len = _first_present(
        input_dict.get("vllm_attention_hit_len"),
        input_dict.get("hybrid_attention_hit_len"),
        input_dict.get("attention_hit_len"),
    )
    if request_id is None and attention_hit_len is None:
        return input_dict
    if request_id is None:
        raise ValueError("hybrid APC request prep requires request_id")
    if attention_hit_len is None:
        raise ValueError("hybrid APC request prep requires attention hit length")

    prepared = bridge.prepare_request(
        request_id=request_id,
        input_dict=input_dict,
        attention_hit_len=_to_python_int(attention_hit_len),
        request_prefix_len=input_dict.get("request_prefix_len"),
        cumulative_hashes_by_prefix_len=_first_present(
            input_dict.get("vllm_or_local_prefix_hashes"),
            input_dict.get("cumulative_hashes_by_prefix_len"),
            input_dict.get("hybrid_cumulative_hashes_by_prefix_len"),
        ),
        attention_block_refs_by_prefix_len=_first_present(
            input_dict.get("attention_block_refs"),
            input_dict.get("attention_block_refs_by_prefix_len"),
            input_dict.get("hybrid_attention_block_refs_by_prefix_len"),
        ),
    )
    input_dict["_hybrid_apc_bridge"] = bridge
    input_dict["_hybrid_apc_prepared"] = prepared
    return prepared.input_dict


def finish_hybrid_apc_request(input_dict: Dict[str, Any]):
    bridge = input_dict.pop("_hybrid_apc_bridge", None)
    prepared = input_dict.pop("_hybrid_apc_prepared", None)
    if bridge is None or prepared is None:
        return

    actual_refs = _first_present(
        input_dict.get("actual_refs"),
        input_dict.get("actual_attention_block_refs"),
        input_dict.get("hybrid_actual_attention_block_refs"),
        getattr(prepared, "attention_block_refs", None),
    )
    try:
        bridge.commit_prefill(prepared, attention_block_refs=actual_refs)
    except Exception:
        bridge.cancel_request(prepared)
        raise
    bridge.finish_request(prepared.request_id)


def cancel_hybrid_apc_request(input_dict: Dict[str, Any]):
    bridge = input_dict.pop("_hybrid_apc_bridge", None)
    prepared = input_dict.pop("_hybrid_apc_prepared", None)
    if bridge is not None and prepared is not None:
        bridge.cancel_request(prepared)


def prepare_hybrid_apc_model_inputs(
    neuron_base_instance: "NeuronBaseForCausalLM",
    input_dict: Dict[str, Any],
) -> list[torch.Tensor]:
    """Build optional Qwen hybrid APC args for prefix-caching execution.

    The vLLM scheduler owns prefix hashes and checkpoint-slot allocation. This
    bridge only translates scheduler-provided values into the fixed traced model
    inputs. If no restore/commit slots are supplied, masks stay zero and the
    model executes without GDN checkpoint reuse.
    """

    if not _is_hybrid_apc_enabled(neuron_base_instance):
        return []

    batch_size = int(input_dict["seq_ids"].reshape(-1).shape[0])
    empty = torch.empty(0)

    computed_context_lens = input_dict.get("computed_context_lens")
    if computed_context_lens is None:
        restore_prefix_lens = torch.zeros((batch_size,), dtype=torch.int32)
    else:
        restore_prefix_lens = computed_context_lens.reshape(-1).to(torch.int32)
        if restore_prefix_lens.shape[0] != batch_size:
            restore_prefix_lens = _batch_vector(
                {"value": restore_prefix_lens},
                "value",
                batch_size=batch_size,
            )

    restore_slot_ids = _batch_vector(
        input_dict,
        "hybrid_restore_slot_ids",
        batch_size=batch_size,
        default=0,
    )
    restore_mask = _batch_vector(
        input_dict,
        "hybrid_restore_mask",
        batch_size=batch_size,
        default=0,
    )

    if "hybrid_restore_prefix_lens" in input_dict:
        restore_prefix_lens = _batch_vector(
            input_dict,
            "hybrid_restore_prefix_lens",
            batch_size=batch_size,
            default=0,
        )

    commit_slot_ids = _batch_vector(
        input_dict,
        "hybrid_commit_slot_ids",
        batch_size=batch_size,
        default=0,
    )
    if "hybrid_commit_mask" in input_dict:
        commit_mask = _batch_vector(
            input_dict,
            "hybrid_commit_mask",
            batch_size=batch_size,
            default=0,
        )
    else:
        commit_mask = torch.zeros((batch_size,), dtype=torch.int32)

    llava_args = input_dict.get("llava_args") or []
    rotary_position_id = _first_present(
        input_dict.get("rotary_position_id"),
        input_dict.get("rotary_position_ids"),
        llava_args[2] if len(llava_args) >= 3 else None,
        empty,
    )
    vision_embeddings = _first_present(
        input_dict.get("vision_embeddings"),
        llava_args[0] if len(llava_args) >= 1 else None,
        empty,
    )
    vision_mask = _first_present(
        input_dict.get("vision_mask"),
        llava_args[1] if len(llava_args) >= 2 else None,
        empty,
    )

    return [
        input_dict.get("tile_q_indices", empty),
        input_dict.get("tile_block_tables", empty),
        input_dict.get("tile_masks", empty),
        input_dict.get("inputs_embeds", empty),
        input_dict.get("kv_cache", empty),
        input_dict.get("active_mask", empty),
        rotary_position_id,
        vision_embeddings,
        vision_mask,
        restore_slot_ids,
        restore_mask,
        restore_prefix_lens,
        commit_slot_ids,
        commit_mask,
    ]


class AsyncTensorWrapper:
    """
    Wrapper class for tensors from models executed with async runtime.

    Attributes:

    1. `async_result`: A 2d list of tensors representing a collection of outputs from all tp ranks.
        **Exception:** If on_cpu=True, ranked_tensor is a 2d list of tensors containing  to be concatenated together
    2. `batch_padded`: A boolean indicating if the result has been padded along batch dimension.
    3. `on_cpu`: A boolean indicating if the ranked tensors have been synced to CPU.

    All 4 possiblilities are to be handled:
    1. `(batch_padded=False, on_cpu=False)` which implies request_batch_size = compiled_batch_size
    2. `(batch_padded=True, on_cpu=False)` which implies request_batch_size < compiled_batch_size
    3. `(batch_padded=True, on_cpu=True)` which implies request_batch_size > compiled_batch_size and request_batch_size % compiled_batch_size != 0
    4. `(batch_padded=False, on_cpu=True)` which implies request_batch_size > compiled_batch_size and request_batch_size % compiled_batch_size == 0
    """

    def __init__(self, async_result: List[List[torch.Tensor]], batch_padded: bool, on_cpu: bool):
        self.async_result = async_result
        self.batch_padded = batch_padded
        self.on_cpu = on_cpu

        if self.on_cpu:
            assert not is_ranked_io(
                self.async_result
            ), f"Initialized with {on_cpu=} but found that async_result is still on Neuron."
        else:
            assert is_ranked_io(
                self.async_result
            ), f"Initialized with {on_cpu=} but found that async_result is still on CPU."

    def get_ranked_tensor(self):
        assert not self.on_cpu, "Can't get ranked tensor if async_result is already on CPU."
        return self.async_result

    def sync_async_result_to_cpu(
        self, seq_ids: torch.Tensor, is_fused_speculation: bool = False, early_exit: bool = False, is_prefix_caching: bool = False
    ):
        if not self.on_cpu:  # cases 1 and 2
            synced_result = get_async_output(self.async_result)
        else:  # cases 3 and 4
            if is_fused_speculation:
                synced_result = [torch.cat(x, dim=0) for x in zip(*self.async_result)]
            else:
                synced_result = torch.cat([x[0] for x in self.async_result], dim=0)
        if early_exit:  # used for discarding results
            return
        # handle unpadding based on supplied seq_ids
        batch_size = seq_ids.shape[0]
        seq_ids = seq_ids.reshape(batch_size)  # make sure it's 1d tensor for index_select
        if is_prefix_caching:
            seq_ids = torch.arange(batch_size)
        if isinstance(synced_result, torch.Tensor):
            return torch.index_select(synced_result, 0, seq_ids)

        index_select = lambda x: torch.index_select(x, 0, seq_ids)  # noqa: E731
        try:
            return list(map(index_select, synced_result))
        except Exception as e:
            raise type(e)(f"Detected failure case: {str(e)}, tensor to select {synced_result}, seq_ids {seq_ids}") from e


def execute_model_prefix_caching(
    neuron_base_instance: "NeuronBaseForCausalLM",
    model_to_execute: "ModelWrapper",
    input_dict: Dict[str, Any],
    pad_type: str = "first_fit",
) -> Tuple[AsyncTensorWrapper, bool]:
    original_input_dict = input_dict
    try:
        input_dict = prepare_hybrid_apc_request_for_execution(
            neuron_base_instance,
            input_dict,
        )
        if "num_queries" not in input_dict:
            full_context_lens = input_dict["full_context_lens"]
            computed_context_lens = input_dict["computed_context_lens"]
            num_queries = full_context_lens - computed_context_lens
            input_dict["num_queries"] = num_queries

        if (
            not neuron_base_instance.neuron_config.enable_fused_speculation
            and not neuron_base_instance.neuron_config.enable_eagle_speculation
        ):
            hybrid_apc_args = prepare_hybrid_apc_model_inputs(
                neuron_base_instance, input_dict
            )
            return model_to_execute(
                input_dict["input_ids"],
                input_dict["attention_mask"],
                input_dict["position_ids"],
                input_dict["seq_ids"],
                input_dict["sampling_params"],
                torch.empty(0),  # prev_hidden
                input_dict["adapter_ids"],
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                input_dict["slot_mapping"],
                input_dict["block_table"],
                input_dict["num_queries"],
                input_dict["computed_context_lens"],
                *hybrid_apc_args,
                pad_type=pad_type
            ), model_to_execute.is_neuron()
        elif neuron_base_instance.neuron_config.enable_eagle_speculation:
            return model_to_execute(
                input_dict["input_ids"],
                input_dict["attention_mask"],
                input_dict["position_ids"],
                input_dict["seq_ids"],
                input_dict["sampling_params"],
                torch.empty(0),  # prev_hidden
                input_dict["adapter_ids"],
                input_dict["slot_mapping"],
                input_dict["block_table"],
                input_dict["num_queries"],
                input_dict["computed_context_lens"],
                torch.empty(0),  # target_input_ids
                torch.empty(0),  # target_attention_mask
                torch.empty(0),  # target_position_ids
                torch.empty(0),  # target_slot_mapping
                torch.empty(0),  # target_active_block_table
                pad_type=pad_type
            ), model_to_execute.is_neuron()
        else:
            raise NotImplementedError("Non-EAGLE fused speculation with prefix caching does not support async mode.")
    except Exception:
        cancel_hybrid_apc_request(original_input_dict)
        raise


def execute_model(
    neuron_base_instance: "NeuronBaseForCausalLM",
    model_to_execute: "ModelWrapper",
    input_dict: Dict[str, Any],
    hits_bucket_boundary: bool = False,
) -> Tuple[AsyncTensorWrapper, bool]:
    pad_type = "first_fit" if not hits_bucket_boundary else "second_fit"

    if neuron_base_instance.neuron_config.is_prefix_caching:
        return execute_model_prefix_caching(neuron_base_instance,
                                            model_to_execute,
                                            input_dict,
                                            pad_type)

    ordered_tuple_inputs = neuron_base_instance._convert_input_dict_to_ordered_tuple(input_dict)
    return model_to_execute(*ordered_tuple_inputs, pad_type=pad_type), model_to_execute.is_neuron()


def get_async_output(ranked_async_tensor: Any, clone: bool = False):
    if not is_ranked_io(ranked_async_tensor):
        return ranked_async_tensor

    maybe_clone = lambda x: x.clone().detach() if clone else x  # noqa: E731
    return [maybe_clone(async_tensor.cpu()) for async_tensor in ranked_async_tensor[0]]


def is_ranked_io(input_ids: Any):
    # make sure the contents are List[List[torch.Tensor]]
    # and that tensor is a privateuseone device tensor (neuron)
    return (
        isinstance(input_ids, list)
        and isinstance(input_ids[0], list)
        and isinstance(input_ids[0][0], torch.Tensor)
        and input_ids[0][0].device.type == "privateuseone"
    )


def within_bounds(inputs: Dict[str, Any], max_length: int, generation_length: int):
    return (max_length - inputs["position_ids"].max().item()) > generation_length * 2


def will_hit_bucket_boundary(known_length: int, buckets: Union[List[int], List[Tuple[int, int]]], max_num_tokens_generated=1):
    hits_bucket_boundary = False

    for bucket in buckets:
        # If buckets are 2D, use last index.
        if not isinstance(bucket, int):
            bucket = bucket[-1]
        # we do max_num_tokens_generated * 2 because we speculatively execute one step
        # ahead, before knowing the results of the current neff execution. In the worst case
        # both neffs will have full matching tokens, which is problematic near the bucket boundary
        # therefore, we will execute at a higher bucket in such cases.
        if known_length < bucket and (known_length + max_num_tokens_generated * 2) >= bucket:
            hits_bucket_boundary = True
            return hits_bucket_boundary

    return hits_bucket_boundary


def causal_lm_async_execution(
    neuron_base_instance: "NeuronBaseForCausalLM",
    inputs: Dict[str, Any],
    is_fused_speculation: bool = False,
):
    is_prefix_caching = neuron_base_instance.neuron_config.is_prefix_caching

    # PREFILL STAGE:
    is_prefill = neuron_base_instance._is_prefill(inputs["position_ids"])
    neuron_base_instance.async_should_stop = False
    prefill_outputs = None
    is_run_on_neuron = None
    if is_prefill:
        try:
            prefill_outputs, is_run_on_neuron = execute_model(
                neuron_base_instance, neuron_base_instance.context_encoding_model, inputs
            )

            # Sequence IDs from vLLM will be in sorted order, but the maximum range of sequence IDs is
            # not [0, num_requested_prefills] but [0, max_num_seqs]. To prevent out-of-bound accesses,
            # we convert the sequence IDs to their argsorted values.
            _seq_ids = torch.argsort(inputs["seq_ids"])

            outputs = prefill_outputs.sync_async_result_to_cpu(
                _seq_ids, is_fused_speculation=is_fused_speculation, is_prefix_caching=is_prefix_caching
            )
            finish_hybrid_apc_request(inputs)
        except Exception:
            cancel_hybrid_apc_request(inputs)
            raise

        # clean up async state
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None

        return outputs, is_run_on_neuron

    # GENERATION STAGE:
    generation_model = (
        neuron_base_instance.token_generation_model
        if not is_fused_speculation
        else neuron_base_instance.fused_spec_model
    )
    generation_length = (
        1 if not is_fused_speculation else neuron_base_instance.neuron_config.speculation_length
    )
    known_seqlen = inputs["attention_mask"].shape[1]
    hits_bucket_boundary = will_hit_bucket_boundary(
        known_seqlen,
        buckets=generation_model.neuron_config.buckets,
        max_num_tokens_generated=generation_length,
    )

    stay_in_sync_mode = (
        not torch.equal(neuron_base_instance.prior_seq_ids, inputs["seq_ids"])
        or hits_bucket_boundary
    )
    start_async = not stay_in_sync_mode and neuron_base_instance.prior_outputs is None
    continue_async = not stay_in_sync_mode and not start_async

    if stay_in_sync_mode:
        # reset async state
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None

    if stay_in_sync_mode or start_async:
        next_outputs, is_run_on_neuron = execute_model(
            neuron_base_instance,
            generation_model,
            inputs,
            hits_bucket_boundary=hits_bucket_boundary,
        )
        if start_async:
            neuron_base_instance.prior_outputs = next_outputs
            neuron_base_instance.prior_seq_ids = inputs["seq_ids"]

    if start_async or continue_async:
        if within_bounds(inputs, neuron_base_instance.neuron_config.seq_len, generation_length):
            next_outputs = neuron_base_instance.prior_outputs
            inputs["input_ids"] = next_outputs.get_ranked_tensor()
            if neuron_base_instance.next_cpu_inputs is not None:
                for key in neuron_base_instance.next_cpu_inputs:
                    inputs[key] = neuron_base_instance.next_cpu_inputs[key]
            elif not is_fused_speculation:
                raise RuntimeError(
                    "Expected next_cpu_inputs to be generated for a non fused_spec model."
                )
            next_outputs, is_run_on_neuron = execute_model(
                neuron_base_instance, generation_model, inputs
            )
        else:
            if neuron_base_instance.prior_outputs is not None:
                outputs = neuron_base_instance.prior_outputs.sync_async_result_to_cpu(
                    inputs["seq_ids"], is_fused_speculation=is_fused_speculation, is_prefix_caching=is_prefix_caching
                )
                neuron_base_instance.prior_outputs = None
                neuron_base_instance.prior_seq_ids = None
                neuron_base_instance.async_should_stop = True
            else:
                raise RuntimeError(
                    "The stopping criteria for fused async should have been triggered, but it wasn't."
                )

            return outputs, True  # assume async mode only runs on neuron

    # output to be returned
    outputs: AsyncTensorWrapper = neuron_base_instance.prior_outputs if not stay_in_sync_mode else next_outputs
    outputs = outputs.sync_async_result_to_cpu(
        inputs["seq_ids"], is_fused_speculation=is_fused_speculation, is_prefix_caching=is_prefix_caching
    )

    if stay_in_sync_mode:
        # make sure prior outputs is not set
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None
        return outputs, is_run_on_neuron

    # next step
    neuron_base_instance.prior_outputs = next_outputs
    neuron_base_instance.prior_seq_ids = inputs["seq_ids"]

    return outputs, is_run_on_neuron
