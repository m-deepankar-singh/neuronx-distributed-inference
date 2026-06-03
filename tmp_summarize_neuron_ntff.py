#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _one_ntff_per_prefix(root: Path) -> list[Path]:
    by_prefix: dict[str, list[Path]] = {}
    for path in root.glob("*.ntff"):
        prefix = path.name.rsplit("_vnc_", 1)[0]
        by_prefix.setdefault(prefix, []).append(path)
    selected = []
    for paths in by_prefix.values():
        selected.append(
            sorted(paths, key=lambda item: (0 if item.name.endswith("_vnc_0.ntff") else 1, item.name))[0]
        )
    return sorted(selected)


def _short_name(name: str) -> str:
    match = re.search(r"/(context_encoding_model|token_generation_model)/([^/]+)/", name)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return name


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: tmp_summarize_neuron_ntff.py INSPECT_LEAF_DIR", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    explorer = os.environ.get("NEURON_EXPLORER", "/opt/aws/neuron/bin/neuron-explorer")
    out_path = root.parent.parent / "ntff_show_summary.tsv"
    rows = []
    for ntff in _one_ntff_per_prefix(root):
        command = [
            "timeout",
            "45",
            explorer,
            "show-session",
            "-s",
            str(ntff),
            "-j",
        ]
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            rows.append(
                [
                    "ERROR",
                    ntff.name,
                    str(ntff.stat().st_size),
                    f"rc={proc.returncode}",
                    proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
                ]
            )
            continue
        data = json.loads(proc.stdout)
        graphs = [
            graph
            for node in data.get("NeffNodes", [])
            for graph in node.get("NodeInfo", {}).get("Graphs", [])
        ]
        cycle = max((int(graph.get("CycleCount") or 0) for graph in graphs), default=0)
        instr = sum(int(graph.get("TotalInstructionTraceCount") or 0) for graph in graphs)
        events = sum(int(graph.get("EventsCount") or 0) for graph in graphs)
        errors = sum(int(graph.get("ErrorCount") or 0) for graph in graphs)
        name = data.get("Name", "")
        rows.append(
            [
                "OK",
                ntff.name,
                str(ntff.stat().st_size),
                f"cycle={cycle}",
                f"instr={instr}",
                f"events={events}",
                f"errors={errors}",
                _short_name(name),
                name,
            ]
        )
    text = "\n".join("\t".join(row) for row in rows) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"SUMMARY_PATH={out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
