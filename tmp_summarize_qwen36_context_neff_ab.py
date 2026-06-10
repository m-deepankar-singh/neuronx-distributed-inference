#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: tmp_summarize_qwen36_context_neff_ab.py ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    explorer = "/opt/aws/neuron/bin/neuron-explorer"
    rows: list[dict[str, object]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        ntffs = sorted(directory.glob("*.ntff"))
        if not ntffs:
            rows.append({"name": directory.name, "status": "missing_ntff"})
            continue
        ntff = ntffs[0]
        proc = subprocess.run(
            ["timeout", "60", explorer, "show-session", "-s", str(ntff), "-j"],
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            rows.append(
                {
                    "name": directory.name,
                    "status": "show_session_failed",
                    "returncode": proc.returncode,
                    "stderr_tail": proc.stderr.strip().splitlines()[-5:],
                }
            )
            continue
        data = json.loads(proc.stdout)
        graphs = [
            graph
            for node in data.get("NeffNodes", [])
            for graph in node.get("NodeInfo", {}).get("Graphs", [])
        ]
        rows.append(
            {
                "name": directory.name,
                "status": "ok",
                "ntff_size": ntff.stat().st_size,
                "cycle": max((int(graph.get("CycleCount") or 0) for graph in graphs), default=0),
                "instructions": sum(
                    int(graph.get("TotalInstructionTraceCount") or 0) for graph in graphs
                ),
                "events": sum(int(graph.get("EventsCount") or 0) for graph in graphs),
                "errors": sum(int(graph.get("ErrorCount") or 0) for graph in graphs),
                "graph_count": len(graphs),
                "session_name": data.get("Name", ""),
            }
        )
    out = root / "context_ab_show_session_summary.json"
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print("name\tstatus\tcycle\tinstr\tevents\terrors\tntff_size")
    for row in rows:
        print(
            "\t".join(
                str(row.get(key, ""))
                for key in [
                    "name",
                    "status",
                    "cycle",
                    "instructions",
                    "events",
                    "errors",
                    "ntff_size",
                ]
            )
        )
    print(f"SUMMARY_JSON={out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
