#!/usr/bin/env python3
"""Scan Qwen3.6 runtime logs for coherence-breaking markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_MARKERS = [
    "negative token_id",
    "out-of-vocab token_id",
    "fallback argmax",
    "finite=0",
    "nan=",
    "NaN",
    "NRT_RESOURCE",
    "Traceback",
    "RuntimeError",
    "Internal Server Error",
    "InternalServerError",
    "EngineDeadError",
]


def scan_text(text: str, markers: Iterable[str] = DEFAULT_MARKERS) -> dict[str, object]:
    marker_list = list(markers)
    matches: dict[str, list[dict[str, object]]] = {marker: [] for marker in marker_list}
    for line_no, line in enumerate(text.splitlines(), start=1):
        for marker in marker_list:
            if marker in line:
                matches[marker].append({"line": line_no, "text": line})
    matches = {marker: rows for marker, rows in matches.items() if rows}
    return {
        "ok": not matches,
        "markers": marker_list,
        "matches": matches,
        "match_count": sum(len(rows) for rows in matches.values()),
    }


def scan_files(paths: Iterable[Path], markers: Iterable[str] = DEFAULT_MARKERS) -> dict[str, object]:
    marker_list = list(markers)
    files = []
    ok = True
    total_matches = 0
    for path in paths:
        result = scan_text(path.read_text(errors="replace"), marker_list)
        result["path"] = str(path)
        files.append(result)
        ok = ok and bool(result["ok"])
        total_matches += int(result["match_count"])
    return {
        "ok": ok,
        "markers": marker_list,
        "files": files,
        "match_count": total_matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--marker",
        action="append",
        dest="markers",
        help="Override default marker list. May be passed more than once.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a short text summary.",
    )
    args = parser.parse_args()

    markers = args.markers if args.markers is not None else DEFAULT_MARKERS
    result = scan_files(args.logs, markers)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print(f"LOG_SCAN_OK files={len(result['files'])}")
    else:
        print(f"LOG_SCAN_FAILED match_count={result['match_count']}")
        for file_result in result["files"]:
            for marker, rows in file_result["matches"].items():
                for row in rows:
                    print(
                        f"{file_result['path']}:{row['line']}: "
                        f"{marker}: {row['text']}"
                    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
