from __future__ import annotations

from typing import Any, Dict


def _task_tolerance(tolerances: Dict[str, float], task: str) -> float:
    return float(tolerances.get(task, tolerances.get("default", 0.02)))


def compare_results(
    hf_results: Dict[str, Any],
    neuron_results: Dict[str, Any],
    tolerances: Dict[str, float],
) -> Dict[str, Any]:
    tasks = sorted(set(hf_results) & set(neuron_results))
    comparisons = {}
    overall_pass = True
    for task in tasks:
        metric = "acc_norm" if "acc_norm" in hf_results[task] else "acc"
        hf_score = float(hf_results[task][metric])
        neuron_score = float(neuron_results[task][metric])
        delta = hf_score - neuron_score
        tolerance = _task_tolerance(tolerances, task)
        passed = delta <= tolerance
        overall_pass = overall_pass and passed
        comparisons[task] = {
            "metric": metric,
            "hf_score": hf_score,
            "neuron_score": neuron_score,
            "delta": delta,
            "tolerance": tolerance,
            "pass": passed,
        }
    return {"overall_pass": overall_pass, "tasks": comparisons}


def print_comparison_report(comparison: Dict[str, Any]) -> None:
    print("\nStage 7 comparison:")
    for task, result in comparison["tasks"].items():
        status = "PASS" if result["pass"] else "FAIL"
        print(
            f"  {task}: {status} "
            f"{result['metric']} hf={result['hf_score']:.4f} "
            f"neuron={result['neuron_score']:.4f} "
            f"delta={result['delta']:.4f} tol={result['tolerance']:.4f}"
        )
    print(f"  overall: {'PASS' if comparison['overall_pass'] else 'FAIL'}")
