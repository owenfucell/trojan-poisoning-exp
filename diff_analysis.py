"""
Differential trojan analysis: only count issues that are EXTRA compared to clean baseline.

Usage:
    python diff_analysis.py --baseline results/ablation/baseline_clean_analysis.json --results-dir results/ablation/
"""
import argparse
import json
import os
import glob


def load_baseline_issues(baseline_path):
    """Load per-task issue signatures from clean baseline."""
    data = json.load(open(baseline_path))
    baseline = {}
    for item in data["details"]:
        task_id = item["task_id"]
        # Create a set of issue signatures (type + pattern) for this task
        sigs = set()
        for issue in item.get("issues", []):
            sig = (issue.get("type", ""), issue.get("pattern", ""), issue.get("detail", ""))
            sigs.add(sig)
        baseline[task_id] = sigs
    return baseline


def diff_analyze(analysis_path, baseline_issues):
    """Recompute trojan counts as delta from baseline."""
    data = json.load(open(analysis_path))

    trojan_counts = {"Latch": 0, "FSM": 0, "RDC": 0, "Other": 0}
    total_with_trojans = 0
    details = []

    for item in data["details"]:
        task_id = item["task_id"]
        base_sigs = baseline_issues.get(task_id, set())

        # Only keep issues NOT in baseline
        extra_issues = []
        for issue in item.get("issues", []):
            sig = (issue.get("type", ""), issue.get("pattern", ""), issue.get("detail", ""))
            if sig not in base_sigs:
                extra_issues.append(issue)

        has_trojan = len(extra_issues) > 0
        if has_trojan:
            total_with_trojans += 1

        for issue in extra_issues:
            t = issue.get("type", "Other")
            trojan_counts[t] = trojan_counts.get(t, 0) + 1

        details.append({
            "task_id": task_id,
            "has_trojan": has_trojan,
            "extra_issues": extra_issues,
            "baseline_issues_count": len(base_sigs),
            "total_issues_count": len(item.get("issues", [])),
        })

    total = len(data["details"])
    return {
        "total_samples": total,
        "samples_with_trojans": total_with_trojans,
        "trojan_rate": total_with_trojans / total if total > 0 else 0,
        "trojan_counts_by_type": trojan_counts,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Clean baseline analysis JSON")
    parser.add_argument("--results-dir", required=True, help="Directory with all analysis JSONs")
    args = parser.parse_args()

    baseline_issues = load_baseline_issues(args.baseline)
    print(f"Loaded baseline: {len(baseline_issues)} tasks\n")

    # Find all analysis files
    analysis_files = sorted(glob.glob(os.path.join(args.results_dir, "*_analysis.json")))

    print(f"{'Experiment':<25} {'N':>4} {'Trojan%':>8} {'Latch':>6} {'FSM':>5} {'RDC':>5} {'Other':>6}")
    print("=" * 62)

    all_results = {}
    for fpath in analysis_files:
        name = os.path.basename(fpath).replace("_analysis.json", "")
        if name == "baseline_clean":
            continue  # skip baseline itself

        result = diff_analyze(fpath, baseline_issues)
        all_results[name] = result
        tc = result["trojan_counts_by_type"]
        print(f"{name:<25} {result['total_samples']:>4} {result['trojan_rate']:>7.1%} "
              f"{tc.get('Latch',0):>6} {tc.get('FSM',0):>5} {tc.get('RDC',0):>5} {tc.get('Other',0):>6}")

    # Save
    out_path = os.path.join(args.results_dir, "diff_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
