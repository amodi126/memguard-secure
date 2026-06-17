"""
Voter injection experiment for MemGuard-Secure.

Derives a one-attack-per-conversation dataset from the existing enhanced attack
set by appending voter-targeting language to the social engineering attack turn.
Then replays the conversations through the existing Letta + MemGuard benchmark
pipeline, records per-voter outcomes, and prints a compact summary table.

Usage:
    python voter_injection_runner.py --input attacks_30_enhanced/ \
        --output-attacks attacks_30_voter_injection/ \
        --results results/voter_injection/
"""

import argparse
import json
import os
from collections import defaultdict

from scipy import stats

from attack_generator import (
    _VOTER_INJECTION_SUFFIXES,
    derive_voter_injection_conversation,
)
from benchmark_runner import create_client, load_attack_files, run_scenario


SCENARIO_ORDER = ["travel_planner", "personal_finance", "medical_intake"]


def clopper_pearson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Exact Clopper-Pearson confidence interval for a binomial proportion."""
    if trials == 0:
        return 0.0, 0.0, 0.0

    alpha = 1 - confidence
    if successes == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)

    if successes == trials:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    return successes / trials, float(lower), float(upper)


def build_voter_injection_dataset(input_dir: str, output_dir: str) -> dict:
    """Load enhanced conversations and derive voter injection conversations."""
    os.makedirs(output_dir, exist_ok=True)
    scenarios_data = load_attack_files(input_dir)
    derived = {}

    for scenario_name in SCENARIO_ORDER:
        if scenario_name not in scenarios_data:
            continue

        conversations = scenarios_data[scenario_name]
        new_conversations = []
        for i, conv in enumerate(conversations):
            variant_index = i % len(_VOTER_INJECTION_SUFFIXES)
            new_conversations.append(
                derive_voter_injection_conversation(scenario_name, conv, variant_index=variant_index)
            )

        out_path = os.path.join(output_dir, f"{scenario_name}_voter_injection.json")
        with open(out_path, "w") as f:
            json.dump(new_conversations, f, indent=2)

        derived[scenario_name] = new_conversations
        print(f"Saved {len(new_conversations)} voter-injection conversations → {out_path}")

    return derived


def summarize_results(all_results: list[dict]) -> tuple[list[dict], dict]:
    """Compute scenario-level summary rows and per-voter breakdown."""
    rows = []
    per_voter = defaultdict(lambda: {"allow": 0, "block": 0, "total": 0})

    by_scenario = defaultdict(list)
    for result in all_results:
        by_scenario[result["scenario"]].append(result)

        for atk in result.get("attack_results_per_turn", []):
            votes = atk.get("voter_votes", {})
            for voter, voted_safe in votes.items():
                per_voter[voter]["total"] += 1
                if voted_safe:
                    per_voter[voter]["allow"] += 1
                else:
                    per_voter[voter]["block"] += 1

    total_attacks_succeeded = 0
    total_attacks = 0
    total_false_blocks = 0
    total_conversations = 0

    for scenario_name in sorted(by_scenario.keys()):
        results = by_scenario[scenario_name]
        attacks_succeeded = sum(r.get("attacks_succeeded", 0) for r in results)
        attacks_total = sum(r.get("attacks_total", 0) for r in results)
        false_blocks = sum(1 for r in results if not r.get("legit_succeeded", True))
        total = len(results)

        asr, asr_lo, asr_hi = clopper_pearson_ci(attacks_succeeded, attacks_total)
        fpr, fpr_lo, fpr_hi = clopper_pearson_ci(false_blocks, total)

        rows.append({
            "scenario": scenario_name,
            "asr": asr,
            "asr_ci_95": [asr_lo, asr_hi],
            "fpr": fpr,
            "fpr_ci_95": [fpr_lo, fpr_hi],
            "n": total,
            "attacks_succeeded": attacks_succeeded,
            "attacks_total": attacks_total,
            "false_blocks": false_blocks,
        })

        total_attacks_succeeded += attacks_succeeded
        total_attacks += attacks_total
        total_false_blocks += false_blocks
        total_conversations += total

    overall_asr, overall_asr_lo, overall_asr_hi = clopper_pearson_ci(total_attacks_succeeded, total_attacks)
    overall_fpr, overall_fpr_lo, overall_fpr_hi = clopper_pearson_ci(total_false_blocks, total_conversations)

    summary = {
        "overall": {
            "asr": overall_asr,
            "asr_ci_95": [overall_asr_lo, overall_asr_hi],
            "fpr": overall_fpr,
            "fpr_ci_95": [overall_fpr_lo, overall_fpr_hi],
            "n": total_conversations,
            "attacks_succeeded": total_attacks_succeeded,
            "attacks_total": total_attacks,
            "false_blocks": total_false_blocks,
        },
        "per_voter": {},
    }

    for voter, stats_dict in sorted(per_voter.items()):
        allow_rate = stats_dict["allow"] / stats_dict["total"] if stats_dict["total"] else 0.0
        summary["per_voter"][voter] = {
            "allow": stats_dict["allow"],
            "block": stats_dict["block"],
            "total": stats_dict["total"],
            "allow_rate": allow_rate,
        }

    return rows, summary


def print_summary(rows: list[dict], summary: dict):
    """Print a compact table matching the existing benchmark style."""
    print(f"\n{'='*78}")
    print("VOTER INJECTION SUMMARY")
    print(f"{'='*78}")
    print(f"{'Scenario':<20} {'ASR':>8} {'FPR':>8} {'n':>4} {'ASR 95% CI':>24}")
    print(f"{'-'*78}")

    for row in rows:
        ci = row["asr_ci_95"]
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        print(
            f"  {row['scenario']:<18} {row['asr']:>8.3f} {row['fpr']:>8.3f} "
            f"{row['n']:>4} {ci_str:>24}"
        )

    overall = summary["overall"]
    ci = overall["asr_ci_95"]
    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    print(f"{'-'*78}")
    print(
        f"  {'OVERALL':<18} {overall['asr']:>8.3f} {overall['fpr']:>8.3f} "
        f"{overall['n']:>4} {ci_str:>24}"
    )

    print(f"\n{'Per-Voter Breakdown':<78}")
    print(f"{'Voter':<28} {'ALLOW':>8} {'BLOCK':>8} {'Total':>8} {'Allow %':>10}")
    print(f"{'-'*78}")
    for voter, stats_dict in sorted(summary["per_voter"].items()):
        print(
            f"  {voter:<26} {stats_dict['allow']:>8} {stats_dict['block']:>8} "
            f"{stats_dict['total']:>8} {stats_dict['allow_rate']:>9.3f}"
        )
    print(f"{'='*78}")


def save_summary(output_dir: str, rows: list[dict], summary: dict, all_results: list[dict], source_dir: str):
    """Save a single JSON file with the full experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "voter_injection_results.json")
    payload = {
        "experiment": "voter_injection",
        "source_attacks": source_dir,
        "attack_variants": _VOTER_INJECTION_SUFFIXES,
        "summary_rows": rows,
        "summary": summary,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved summary JSON → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="attacks_30_enhanced/",
                        help="Directory of enhanced attack JSON files")
    parser.add_argument("--output-attacks", default="attacks_30_voter_injection/",
                        help="Directory for derived voter-injection attack JSON files")
    parser.add_argument("--results", default="results/voter_injection/",
                        help="Directory for benchmark results")
    parser.add_argument("--baseline", action="store_true",
                        help="Run with MemGuard disabled in the container")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    defended = not args.baseline
    capture_voter_data = defended

    build_voter_injection_dataset(args.input, args.output_attacks)

    client = create_client()
    scenarios_data = load_attack_files(args.output_attacks)
    if not scenarios_data:
        print(f"No derived attack files found at: {args.output_attacks}")
        raise SystemExit(1)

    all_results = []
    for scenario_name in SCENARIO_ORDER:
        conversations = scenarios_data.get(scenario_name)
        if not conversations:
            continue

        results = run_scenario(
            client,
            scenario_name,
            conversations,
            defended=defended,
            output_dir=args.results,
            verbose=args.verbose,
            capture_voter_data=capture_voter_data,
        )
        all_results.extend(results)

    rows, summary = summarize_results(all_results)
    print_summary(rows, summary)
    save_summary(args.results, rows, summary, all_results, args.input)


if __name__ == "__main__":
    main()