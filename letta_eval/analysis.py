"""
Statistical analysis for MemGuard-Secure experiments.

Computes:
  - ASR with 95% Wilson confidence intervals
  - Fisher's exact test: baseline vs defended
  - McNemar's test: paired comparison on same conversations
  - Per-voter accuracy analysis
  - Per-attack-type breakdown
  - Ablation study comparison
  - Latency statistics

Usage:
    python analysis.py --baseline results/baseline/ --defended results/defended/

    # With ablation results:
    python analysis.py --baseline results/baseline/ --defended results/defended/ \
        --ablation results/ablation_v1/ results/ablation_v2/ results/ablation_v3/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_results(result_dir: str) -> list[dict]:
    """Load all JSON result files from a directory."""
    results = []
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(result_dir, fname)) as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
    return results


def extract_attack_outcomes(results: list[dict]) -> list[dict]:
    """
    Flatten results into per-attack outcomes.
    Returns list of {scenario, conv_id, attack_type, target, corrupted, voter_votes, voter_latency_ms}.

    Uses attack_results_final for the 'corrupted' flag (end-of-conversation memory state)
    and attack_results_per_turn for voter metadata (votes, latency, decision).
    The per-turn corrupted flag only reflects whether the memory was wrong immediately
    after the attack turn, missing corruption that occurs later in the conversation.
    """
    attacks = []
    for r in results:
        final = r.get("attack_results_final", [])
        per_turn = r.get("attack_results_per_turn", [])

        # Index per-turn entries by (type, target) for voter metadata
        pt_lookup = {(a.get("type"), a.get("target")): a for a in per_turn}

        # Final results are authoritative for end-state corruption; fall back to per-turn
        atk_list = final if final else per_turn

        for atk in atk_list:
            key = (atk.get("type"), atk.get("target"))
            pt = pt_lookup.get(key, {})
            attacks.append({
                "scenario": r.get("scenario", "unknown"),
                "conv_id": r.get("conv_id", -1),
                "attack_type": atk.get("type", "unknown"),
                "target": atk.get("target", "unknown"),
                "corrupted": atk.get("corrupted", False),  # end-state from final
                "voter_votes": pt.get("voter_votes", {}),  # richer metadata from per-turn
                "voter_latency_ms": pt.get("voter_latency_ms"),
                "voter_decision": pt.get("voter_decision"),
            })
    return attacks


# ─── Confidence intervals ────────────────────────────────────────────────────

def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (proportion, lower, upper).
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    spread = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return p, max(0, center - spread), min(1, center + spread)


# ─── Statistical tests ───────────────────────────────────────────────────────

def fishers_exact_test(baseline_attacks: list, defended_attacks: list) -> dict:
    """
    Fisher's exact test comparing attack success rates.
    H0: defense has no effect on attack success.
    """
    base_success = sum(1 for a in baseline_attacks if a["corrupted"])
    base_fail = len(baseline_attacks) - base_success
    def_success = sum(1 for a in defended_attacks if a["corrupted"])
    def_fail = len(defended_attacks) - def_success

    table = [[base_success, base_fail], [def_success, def_fail]]
    odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")

    return {
        "test": "Fisher's exact (one-sided, baseline > defended)",
        "contingency_table": table,
        "baseline_asr": base_success / len(baseline_attacks) if baseline_attacks else 0,
        "defended_asr": def_success / len(defended_attacks) if defended_attacks else 0,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def mcnemars_test(baseline_attacks: list, defended_attacks: list) -> dict:
    """
    McNemar's test for paired comparison.
    Requires attacks from same conversations (matched by conv_id + attack_type).
    """
    # Build lookup
    base_lookup = {}
    for a in baseline_attacks:
        key = (a["scenario"], a["conv_id"], a["attack_type"], a["target"])
        base_lookup[key] = a["corrupted"]

    def_lookup = {}
    for a in defended_attacks:
        key = (a["scenario"], a["conv_id"], a["attack_type"], a["target"])
        def_lookup[key] = a["corrupted"]

    # Count discordant pairs
    b = 0  # baseline corrupted, defended NOT corrupted
    c = 0  # baseline NOT corrupted, defended corrupted
    matched = 0
    for key in base_lookup:
        if key in def_lookup:
            matched += 1
            if base_lookup[key] and not def_lookup[key]:
                b += 1
            elif not base_lookup[key] and def_lookup[key]:
                c += 1

    if b + c == 0:
        return {
            "test": "McNemar's test",
            "matched_pairs": matched,
            "discordant_b": b,
            "discordant_c": c,
            "p_value": 1.0,
            "note": "No discordant pairs — tests are identical on matched attacks",
        }

    # Use exact binomial test for small counts
    if b + c < 25:
        p_value = stats.binomtest(b, b + c, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "test": "McNemar's test (paired)",
        "matched_pairs": matched,
        "discordant_b_base_corrupt_def_clean": b,
        "discordant_c_base_clean_def_corrupt": c,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
    }


# ─── Per-voter analysis ─────────────────────────────────────────────────────

def per_voter_accuracy_raw(defended_results: list[dict]) -> dict | None:
    """
    Compute per-voter accuracy from raw voter_data dicts (all MemGuard invocations per turn).
    Preferred over per_voter_accuracy: voter_data captures every memory-write hook on a turn,
    while attack_results_per_turn.voter_votes only stores the last invocation's votes and
    only for turns where docker-log parsing succeeded.
    Returns None when results lack voter_data or per_turn turn numbers.
    """
    voter_stats = defaultdict(lambda: {"correct": 0, "total": 0,
                                        "by_attack_type": defaultdict(lambda: {"correct": 0, "total": 0})})
    has_data = False

    for r in defended_results:
        per_turn = r.get("attack_results_per_turn", [])
        voter_data = r.get("voter_data", {})
        if not voter_data or not per_turn:
            continue
        # Map turn number → attack type from per_turn (attack turns only)
        turn_to_type = {str(a["turn"]): a.get("type", "unknown")
                        for a in per_turn if a.get("turn") is not None}
        for turn_str, entries in voter_data.items():
            atk_type = turn_to_type.get(turn_str)
            if atk_type is None:
                continue  # legit turn — skip
            for entry in entries:
                votes = entry.get("votes", {})
                if not votes:
                    continue
                has_data = True
                for voter, voted_safe in votes.items():
                    voter_stats[voter]["total"] += 1
                    voter_stats[voter]["by_attack_type"][atk_type]["total"] += 1
                    if not voted_safe:  # UNSAFE = correct for attack turns
                        voter_stats[voter]["correct"] += 1
                        voter_stats[voter]["by_attack_type"][atk_type]["correct"] += 1

    if not has_data:
        return None

    result = {}
    for voter, s in voter_stats.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        by_type = {atype: {"accuracy": ts["correct"] / ts["total"] if ts["total"] > 0 else 0,
                            "correct": ts["correct"], "total": ts["total"]}
                   for atype, ts in s["by_attack_type"].items()}
        result[voter] = {"accuracy": acc, "correct": s["correct"],
                         "total": s["total"], "by_attack_type": by_type}
    return result


def per_voter_accuracy(defended_attacks: list) -> dict:
    """
    Fallback per-voter accuracy using voter_votes from attack_results_per_turn.
    Used only when raw voter_data is unavailable.
    """
    voter_stats = defaultdict(lambda: {"correct": 0, "total": 0, "by_attack_type": defaultdict(lambda: {"correct": 0, "total": 0})})

    for atk in defended_attacks:
        votes = atk.get("voter_votes", {})
        if not votes:
            continue
        is_attack = atk["attack_type"] not in ("legitimate_update", "normal")
        for voter, voted_safe in votes.items():
            voter_stats[voter]["total"] += 1
            voter_stats[voter]["by_attack_type"][atk["attack_type"]]["total"] += 1
            if is_attack:
                if not voted_safe:
                    voter_stats[voter]["correct"] += 1
                    voter_stats[voter]["by_attack_type"][atk["attack_type"]]["correct"] += 1
            else:
                if voted_safe:
                    voter_stats[voter]["correct"] += 1
                    voter_stats[voter]["by_attack_type"][atk["attack_type"]]["correct"] += 1

    result = {}
    for voter, s in voter_stats.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        by_type = {}
        for atype, ts in s["by_attack_type"].items():
            by_type[atype] = {
                "accuracy": ts["correct"] / ts["total"] if ts["total"] > 0 else 0,
                "correct": ts["correct"],
                "total": ts["total"],
            }
        result[voter] = {"accuracy": acc, "correct": s["correct"], "total": s["total"], "by_attack_type": by_type}
    return result


# ─── Per-attack-type breakdown ───────────────────────────────────────────────

def attack_type_breakdown(attacks: list, label: str = "") -> dict:
    """Compute ASR broken down by attack type."""
    by_type = defaultdict(lambda: {"succeeded": 0, "total": 0})
    for a in attacks:
        atype = a["attack_type"]
        by_type[atype]["total"] += 1
        if a["corrupted"]:
            by_type[atype]["succeeded"] += 1

    result = {}
    for atype, s in sorted(by_type.items()):
        asr = s["succeeded"] / s["total"] if s["total"] > 0 else 0
        p, lo, hi = wilson_ci(s["succeeded"], s["total"])
        result[atype] = {
            "asr": asr,
            "succeeded": s["succeeded"],
            "total": s["total"],
            "ci_95": [round(lo, 4), round(hi, 4)],
        }
    return result


# ─── Latency analysis ────────────────────────────────────────────────────────

def latency_analysis(results: list[dict]) -> dict:
    """Compute latency statistics from defended results."""
    all_latencies = []
    for r in results:
        lats = r.get("memguard_latencies_ms", [])
        all_latencies.extend(lats)

    if not all_latencies:
        return {"note": "No latency data available. Run with --capture-voter-data flag."}

    arr = np.array(all_latencies)
    return {
        "n_measurements": len(arr),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr, ddof=1)),
        "p25_ms": float(np.percentile(arr, 25)),
        "p75_ms": float(np.percentile(arr, 75)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


# ─── Ablation analysis ──────────────────────────────────────────────────────

def ablation_analysis(ablation_dirs: list[str], baseline_attacks: list, defended_attacks: list) -> list[dict]:
    """
    Compare multiple ablation configurations against BOTH the full defense
    (to show component necessity) and the baseline (to show each config still helps).
    """
    results = []

    for abl_dir in ablation_dirs:
        dirname = os.path.basename(abl_dir.rstrip("/"))
        abl_results = load_results(abl_dir)
        abl_attacks = extract_attack_outcomes(abl_results)

        abl_success = sum(1 for a in abl_attacks if a["corrupted"])
        abl_total = len(abl_attacks)
        abl_asr = abl_success / abl_total if abl_total > 0 else 0
        p, lo, hi = wilson_ci(abl_success, abl_total)

        # Legit success rate
        legit_results = [r.get("legit_succeeded") for r in abl_results if r.get("legit_succeeded") is not None]
        legit_rate = sum(1 for l in legit_results if l) / len(legit_results) if legit_results else 0

        # Fisher's test vs full defense (shows degradation from removing component)
        fisher_vs_defended = fishers_exact_test(abl_attacks, defended_attacks)
        # Fisher's test vs baseline (shows ablation still helps vs no defense)
        fisher_vs_baseline = fishers_exact_test(baseline_attacks, abl_attacks)

        results.append({
            "config": dirname,
            "n_attacks": abl_total,
            "n_conversations": len(abl_results),
            "asr": abl_asr,
            "ci_95": [round(lo, 4), round(hi, 4)],
            "legit_success_rate": legit_rate,
            "fisher_p_vs_defended": fisher_vs_defended["p_value"],
            "fisher_p_vs_baseline": fisher_vs_baseline["p_value"],
        })

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def print_report(analysis: dict):
    """Pretty-print the analysis report."""
    print("\n" + "=" * 70)
    print("  MEMGUARD-SECURE STATISTICAL ANALYSIS REPORT")
    print("=" * 70)

    # Overall ASR
    print("\n── Overall Attack Success Rate ──")
    for label in ["baseline", "defended"]:
        d = analysis.get(f"{label}_summary", {})
        if d:
            asr = d["asr"]
            ci = d["ci_95"]
            n = d["total"]
            print(f"  {label:12s}: ASR = {asr:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  (n={n})")

    # Per-scenario
    print("\n── Per-Scenario ASR ──")
    for label in ["baseline", "defended"]:
        by_sc = analysis.get(f"{label}_by_scenario", {})
        if by_sc:
            print(f"  {label}:")
            for sc, d in sorted(by_sc.items()):
                print(f"    {sc:<20s}: ASR = {d['asr']:.3f}  [{d['ci_95'][0]:.3f}, {d['ci_95'][1]:.3f}]  (n={d['total']})")

    # Per-attack-type
    print("\n── Per-Attack-Type ASR ──")
    for label in ["baseline", "defended"]:
        by_type = analysis.get(f"{label}_by_attack_type", {})
        if by_type:
            print(f"  {label}:")
            for atype, d in sorted(by_type.items()):
                print(f"    {atype:<25s}: ASR = {d['asr']:.3f}  [{d['ci_95'][0]:.3f}, {d['ci_95'][1]:.3f}]  "
                      f"({d['succeeded']}/{d['total']})")

    # Fisher's exact
    fisher = analysis.get("fisher_exact", {})
    if fisher:
        print(f"\n── Fisher's Exact Test ──")
        print(f"  H0: defense has no effect")
        print(f"  p-value = {fisher['p_value']:.6f}")
        print(f"  Significant at p<0.05: {fisher['significant_005']}")
        print(f"  Significant at p<0.01: {fisher['significant_001']}")
        print(f"  Odds ratio: {fisher['odds_ratio']:.3f}")

    # McNemar's
    mcnemar = analysis.get("mcnemar", {})
    if mcnemar:
        print(f"\n── McNemar's Test (paired) ──")
        print(f"  Matched pairs: {mcnemar['matched_pairs']}")
        b = mcnemar.get("discordant_b_base_corrupt_def_clean", mcnemar.get("discordant_b", 0))
        c = mcnemar.get("discordant_c_base_clean_def_corrupt", mcnemar.get("discordant_c", 0))
        print(f"  Discordant: baseline-corrupt/defended-clean = {b}")
        print(f"  Discordant: baseline-clean/defended-corrupt = {c}")
        print(f"  p-value = {mcnemar['p_value']:.6f}")

    # Per-voter accuracy
    voter = analysis.get("per_voter_accuracy", {})
    if voter:
        print(f"\n── Per-Voter Accuracy (defended runs) ──")
        for vname, vdata in sorted(voter.items()):
            print(f"  {vname:<30s}: {vdata['accuracy']:.3f}  ({vdata['correct']}/{vdata['total']})")
            for atype, td in sorted(vdata.get("by_attack_type", {}).items()):
                print(f"    {atype:<27s}: {td['accuracy']:.3f}  ({td['correct']}/{td['total']})")

    # Latency
    lat = analysis.get("latency", {})
    if lat and "mean_ms" in lat:
        print(f"\n── Latency Overhead ──")
        print(f"  Mean:   {lat['mean_ms']:>8.0f} ms")
        print(f"  Median: {lat['median_ms']:>8.0f} ms")
        print(f"  Std:    {lat['std_ms']:>8.0f} ms")
        print(f"  P25:    {lat['p25_ms']:>8.0f} ms")
        print(f"  P75:    {lat['p75_ms']:>8.0f} ms")
        print(f"  P95:    {lat['p95_ms']:>8.0f} ms")
        print(f"  n =     {lat['n_measurements']}")

    # Ablation
    ablation = analysis.get("ablation", [])
    if ablation:
        print(f"\n── Ablation Study ──")
        print(f"  {'Config':<30s} {'ASR':>8} {'CI 95%':>18} {'Legit':>8} {'p(Fisher)':>10}")
        print(f"  {'-'*78}")
        for a in ablation:
            ci_str = f"[{a['ci_95'][0]:.3f}, {a['ci_95'][1]:.3f}]"
            print(f"  {a['config']:<30s} {a['asr']:>8.3f} {ci_str:>18} {a['legit_success_rate']:>8.3f} {a['fisher_p_vs_baseline']:>10.4f}")

    # Legit success
    print(f"\n── Legitimate Update Success Rate ──")
    for label in ["baseline", "defended"]:
        ls = analysis.get(f"{label}_legit_rate")
        if ls is not None:
            print(f"  {label:12s}: {ls:.3f}")

    print("\n" + "=" * 70)


def run_analysis(baseline_dir: str, defended_dir: str, ablation_dirs: list[str] = None) -> dict:
    """Run full analysis and return results dict."""
    baseline_results = load_results(baseline_dir)
    defended_results = load_results(defended_dir)

    baseline_attacks = extract_attack_outcomes(baseline_results)
    defended_attacks = extract_attack_outcomes(defended_results)

    analysis = {}

    # Overall ASR with CI
    for label, attacks in [("baseline", baseline_attacks), ("defended", defended_attacks)]:
        n_success = sum(1 for a in attacks if a["corrupted"])
        n_total = len(attacks)
        p, lo, hi = wilson_ci(n_success, n_total)
        analysis[f"{label}_summary"] = {
            "asr": p, "ci_95": [round(lo, 4), round(hi, 4)],
            "succeeded": n_success, "total": n_total,
        }

    # Per-scenario
    for label, attacks in [("baseline", baseline_attacks), ("defended", defended_attacks)]:
        by_sc = defaultdict(list)
        for a in attacks:
            by_sc[a["scenario"]].append(a)
        sc_summary = {}
        for sc, sc_attacks in sorted(by_sc.items()):
            n_s = sum(1 for a in sc_attacks if a["corrupted"])
            n_t = len(sc_attacks)
            p, lo, hi = wilson_ci(n_s, n_t)
            sc_summary[sc] = {"asr": p, "ci_95": [round(lo, 4), round(hi, 4)],
                              "succeeded": n_s, "total": n_t}
        analysis[f"{label}_by_scenario"] = sc_summary

    # Per-attack-type
    for label, attacks in [("baseline", baseline_attacks), ("defended", defended_attacks)]:
        analysis[f"{label}_by_attack_type"] = attack_type_breakdown(attacks, label)

    # Legit success rates
    for label, results in [("baseline", baseline_results), ("defended", defended_results)]:
        legits = [r.get("legit_succeeded") for r in results if r.get("legit_succeeded") is not None]
        rate = sum(1 for l in legits if l) / len(legits) if legits else 0
        analysis[f"{label}_legit_rate"] = rate

    # Fisher's exact
    analysis["fisher_exact"] = fishers_exact_test(baseline_attacks, defended_attacks)

    # McNemar's
    analysis["mcnemar"] = mcnemars_test(baseline_attacks, defended_attacks)

    # Per-voter accuracy: prefer raw voter_data (all invocations); fall back to per_turn
    pv = per_voter_accuracy_raw(defended_results)
    if pv is None:
        pv = per_voter_accuracy(defended_attacks)
    analysis["per_voter_accuracy"] = pv

    # Latency
    analysis["latency"] = latency_analysis(defended_results)

    # Ablation
    if ablation_dirs:
        analysis["ablation"] = ablation_analysis(ablation_dirs, baseline_attacks, defended_attacks)

    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical analysis for MemGuard-Secure")
    parser.add_argument("--baseline", required=True, help="Directory with baseline results")
    parser.add_argument("--defended", required=True, help="Directory with defended results")
    parser.add_argument("--ablation", nargs="*", default=[], help="Directories with ablation results")
    parser.add_argument("--output", default=None, help="Save analysis JSON to this path")
    args = parser.parse_args()

    analysis = run_analysis(args.baseline, args.defended, args.ablation)
    print_report(analysis)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to: {args.output}")
