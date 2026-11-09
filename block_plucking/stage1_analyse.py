"""Analyse Stage 1 records into Stage1Results.md and stage1-results.json.

    python3 SafeBlock/experiments/stage1_analyse.py --root SafeBlock/outputs/stage1-20261102 --output SafeBlock
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def load_jobs(root, probe):
    out = {}
    for job_dir in sorted((root / probe).glob("*")):
        if not job_dir.is_dir():
            continue
        attempts = sorted(job_dir.glob("attempt-*"))
        done = [a for a in attempts if (a / "summary.json").exists()]
        if done:
            out[job_dir.name] = (done[-1], json.loads((done[-1] / "summary.json").read_text()))
    return out


def rows(attempt):
    return [json.loads(l) for l in (attempt / "steps.jsonl").open()]


def violated(summary):
    """Frozen definition: a violation at any control, benchmark or continuation."""
    return bool(summary["violation"]) or bool(summary.get("post_success_violation"))


def failure_kind(summary, last_row):
    if summary["outcome"] != "physical_failure":
        return summary["outcome"]
    t = last_row["termination_terms"]
    if last_row["after"]["top_displaced"]:
        return "top_displaced"
    if t["cube_2_dropping"]:
        return "middle_drop"
    if t["cube_3_dropping"]:
        return "top_drop"
    return "other_physical"


def compare_traces(a, b):
    """Per-control comparison of two episodes; returns a dict of agreement counts."""
    ra, rb = rows(a), rows(b)
    n = min(len(ra), len(rb))
    same_action = sum(ra[i]["executed_normalized_action"] == rb[i]["executed_normalized_action"] for i in range(n))
    same_camera = sum(ra[i]["camera_sha256"] == rb[i]["camera_sha256"] for i in range(n))
    same_cubes = sum(ra[i]["after"]["cube_positions"] == rb[i]["after"]["cube_positions"] for i in range(n))
    max_action_diff = max(
        (max(abs(x - y) for x, y in zip(ra[i]["executed_normalized_action"], rb[i]["executed_normalized_action"])) for i in range(n)),
        default=0.0,
    )
    first_action_diff = next((i for i in range(n) if ra[i]["executed_normalized_action"] != rb[i]["executed_normalized_action"]), None)
    first_camera_diff = next((i for i in range(n) if ra[i]["camera_sha256"] != rb[i]["camera_sha256"]), None)
    return {
        "controls_a": len(ra), "controls_b": len(rb), "compared": n,
        "same_action": same_action, "same_camera": same_camera, "same_cube_positions": same_cubes,
        "first_action_diff": first_action_diff, "first_camera_diff": first_camera_diff,
        "max_action_abs_diff": max_action_diff,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    R = {}
    md = ["# Stage 1 results", "", f"Records: `{args.root}`. Protocol: Stage1.md. Development seeds only.", ""]

    # ---------------- Parity ----------------
    P = load_jobs(args.root, "S1P")
    parity = {"episodes": len(P), "zero_adapter_changed_blocks": 0, "zero_adapter_blocks": 0, "pairs": []}
    for name, (att, s) in P.items():
        if s.get("adapter"):
            parity["zero_adapter_blocks"] += s["adapter_blocks"] if s["adapter"]["kind"] == "residual" else s["plans"]
            parity["zero_adapter_changed_blocks"] += s["adapter_blocks_changed"] + s.get("adapter_choice_changes", 0)
    seeds = sorted({s["job"]["seed"] for _, s in P.values()})
    for seed in seeds:
        bare = P.get(f"S1P-{seed}-bare")
        for kind in ("residual-zero", "selection-zero", "bare-repeat"):
            other = P.get(f"S1P-{seed}-{kind}")
            if bare and other:
                c = compare_traces(bare[0], other[0])
                c.update(seed=seed, pair=kind, outcome_bare=bare[1]["outcome"], outcome_other=other[1]["outcome"], same_outcome=bare[1]["outcome"] == other[1]["outcome"] and bare[1]["steps"] == other[1]["steps"])
                parity["pairs"].append(c)
    R["parity"] = parity
    md += ["## Gate 1: parity", "", f"Zero-weight adapters changed {parity['zero_adapter_changed_blocks']} of {parity['zero_adapter_blocks']} blocks in-process (must be 0).", "",
           "| seed | pair | same outcome and length | identical actions | identical camera hashes | first action difference | max action diff |", "|---|---|---|---:|---:|---:|---:|"]
    for c in parity["pairs"]:
        md.append(f"| {c['seed']} | {c['pair']} | {c['same_outcome']} | {c['same_action']}/{c['compared']} | {c['same_camera']}/{c['compared']} | {c['first_action_diff']} | {c['max_action_abs_diff']:.2e} |")
    md.append("")

    # ---------------- Sweep ----------------
    S = load_jobs(args.root, "S1S")
    sweep = {"bare": {}, "residual": defaultdict(list), "selection": []}
    for name, (att, s) in S.items():
        last = rows(att)[-1]
        rec = {"id": name, "seed": s["job"]["seed"], "outcome": s["outcome"], "kind": failure_kind(s, last), "safe_success": bool(s["safe_success"]), "sustained": bool(s["sustained_safe_success"]), "violation": violated(s), "benchmark_violation": bool(s["violation"]), "post_success_violation": bool(s.get("post_success_violation")), "steps": s["steps"], "changed_blocks": s.get("adapter_blocks_changed", 0), "gripper_overrides": s.get("adapter_gripper_overrides", 0), "choice_changes": s.get("adapter_choice_changes", 0)}
        if not s.get("adapter"):
            sweep["bare"][s["job"]["seed"]] = rec
        elif s["adapter"]["kind"] == "residual":
            key = ("gripper" if s["adapter"]["gripper_bound"] > 0 else "arm_only", s["adapter"]["bound"])
            rec.update(genotype_seed=s["adapter"]["genotype_seed"], bound=s["adapter"]["bound"], gripper_bound=s["adapter"]["gripper_bound"])
            sweep["residual"][key].append(rec)
        else:
            rec.update(genotype_seed=s["adapter"]["genotype_seed"])
            sweep["selection"].append(rec)

    def gate(recs, label):
        if not recs:
            return {"label": label, "episodes": 0}
        bare = sweep["bare"]
        by_g = defaultdict(list)
        for r in recs:
            by_g[r["genotype_seed"]].append(r)
        completes_where_base_fails = sum(1 for r in recs if r["safe_success"] and bare.get(r["seed"], {}).get("safe_success") is False)
        fails_where_base_completes = sum(1 for r in recs if not r["safe_success"] and bare.get(r["seed"], {}).get("safe_success") is True)
        viol = sum(r["violation"] for r in recs)
        kinds = defaultdict(int)
        for r in recs:
            kinds[r["kind"]] += 1
        middle_drops = sum(1 for r in recs if r["kind"] == "middle_drop") + sum(1 for r in recs if r["post_success_violation"])
        g = {
            "label": label, "episodes": len(recs), "genotypes": len(by_g),
            "safe_success": sum(r["safe_success"] for r in recs), "sustained": sum(r["sustained"] for r in recs),
            "violation_episodes": viol, "violation_fraction": round(viol / len(recs), 3),
            "middle_drops_incl_post_success": middle_drops, "kinds": dict(kinds),
            "completes_where_base_fails": completes_where_base_fails, "fails_where_base_completes": fails_where_base_completes,
            "genotypes_with_any_success": sum(1 for v in by_g.values() if any(r["safe_success"] for r in v)),
            "genotypes_all_violating": sum(1 for v in by_g.values() if all(r["violation"] for r in v)),
            "mean_changed_blocks": round(sum(r["changed_blocks"] + r["choice_changes"] for r in recs) / len(recs), 1),
            "gripper_overrides_total": sum(r["gripper_overrides"] for r in recs),
        }
        g["gate_both_ways"] = completes_where_base_fails > 0 and fails_where_base_completes > 0
        g["gate_not_degenerate"] = g["violation_fraction"] < 0.9
        g["gate_signal"] = g["safe_success"] > 0 and viol > 0
        g["gate_sustained_nonzero"] = g["sustained"] > 0
        g["passes"] = all((g["gate_both_ways"], g["gate_not_degenerate"], g["gate_signal"], g["gate_sustained_nonzero"]))
        return g

    residual_all = [r for v in sweep["residual"].values() for r in v]
    gripper_half = [r for (h, b), v in sweep["residual"].items() if h == "gripper" for r in v]
    arm_half = [r for (h, b), v in sweep["residual"].items() if h == "arm_only" for r in v]
    G = {
        "bare": {"episodes": len(sweep["bare"]), "safe_success": sum(r["safe_success"] for r in sweep["bare"].values()), "violation": sum(r["violation"] for r in sweep["bare"].values()), "per_seed": {str(k): v["kind"] for k, v in sorted(sweep["bare"].items())}},
        "residual": gate(residual_all, "residual (all)"),
        "residual_gripper": gate(gripper_half, "residual, gripper authority"),
        "residual_arm_only": gate(arm_half, "residual, arm only"),
        "residual_by_cell": {f"{h}-B{b}": gate(v, f"{h} B={b}") for (h, b), v in sorted(sweep["residual"].items())},
        "selection": gate(sweep["selection"], "selection"),
    }
    gr, ar = G["residual_gripper"], G["residual_arm_only"]
    if gr.get("episodes") and ar.get("episodes"):
        G["gripper_rule"] = {
            "gripper_middle_drops": gr["middle_drops_incl_post_success"], "arm_only_middle_drops": ar["middle_drops_incl_post_success"],
            "gripper_violations": gr["violation_episodes"], "arm_only_violations": ar["violation_episodes"],
            "keep_gripper": gr["middle_drops_incl_post_success"] < ar["middle_drops_incl_post_success"] and gr["violation_episodes"] <= ar["violation_episodes"],
        }
    R["sweep"] = {"gates": G, "records": {"residual": residual_all, "selection": sweep["selection"], "bare": list(sweep["bare"].values())}}
    md += ["## Gate 2: authority sweep", "", f"Bare planner on the 4 sweep starts: {G['bare']['safe_success']} completions, {G['bare']['violation']} violations ({G['bare']['per_seed']}).", "",
           "| interface | episodes | genotypes | safe success | sustained | violation episodes | violation fraction | middle drops (incl. post-success) | completes where base fails | fails where base completes | genotypes all-violating | passes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key in ("residual", "residual_gripper", "residual_arm_only", "selection"):
        g = G[key]
        if g.get("episodes"):
            md.append(f"| {g['label']} | {g['episodes']} | {g['genotypes']} | {g['safe_success']} | {g['sustained']} | {g['violation_episodes']} | {g['violation_fraction']} | {g['middle_drops_incl_post_success']} | {g['completes_where_base_fails']} | {g['fails_where_base_completes']} | {g['genotypes_all_violating']} | {g['passes']} |")
    for key, g in G["residual_by_cell"].items():
        if g.get("episodes"):
            md.append(f"| {g['label']} | {g['episodes']} | {g['genotypes']} | {g['safe_success']} | {g['sustained']} | {g['violation_episodes']} | {g['violation_fraction']} | {g['middle_drops_incl_post_success']} | {g['completes_where_base_fails']} | {g['fails_where_base_completes']} | {g['genotypes_all_violating']} | {g['passes']} |")
    if "gripper_rule" in G:
        gr_ = G["gripper_rule"]
        md += ["", f"Gripper rule: gripper half {gr_['gripper_middle_drops']} middle drops and {gr_['gripper_violations']} violation episodes; arm-only half {gr_['arm_only_middle_drops']} and {gr_['arm_only_violations']}. Keep gripper override: **{gr_['keep_gripper']}**."]
    md.append("")

    # ---------------- Calibration ----------------
    C = load_jobs(args.root, "S1C")
    cal = {"episodes": len(C)}
    if C:
        recs = []
        for name, (att, s) in C.items():
            recs.append({"seed": s["job"]["seed"], "outcome": s["outcome"], "kind": failure_kind(s, rows(att)[-1]) + ("+post_success_violation" if s.get("post_success_violation") else ""), "safe_success": bool(s["safe_success"]), "sustained": bool(s["sustained_safe_success"]), "violation": violated(s), "benchmark_violation": bool(s["violation"]), "steps": s["steps"]})
        n = len(recs)
        v = sum(r["violation"] for r in recs); ss = sum(r["safe_success"] for r in recs); su = sum(r["sustained"] for r in recs)
        kinds = defaultdict(int)
        for r in recs:
            kinds[r["kind"]] += 1
        p = v / n
        table = []
        for limit in (0.5, 0.375, 0.25, 0.125):
            k = math.floor(limit * 8 + 1e-9)
            table.append({"limit": limit, "max_violations_of_8": k, "p_base_looks_feasible": round(binom_cdf(k, 8, p), 3), "p_half_rate_looks_feasible": round(binom_cdf(k, 8, p / 2), 3), "p_quarter_rate_looks_feasible": round(binom_cdf(k, 8, p / 4), 3)})
        bv = sum(r["benchmark_violation"] for r in recs)
        cal.update(violation_rate=round(p, 3), violation_ci=wilson(v, n), benchmark_violation_rate=round(bv / n, 3), safe_success_rate=round(ss / n, 3), safe_success_ci=wilson(ss, n), sustained_rate=round(su / n, 3), sustained_ci=wilson(su, n), kinds=dict(kinds), feasibility_illusion_n8=table, records=recs)
        md += ["## Calibration: bare planner on 48 fresh starts", "", f"Violation rate at any control {cal['violation_rate']} (Wilson {cal['violation_ci']}; benchmark-only {cal['benchmark_violation_rate']}), safe success {cal['safe_success_rate']} {cal['safe_success_ci']}, sustained {cal['sustained_rate']} {cal['sustained_ci']}. Outcome kinds: {cal['kinds']}.", "",
               "| requested limit | max violations of 8 | P(base-rate candidate looks feasible) | P(half-rate candidate looks feasible) | P(quarter-rate looks feasible) |", "|---:|---:|---:|---:|---:|"]
        for t in table:
            md.append(f"| {t['limit']} | {t['max_violations_of_8']} | {t['p_base_looks_feasible']} | {t['p_half_rate_looks_feasible']} | {t['p_quarter_rate_looks_feasible']} |")
        md.append("")
    R["calibration"] = cal

    # ---------------- Reset test ----------------
    T = load_jobs(args.root, "S1R")
    reset = {"pairs": []}
    inproc = T.get("S1R-in-process")
    if inproc:
        att, s = inproc
        for ep in s["episodes"]:
            if ep.get("warmup"):
                continue
            fresh = T.get(f"S1R-{ep['seed']}-fresh")
            if fresh:
                c = compare_traces(fresh[0], att / ep["dir"])
                c.update(seed=ep["seed"], outcome_fresh=fresh[1]["outcome"], outcome_in_process=ep["outcome"], same_outcome=fresh[1]["outcome"] == ep["outcome"] and fresh[1]["steps"] == ep["steps"], fresh_episode_seconds=fresh[1]["episode_seconds"], fresh_elapsed=fresh[1]["elapsed_seconds"], in_process_episode_seconds=json.loads((att / ep["dir"] / "summary.json").read_text())["episode_seconds"])
                reset["pairs"].append(c)
        if reset["pairs"]:
            reset["all_identical"] = all(c["same_action"] == c["compared"] and c["same_camera"] == c["compared"] and c["same_outcome"] for c in reset["pairs"])
            reset["all_same_outcome"] = all(c["same_outcome"] for c in reset["pairs"])
            reset["mean_fresh_total_s"] = round(sum(c["fresh_elapsed"] for c in reset["pairs"]) / len(reset["pairs"]), 1)
            reset["mean_in_process_episode_s"] = round(sum(c["in_process_episode_seconds"] for c in reset["pairs"]) / len(reset["pairs"]), 1)
            reset["decision"] = "in-process everywhere" if reset["all_identical"] else ("in-process for training, fresh for selection and final" if reset["all_same_outcome"] else "fresh everywhere")
        md += ["## In-process resets", "", "| seed | same outcome and length | identical actions | identical cameras | first action diff | max action diff | fresh total s | in-process episode s |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for c in reset["pairs"]:
            md.append(f"| {c['seed']} | {c['same_outcome']} | {c['same_action']}/{c['compared']} | {c['same_camera']}/{c['compared']} | {c['first_action_diff']} | {c['max_action_abs_diff']:.2e} | {c['fresh_elapsed']:.0f} | {c['in_process_episode_seconds']:.0f} |")
        md += ["", f"Decision by the pre-stated rule: **{reset.get('decision')}**.", ""]
    R["reset"] = reset

    (args.output / "Stage1Results.md").write_text("\n".join(md) + "\n")
    (args.output / "outputs" / "stage1-20261102" / "stage1-results.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output / "outputs" / "stage1-20261102" / "stage1-results.json").write_text(json.dumps(R, indent=2, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
