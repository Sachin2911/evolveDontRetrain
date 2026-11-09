"""Analyse the gated-residual run (9 Nov 2026) from a mirrored run directory.

    python SafeBlock/experiments/gated_analyse.py --run SafeBlock/outputs/residual-gated-20261109/host [--quantile 0.75]

Phases are subdirectories: calibration/, parity/, authority/, gated_extra/. Each job directory
holds attempt-XX/summary.json and steps.jsonl. Writes analysis.json and analysis.md in --run.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_jobs(phase_dir):
    out = {}
    if not phase_dir.exists():
        return out
    for job_dir in sorted(phase_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        for attempt in sorted(job_dir.glob("attempt-*")):
            s = attempt / "summary.json"
            if s.exists():
                out[job_dir.name] = {"summary": json.loads(s.read_text()), "dir": attempt}
                break
        else:
            errors = [a / "worker-error.json" for a in sorted(job_dir.glob("attempt-*")) if (a / "worker-error.json").exists()]
            out[job_dir.name] = {"summary": None, "dir": job_dir, "error": json.loads(errors[-1].read_text())["error"] if errors else "incomplete (no summary, no error)"}
    return out


def rows(attempt_dir):
    return [json.loads(l) for l in (attempt_dir / "steps.jsonl").read_text().splitlines() if l.strip()]


def kind(summary, last_row):
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


def record(job_id, entry):
    s = entry["summary"]
    r = rows(entry["dir"])
    plan_rows = [x for x in r if x.get("new_plan")]
    return {
        "id": job_id, "seed": s["job"]["seed"], "arm": None, "genotype_seed": s["job"].get("genotype_seed"),
        "outcome": s["outcome"], "kind": kind(s, r[-1]), "safe_success": bool(s["safe_success"]),
        "sustained": bool(s.get("sustained_safe_success")), "violation": bool(s["violation"]),
        "post_success_violation": bool(s.get("post_success_violation")), "steps": s["steps"], "plans": s["plans"],
        "changed_blocks": s.get("adapter_blocks_changed", 0), "gripper_overrides": s.get("adapter_gripper_overrides", 0),
        "gate_blocks": s.get("gate_blocks", 0), "gate_open_blocks": s.get("gate_open_blocks", 0),
        "failure_at_plans": [x["failure_prediction"] for x in plan_rows if "failure_prediction" in x],
        "failure_max": s.get("failure_prediction_max"), "min_q": s.get("next_min_q_min"),
        "startup_seconds": s.get("startup_seconds"), "episode_seconds": s.get("episode_seconds"),
        "action_hashes": [json.dumps(x["executed_normalized_action"]) for x in r],
        "actions": [x["executed_normalized_action"] for x in r],
        "camera_hashes": [x["camera_sha256"] for x in r],
    }


def arm_table(label, recs, bare_by_seed):
    if not recs:
        return {"label": label, "episodes": 0}
    viol = sum(r["violation"] for r in recs)
    kinds = Counter(r["kind"] for r in recs)
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["genotype_seed"]].append(r)
    cwbf = sum(1 for r in recs if r["safe_success"] and bare_by_seed.get(r["seed"]) is not None and not bare_by_seed[r["seed"]]["safe_success"])
    fwbc = sum(1 for r in recs if not r["safe_success"] and bare_by_seed.get(r["seed"]) is not None and bare_by_seed[r["seed"]]["safe_success"])
    vadd = sum(1 for r in recs if r["violation"] and bare_by_seed.get(r["seed"]) is not None and not bare_by_seed[r["seed"]]["violation"])
    vrem = sum(1 for r in recs if not r["violation"] and bare_by_seed.get(r["seed"]) is not None and bare_by_seed[r["seed"]]["violation"])
    gate_total = sum(r["gate_blocks"] for r in recs)
    return {
        "label": label, "episodes": len(recs), "genotypes": len(by_g),
        "safe_success": sum(r["safe_success"] for r in recs), "sustained": sum(r["sustained"] for r in recs),
        "violation_episodes": viol, "violation_fraction": round(viol / len(recs), 3),
        "middle_drops_incl_post_success": kinds["middle_drop"] + sum(r["post_success_violation"] for r in recs),
        "boundary_exits": kinds["boundary_exit"], "top_drops": kinds["top_drop"], "top_displaced": kinds["top_displaced"],
        "timeouts": kinds["timeout"], "kinds": dict(kinds),
        "completes_where_base_fails": cwbf, "fails_where_base_completes": fwbc,
        "violations_added": vadd, "violations_removed": vrem,
        "gate_open_fraction": round(sum(r["gate_open_blocks"] for r in recs) / gate_total, 3) if gate_total else None,
        "mean_changed_blocks": round(sum(r["changed_blocks"] for r in recs) / len(recs), 2),
        "mean_steps": round(float(np.mean([r["steps"] for r in recs])), 1),
        "genotypes_all_violating": sum(1 for v in by_g.values() if all(r["violation"] for r in v)),
        "genotypes_with_any_success": sum(1 for v in by_g.values() if any(r["safe_success"] for r in v)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.75)
    args = parser.parse_args()
    R = {}
    md = ["# Gated residual run: analysis", ""]

    # Calibration
    cal = load_jobs(args.run / "calibration")
    bare = {}
    failed = {k: v["error"] for k, v in cal.items() if v["summary"] is None}
    for k, v in cal.items():
        if v["summary"] is not None:
            rec = record(k, v)
            rec["arm"] = "bare"
            bare[rec["seed"]] = rec
    if bare:
        recs = sorted(bare.values(), key=lambda r: r["seed"])
        pooled = np.array([f for r in recs for f in r["failure_at_plans"]])
        onset = float(np.quantile(pooled, args.quantile)) if len(pooled) else None
        open_frac = float((pooled > onset).mean()) if onset is not None else None
        violated = [r["seed"] for r in recs if r["violation"]]
        R["calibration"] = {
            "episodes": len(recs), "failed_jobs": failed,
            "safe_success": sum(r["safe_success"] for r in recs), "sustained": sum(r["sustained"] for r in recs),
            "violation": sum(r["violation"] for r in recs), "kinds": dict(Counter(r["kind"] for r in recs)),
            "violated_seeds": violated, "first_four_violated": violated[:4],
            "failure_probability_at_plans": {"n_blocks": int(len(pooled)),
                "quantiles": {q: float(np.quantile(pooled, q)) for q in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)} if len(pooled) else {}},
            "gate_onset": {"quantile": args.quantile, "threshold": onset, "bare_open_fraction": open_frac},
            "per_episode_bare_open_fraction": {r["seed"]: round(float(np.mean(np.array(r["failure_at_plans"]) > onset)), 3) for r in recs} if onset is not None else {},
            "startup_seconds_median": float(np.median([r["startup_seconds"] for r in recs if r["startup_seconds"]])),
            "episode_seconds_median": float(np.median([r["episode_seconds"] for r in recs if r["episode_seconds"]])),
            "records": [{k: r[k] for k in ("seed", "outcome", "kind", "safe_success", "sustained", "violation", "steps", "plans", "failure_max", "min_q", "episode_seconds")} for r in recs],
        }
        c = R["calibration"]
        md += ["## Bare-planner calibration", "", f"{c['episodes']} fresh Hard starts: safe success {c['safe_success']}, sustained {c['sustained']}, violation {c['violation']} ({c['violation']/c['episodes']:.3f}). Kinds: {c['kinds']}. Failed jobs: {len(failed)}.", "",
               f"Failure-head probability at the {c['failure_probability_at_plans']['n_blocks']} replan controls, quantiles: " + ", ".join(f"q{q}={v:.4f}" for q, v in c['failure_probability_at_plans']['quantiles'].items()), "",
               f"**Gate onset: quantile {args.quantile} -> threshold {onset:.6f}; the gate would open on {open_frac:.1%} of bare blocks.** Violated seeds (ascending): {violated}; panel = first four: {violated[:4]}.", "",
               "| seed | outcome kind | safe | sustained | violation | steps | plans | max failure p | min Q | s |", "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
        for r in recs:
            md.append(f"| {r['seed']} | {r['kind']} | {int(r['safe_success'])} | {int(r['sustained'])} | {int(r['violation'])} | {r['steps']} | {r['plans']} | {r['failure_max']:.4f} | {r['min_q']:.3f} | {r['episode_seconds']:.0f} |")
        md.append("")

    # Parity
    par = load_jobs(args.run / "parity")
    if par:
        P = []
        by_seed = defaultdict(dict)
        for k, v in par.items():
            if v["summary"] is None:
                P.append({"id": k, "error": v["error"]}); continue
            rec = record(k, v)
            by_seed[rec["seed"]][k.rsplit("-", 1)[0] if not k.startswith("bare") else "bare"] = rec
        for seed, d in sorted(by_seed.items()):
            ref = d.get("bare") or bare.get(seed)
            for label, rec in d.items():
                if label == "bare" or ref is None:
                    continue
                n = min(len(ref["action_hashes"]), len(rec["action_hashes"]))
                max_diff = max((max(abs(a - b) for a, b in zip(ref["actions"][i], rec["actions"][i])) for i in range(n)), default=0.0)
                P.append({"seed": seed, "pair": label, "same_outcome_and_length": ref["outcome"] == rec["outcome"] and ref["steps"] == rec["steps"],
                          "identical_actions": f"{sum(ref['actions'][i] == rec['actions'][i] for i in range(n))}/{max(len(ref['action_hashes']), len(rec['action_hashes']))}",
                          "max_action_diff": max_diff,
                          "signed_zero_only_diffs": sum(ref['action_hashes'][i] != rec['action_hashes'][i] for i in range(n)),
                          "identical_cameras": f"{sum(ref['camera_hashes'][i] == rec['camera_hashes'][i] for i in range(n))}/{max(len(ref['camera_hashes']), len(rec['camera_hashes']))}",
                          "changed_blocks": rec["changed_blocks"], "gate_open_blocks": rec["gate_open_blocks"], "gate_blocks": rec["gate_blocks"]})
        R["parity"] = P
        md += ["## Parity (gate forced closed / zero genotype vs bare, same host)", "", "| seed | pair | same outcome+length | identical actions (numeric) | max action diff | -0.0/0.0 text-only diffs | identical cameras | changed blocks | gate open/blocks |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for p in P:
            if "error" in p:
                md.append(f"| {p['id']} | FAILED | {p['error']} | | | | |")
            else:
                md.append(f"| {p['seed']} | {p['pair']} | {p['same_outcome_and_length']} | {p['identical_actions']} | {p['max_action_diff']:.1e} | {p['signed_zero_only_diffs']} | {p['identical_cameras']} | {p['changed_blocks']} | {p['gate_open_blocks']}/{p['gate_blocks']} |")
        md.append("")

    # Authority
    for phase in ("authority", "gated_extra", "control"):
        auth = load_jobs(args.run / phase)
        if not auth:
            continue
        recs, errors = [], {}
        for k, v in auth.items():
            if v["summary"] is None:
                errors[k] = v["error"]; continue
            rec = record(k, v)
            rec["arm"] = k.split("-g")[0]
            recs.append(rec)
        arms = sorted({r["arm"] for r in recs})
        seeds = sorted({r["seed"] for r in recs})
        tables = {a: arm_table(a, [r for r in recs if r["arm"] == a], bare) for a in arms}
        bare_panel = arm_table("bare (same starts)", [bare[s] for s in seeds if s in bare], {})
        R[phase] = {"errors": errors, "arms": tables, "bare_panel": bare_panel,
                    "per_seed": {a: {s: {"safe": sum(r["safe_success"] for r in recs if r["arm"] == a and r["seed"] == s), "viol": sum(r["violation"] for r in recs if r["arm"] == a and r["seed"] == s), "n": sum(1 for r in recs if r["arm"] == a and r["seed"] == s)} for s in seeds} for a in arms},
                    "per_genotype": {a: {g: {"safe": sum(r["safe_success"] for r in recs if r["arm"] == a and r["genotype_seed"] == g), "viol": sum(r["violation"] for r in recs if r["arm"] == a and r["genotype_seed"] == g), "n": sum(1 for r in recs if r["arm"] == a and r["genotype_seed"] == g)} for g in sorted({r["genotype_seed"] for r in recs if r["arm"] == a})} for a in arms},
                    "records": [{k: r[k] for k in ("id", "arm", "seed", "genotype_seed", "kind", "safe_success", "sustained", "violation", "steps", "changed_blocks", "gate_blocks", "gate_open_blocks", "episode_seconds")} for r in recs]}
        failed_real = {k: v for k, v in errors.items() if not v.startswith("incomplete")}
        md += [f"## {phase}: residual arms on {len(seeds)} starts {seeds} (failed jobs: {len(failed_real)}, incomplete: {len(errors) - len(failed_real)})", "",
               (("Failures: " + "; ".join(f"{k}: {v[:120]}" for k, v in failed_real.items())) if failed_real else ""), "",
               "| arm | episodes | genotypes | safe success | sustained | violation episodes | violation fraction | middle drops (incl. post) | boundary exits | top drops | top displaced | timeouts | completes where bare fails | fails where bare completes | viol added | viol removed | gate open frac | mean changed blocks | mean steps |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for a in ["bare (same starts)"] + arms:
            t = bare_panel if a == "bare (same starts)" else tables[a]
            if not t.get("episodes"):
                continue
            md.append(f"| {t['label']} | {t['episodes']} | {t['genotypes']} | {t['safe_success']} | {t['sustained']} | {t['violation_episodes']} | {t['violation_fraction']} | {t['middle_drops_incl_post_success']} | {t['boundary_exits']} | {t['top_drops']} | {t['top_displaced']} | {t['timeouts']} | {t['completes_where_base_fails']} | {t['fails_where_base_completes']} | {t['violations_added']} | {t['violations_removed']} | {t['gate_open_fraction']} | {t['mean_changed_blocks']} | {t['mean_steps']} |")
        md += ["", "Per start (safe successes / violations / episodes):", "", "| start | bare | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
        for s in seeds:
            b = bare.get(s)
            md.append(f"| {s} | {b['kind'] if b else '-'} | " + " | ".join(f"{R[phase]['per_seed'][a][s]['safe']}/{R[phase]['per_seed'][a][s]['viol']}/{R[phase]['per_seed'][a][s]['n']}" for a in arms) + " |")
        md.append("")

    (args.run / "analysis.json").write_text(json.dumps(R, indent=1, default=str))
    (args.run / "analysis.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
