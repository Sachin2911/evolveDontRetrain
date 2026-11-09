"""Write the job lists for the gated-residual run (9 Nov 2026) as JSONL.

    python SafeBlock/experiments/gated_make_jobs.py calibration --seeds 4330000-4330023 --out jobs/calibration.jsonl
    python SafeBlock/experiments/gated_make_jobs.py parity --seeds A,B --threshold T --out jobs/parity.jsonl
    python SafeBlock/experiments/gated_make_jobs.py authority --seeds A,B,C,D --genotypes 5000000-5000011 --threshold T --out jobs/authority.jsonl
"""

import argparse
import json
from pathlib import Path

RESIDUAL = ["--adapter", "residual", "--bound", "0.1", "--init-scale", "0.2"]


def parse_range(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibration", "parity", "authority", "gated_extra"))
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--genotypes", default="5000000-5000011")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--arms", default="gated,ungated", help="authority arms: gated, ungated, gated_gripper, ungated_gripper (prefix 'tiny_' = control with --bound)")
    parser.add_argument("--bound", default="0.1", help="residual bound for authority/gated_extra jobs (default 0.1)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = parse_range(args.seeds)
    jobs = []
    if args.phase == "calibration":
        for seed in seeds:
            jobs.append({"id": f"bare-{seed}", "seed": seed, "arm": "bare", "args": []})
    elif args.phase == "parity":
        for seed in seeds:
            jobs.append({"id": f"bare-{seed}", "seed": seed, "arm": "bare", "args": []})
            jobs.append({"id": f"gated-closed-g5000000-{seed}", "seed": seed, "arm": "gated_forced_closed",
                         "args": RESIDUAL + ["--genotype-seed", "5000000", "--gripper-bound", "0.0",
                                             "--gate-threshold", str(args.threshold), "--gate-force-closed"]})
            jobs.append({"id": f"gated-zero-{seed}", "seed": seed, "arm": "gated_zero_genotype",
                         "args": RESIDUAL + ["--zero-genotype", "--gripper-bound", "0.0", "--gate-threshold", str(args.threshold)]})
    else:
        genotypes = parse_range(args.genotypes)
        # Interleave seeds so every lane mixes starts; genotype-major so partial runs keep whole genotypes.
        for g in genotypes:
            for seed in seeds:
                for arm in args.arms.split(","):
                    gripper = "2.0" if arm.endswith("gripper") else "0.0"
                    a = ["--adapter", "residual", "--bound", args.bound, "--init-scale", "0.2", "--genotype-seed", str(g), "--gripper-bound", gripper]
                    if arm.replace("tiny_", "").startswith("gated"):
                        a = a + ["--gate-threshold", str(args.threshold)]
                    jobs.append({"id": f"{arm}-g{g}-{seed}", "seed": seed, "arm": arm, "genotype_seed": g, "args": a})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"{len(jobs)} jobs -> {args.out}")


if __name__ == "__main__":
    main()
