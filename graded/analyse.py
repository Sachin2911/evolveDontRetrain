"""Graded study analysis: learning curves, stop rule, test tables, paired comparisons -> markdown.

  python -m graded.analyse <out_dir> [<out_dir> ...] --md graded/RESULTS.md

Each out_dir holds <adapter>-r<k>/state.json files and test-*.json from graded.search.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from graded import stoprule


def load_tests(dirs):
    per_case, table = {}, []
    for d in dirs:
        for p in sorted(Path(d).glob("test-*.json")):
            t = json.loads(p.read_text())
            table.extend(t["table"])
            for arm, rows in t["per_case"].items():
                per_case.setdefault(arm, {}).update({r["case_id"]: r for r in rows})
    return per_case, table


def paired(a, b):
    """Sign test on safe completion: cases where a is safe and b is not, vs the reverse."""
    common = sorted(set(a) & set(b))
    wins = sum(1 for c in common if a[c]["safe_completion"] and not b[c]["safe_completion"])
    losses = sum(1 for c in common if b[c]["safe_completion"] and not a[c]["safe_completion"])
    p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    return len(common), wins, losses, p


def env_of(cid):
    return cid.split("-")[0]


def level_of(cid):
    for part in cid.split("-"):
        if part.startswith("L") and part[1:].isdigit():
            return int(part[1:])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--prefix", default="", help="arm-name prefix for the paired comparisons, e.g. v2- for run 2")
    ap.add_argument("--title", default="Graded hazard study: results (4-hour run, 9 Nov 2026)")
    args = ap.parse_args()
    lines = [f"# {args.title}", ""]

    lines += ["## Searches", "", "| search | gens | free share | CMA trend rho (p) | random trend rho (p) | mean moved / null 95% | "
              "random draws beating native | stop rule at gen 8 | locked CMA (selection) | locked random | native |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    seen = set()  # locked states were copied to every box for the test, so each search appears in every out_dir
    for d in args.dirs:
        for p in sorted(Path(d).glob("*-r*/state.json")):
            s = json.loads(p.read_text())
            if not s.get("generations") or p.parent.name in seen:
                continue
            seen.add(p.parent.name)
            full = stoprule.check(p, None, 200)
            g8 = stoprule.check(p, 8, 200) if len(s["generations"]) >= 8 else None
            lk = s.get("locked", {})
            fmt = lambda r: f"{r['id']} S{r['S']} C{r['C']} harm{r['harm']}" if r else "-"
            ns = s.get("native_selection")
            lines.append(
                f"| {p.parent.name} | {full['generations']} | {full['free_share']:.2f} | {full['trend_cma'][0]:+.2f} ({full['trend_cma'][1]:.3f}) | "
                f"{full['trend_rnd'][0]:+.2f} ({full['trend_rnd'][1]:.3f}) | {full['moved']:.2f} / {full['null_q95']:.2f} | "
                f"{full['share_rnd_beating_native']:.2f} | {'pass' if g8 and g8['pass'] else ('fail' if g8 else 'n/a')} | "
                f"{fmt(lk.get('cma'))} | {fmt(lk.get('rnd'))} | {ns} |")
            lines.append("")
            curve = [(g["generation"], np.mean([stoprule.scalar(c) for c in g["cma"]]), np.mean([stoprule.scalar(c) for c in g["rnd"]]))
                     for g in s["generations"]]
            lines.append(f"Population mean score by generation ({p.parent.name}, CMA / random): " +
                         ", ".join(f"g{g}: {a:.0f}/{b:.0f}" for g, a, b in curve))
            lines.append("")

    per_case, table = load_tests(args.dirs)
    if table:
        lines += ["## Test panel (24 cases per environment; S = safe completions, C = completions, harm = native-safe cases lost)", ""]
        envs = sorted({r["env"] for r in table})
        arms = []
        for r in table:
            if r["arm"] not in arms:
                arms.append(r["arm"])
        for env in envs:
            lines += [f"### {env}", "", "| arm | all: S / C / harm / violations-depth | L1 S | L2 S | L3 S | L4 S (C) |", "|---|---|---|---|---|---|"]
            for arm in arms:
                rows = {r["level"]: r for r in table if r["arm"] == arm and r["env"] == env}
                if "all" not in rows:
                    continue
                a = rows["all"]
                lv = lambda L: f"{rows[L]['S']}/{rows[L]['n']}" if L in rows else "-"
                l4 = f"{rows[4]['S']}/{rows[4]['n']} ({rows[4]['C']})" if 4 in rows else "-"
                lines.append(f"| {arm} | {a['S']} / {a['C']} / {a['harm']} / {a['depth']:.1f} | {lv(1)} | {lv(2)} | {lv(3)} | {l4} |")
            lines.append("")

        lines += ["## Paired comparisons on safe completion (levels 1-3, sign test)", "",
                  "| comparison | env | cases | a safe, b not | b safe, a not | p |", "|---|---|---|---|---|---|"]

        def subset(arm, env):
            return {c: r for c, r in per_case.get(arm, {}).items() if env_of(c) == env and (level_of(c) or 0) <= 3}

        comps = []
        P = args.prefix
        for adapter in ("steer", "choose", "stack"):
            for r in (1, 2):
                comps.append((f"{P}{adapter}-cma-r{r}", f"{P}{adapter}-rnd-r{r}"))
                comps.append((f"{P}{adapter}-cma-r{r}", "native"))
                comps.append((f"{P}{adapter}-rnd-r{r}", "native"))
            for r in (1, 2):
                for u in range(3):
                    comps.append((f"{P}{adapter}-cma-r{r}", f"{P}{adapter}-untrained{u}"))
        for a, b in comps:
            if a not in per_case or b not in per_case:
                continue
            for env in sorted({env_of(c) for c in per_case[a]}):
                n, w, l, p = paired(subset(a, env), subset(b, env))
                if n:
                    lines.append(f"| {a} vs {b} | {env} | {n} | {w} | {l} | {p:.3f} |")
        lines.append("")

        # Pooled over the three training environments, and on held-out Reacher (levels 1-3), against native.
        lines += ["## Pooled against native (levels 1-3; training envs = TwoRoom + PushT + basketball)", "",
                  "| arm | train envs S | gained | lost | p | Reacher S | gained | lost | p |", "|---|---|---|---|---|---|---|---|---|"]
        nat = per_case.get("native", {})
        for arm in sorted(per_case):
            row = [arm]
            for group in (("tworoom", "pusht", "basketball"), ("reacher",)):
                a = {c: r for c, r in per_case[arm].items() if env_of(c) in group and (level_of(c) or 0) <= 3}
                b = {c: r for c, r in nat.items() if c in a}
                n, w, l, pv = paired(a, b)
                S = sum(bool(r["safe_completion"]) for r in a.values())
                row += [f"{S}/{n}", str(w), str(l), f"{pv:.3f}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Diagnosis: did the gate open, and did clearance move? (levels 2-3, where native completes but violates)
        lines += ["## Diagnosis on levels 2-3: gate-open cases and mean min clearance (steps; negative = depth inside)", "",
                  "| arm | env | gate opened | native clearance | arm clearance | safe | completed |", "|---|---|---|---|---|---|---|"]
        for arm in sorted(per_case):
            if arm == "native":
                continue
            for env in ("tworoom", "pusht", "basketball", "reacher"):
                cs = [c for c in per_case[arm] if env_of(c) == env and level_of(c) in (2, 3) and c in nat]
                if not cs:
                    continue
                op = sum(1 for c in cs if (per_case[arm][c].get("steer_open") or 0) + (per_case[arm][c].get("choose_open") or 0) > 0)
                mc = lambda d: float(np.mean([float(d[c].get("min_clear_steps") or 0.0) for c in cs]))
                S = sum(bool(per_case[arm][c]["safe_completion"]) for c in cs)
                C = sum(bool(per_case[arm][c]["success"]) for c in cs)
                lines.append(f"| {arm} | {env} | {op}/{len(cs)} | {mc(nat):+.2f} | {mc(per_case[arm]):+.2f} | {S} | {C} |")
        lines.append("")
    text = "\n".join(lines)
    if args.md:
        Path(args.md).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
