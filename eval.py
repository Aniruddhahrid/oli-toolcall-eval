#!/usr/bin/env python3
"""eval.py -- score the tool calls in an assistant transcript against the golden set.

The golden set (golden_set.json) is 20 creator requests, each paired with the
tool calls a correct Oli assistant should make. This script extracts the tool
calls from a transcript, lines each request up with its golden case, and scores
the match.

    python eval.py --transcript transcripts/reference_full.json
    python eval.py --transcript transcripts/redteam_real.json --png metrics.png
    python eval.py --all --json metrics.json --png metrics.png

Metrics, per case:
    exact   - called tool names == expected tool names, nothing forbidden, and
              "no call" respected for no-action cases  (the headline metric)
    recall  - expected tools that were called        (catches missed calls)
    precision - called tools that were expected       (catches extra calls)
    args    - expected argument values present        (only where the case sets them)

Transcript format (one JSON file per session):
    {"turns": [
        {"role": "user",      "case_id": "C08", "content": "..."},
        {"role": "assistant", "content": "...",
         "tool_calls": [{"name": "update_deal", "arguments": {"stage": "agreed"}}]}
    ]}
`case_id` on a user turn pins it to a golden case; without it the request text
is matched by keyword overlap.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLDEN = os.path.join(HERE, "golden_set.json")
DEFAULT_DIR = os.path.join(HERE, "transcripts")
MATCH_THRESHOLD = 0.34


# ---------------------------------------------------------------- helpers ----
def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def toks(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def resolve(args, key):
    """Find key at top level or one level deep (e.g. inside `changes`)."""
    if key in args:
        return True, args.get(key)
    for v in args.values():
        if isinstance(v, dict) and key in v:
            return True, v.get(key)
    return False, None


def arg_ok(args, key, want):
    found, got = resolve(args, key)
    if want == "*":
        return found and got not in (None, "", [], {})
    if not found:
        return False
    if isinstance(want, bool):
        return got is want
    if isinstance(want, (int, float)):
        return got == want
    if isinstance(want, str):
        return want.lower() in str(got).lower()
    return got == want


# ----------------------------------------------------------- extraction ------
def extract_episodes(transcript):
    """Turn a transcript into episodes: {case_id, request, calls}."""
    eps, cur = [], None
    for t in transcript.get("turns", []):
        if t.get("role") == "user":
            if cur:
                eps.append(cur)
            cur = {"case_id": t.get("case_id"), "request": t.get("content", ""), "calls": []}
        else:
            if cur is None:
                cur = {"case_id": None, "request": "", "calls": []}
            cur["calls"].extend(t.get("tool_calls") or [])
    if cur:
        eps.append(cur)
    return eps


def match_case(request, cases):
    rt = toks(request)
    best, bs = None, 0.0
    for c in cases:
        s = jaccard(rt, toks(c["request"]))
        if s > bs:
            best, bs = c, s
    return (best, round(bs, 3)) if best and bs >= MATCH_THRESHOLD else (None, round(bs, 3))


# ------------------------------------------------------------- scoring -------
def score_episode(case, calls):
    expected = set(case["expect_calls"])
    forbidden = set(case["forbid_calls"])
    called = [c.get("name") for c in calls]
    called_set = set(called)

    tp = len(expected & called_set)
    fn = len(expected - called_set)
    fp = len(called_set - expected)
    recall = tp / len(expected) if expected else (1.0 if not called_set else 0.0)
    precision = tp / len(called_set) if called_set else (1.0 if not expected else 0.0)

    forbid_hit = sorted(called_set & forbidden)
    no_call_ok = (not case["expect_none"]) or (len(calls) == 0)

    # argument checks (only where the case defines them)
    arg_specs = case.get("expect_args") or {}
    arg_fails = []
    for tool, want in arg_specs.items():
        tool_calls = [c for c in calls if c.get("name") == tool]
        if not any(all(arg_ok(c.get("arguments") or {}, k, v) for k, v in want.items())
                   for c in tool_calls):
            arg_fails.append(tool)
    args_ok = not arg_fails

    exact = (expected == called_set) and not forbid_hit and no_call_ok and args_ok
    return {
        "case_id": case["id"],
        "category": case["category"],
        "expected": sorted(expected),
        "called": called,
        "exact": exact,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "tp": tp, "fp": fp, "fn": fn,
        "forbid_hit": forbid_hit,
        "arg_fails": arg_fails,
        "args_ok": args_ok,
    }


def aggregate(results):
    scored = [r for r in results if r.get("matched")]
    n = len(scored) or 1
    return {
        "cases_scored": len(scored),
        "exact_match_rate": round(sum(r["exact"] for r in scored) / n, 3),
        "mean_recall": round(sum(r["recall"] for r in scored) / n, 3),
        "mean_precision": round(sum(r["precision"] for r in scored) / n, 3),
        "arg_pass_rate": round(sum(r["args_ok"] for r in scored) / n, 3),
        "forbidden_violations": sum(len(r["forbid_hit"]) for r in scored),
        "false_positives": sum(r["fp"] for r in scored),
        "false_negatives": sum(r["fn"] for r in scored),
    }


def run(transcript_path, golden):
    cases = golden["cases"]
    by_id = {c["id"]: c for c in cases}
    tr = load(transcript_path)
    results, exercised = [], set()

    for ep in extract_episodes(tr):
        if ep["case_id"] and ep["case_id"] in by_id:
            case, mscore = by_id[ep["case_id"]], 1.0
        else:
            case, mscore = match_case(ep["request"], cases)
        if case is None:
            results.append({"matched": False, "request": ep["request"]})
            continue
        exercised.add(case["id"])
        r = score_episode(case, ep["calls"])
        r["matched"] = True
        r["request"] = ep["request"]
        r["match_score"] = mscore
        results.append(r)

    return {"transcript": os.path.basename(transcript_path),
            "results": results, "aggregate": aggregate(results),
            "exercised": sorted(exercised),
            "not_exercised": sorted(set(by_id) - exercised)}


# ------------------------------------------------------------- reporting -----
def print_report(rep):
    agg = rep["aggregate"]
    print("=" * 70)
    print("TRANSCRIPT:", rep["transcript"])
    print("  cases scored=%d  exact_match=%.2f  recall=%.2f  precision=%.2f  args=%.2f"
          % (agg["cases_scored"], agg["exact_match_rate"], agg["mean_recall"],
             agg["mean_precision"], agg["arg_pass_rate"]))
    print("  forbidden violations=%d  false_positives=%d  false_negatives=%d"
          % (agg["forbidden_violations"], agg["false_positives"], agg["false_negatives"]))
    print("-" * 70)
    for r in rep["results"]:
        if not r.get("matched"):
            print("  [ -- ] no golden match:", (r["request"] or "")[:52])
            continue
        flag = "PASS" if r["exact"] else "FAIL"
        print("  [%s] %-4s %-12s expected=%-34s called=%s"
              % (flag, r["case_id"], r["category"], ",".join(r["expected"]) or "(none)",
                 ",".join(r["called"]) or "(none)"))
        if r["forbid_hit"]:
            print("         FORBIDDEN CALL:", r["forbid_hit"])
        if r["arg_fails"]:
            print("         ARG MISMATCH:", r["arg_fails"])
    print("=" * 70)


def make_chart(runs, path):
    """runs: list of (label, rep). Panel 1 = per-case exact match of the widest
    run; Panel 2 = aggregate metric bars across runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widest = max(runs, key=lambda kv: kv[1]["aggregate"]["cases_scored"])[1]
    ids = [r["case_id"] for r in widest["results"] if r.get("matched")]
    exact = [1 if r["exact"] else 0 for r in widest["results"] if r.get("matched")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("Oli tool-call eval", fontsize=14, fontweight="bold")

    colors = ["#2e9e5b" if e else "#d1495b" for e in exact]
    ax1.bar(ids, exact, color=colors)
    ax1.set_title("Per-case exact match  (%s)" % widest["transcript"])
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("1 = exact match, 0 = miss")
    ax1.tick_params(axis="x", rotation=90, labelsize=8)

    metrics = ["exact_match_rate", "mean_recall", "mean_precision", "arg_pass_rate"]
    labels = ["exact\nmatch", "recall", "precision", "args"]
    n = len(runs)
    width = 0.8 / n
    for i, (label, rep) in enumerate(runs):
        vals = [rep["aggregate"][m] for m in metrics]
        xs = [j - 0.4 + width * (i + 0.5) for j in range(len(metrics))]
        ax2.bar(xs, vals, width=width, label=label)
        for x, v in zip(xs, vals):
            ax2.text(x, v + 0.02, "%.2f" % v, ha="center", fontsize=7)
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Aggregate metrics")
    ax2.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    print("wrote", path)


# ---------------------------------------------------------------- main -------
def main():
    ap = argparse.ArgumentParser(description="Score transcript tool calls against the golden set.")
    ap.add_argument("--transcript")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--json")
    ap.add_argument("--png")
    args = ap.parse_args()

    if args.all:
        paths = sorted(os.path.join(DEFAULT_DIR, f) for f in os.listdir(DEFAULT_DIR)
                       if f.endswith(".json"))
    elif args.transcript:
        paths = [args.transcript]
    else:
        ap.error("pass --transcript <path> or --all")

    golden = load(args.golden)
    runs, out = [], {}
    for p in paths:
        rep = run(p, golden)
        print_report(rep)
        runs.append((os.path.basename(p).replace(".json", ""), rep))
        out[rep["transcript"]] = rep

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("wrote", args.json)
    if args.png and runs:
        make_chart(runs, args.png)


if __name__ == "__main__":
    main()
