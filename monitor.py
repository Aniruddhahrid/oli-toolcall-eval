#!/usr/bin/env python3
"""monitor.py -- watch Oli tool calls as they happen and flag anomalies live.

eval.py scores a finished transcript. This runs continuously: feed it every Oli
tool call (from logs, an event stream, or a tail-able JSONL file) and it checks
each one against the golden set, printing an anomaly line the moment something
looks wrong, plus a running summary.

    python monitor.py --replay sample_feed.jsonl      # process a file, then exit
    python monitor.py --watch  oli_calls.jsonl        # follow a live file
    cat feed.jsonl | python monitor.py --stdin        # read a stream

Anomaly types
    NO_GOLDEN_MATCH  request doesn't resemble any golden case (unseen behaviour)
    FORBIDDEN_TOOL   a tool the golden case says must never be called here
    UNEXPECTED_TOOL  a tool that isn't expected for this request (extra call)
    MISSING_EXPECTED at turn end, an expected tool was never called
    ARG_MISMATCH     expected argument value missing or wrong

Event format (one JSON object per line). Either one call at a time:
    {"case_id": "C08", "request": "...", "tool": "update_deal",
     "arguments": {"stage": "agreed"}, "turn_end": true}
or a whole turn at once:
    {"request": "...", "tool_calls": [{"name": "...", "arguments": {...}}]}
`case_id` is optional; without it the request text is matched by keyword.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLDEN = os.path.join(HERE, "golden_set.json")
MATCH_THRESHOLD = 0.34


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def toks(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def resolve(args, key):
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


def match_case(request, cases):
    rt = toks(request)
    best, bs = None, 0.0
    for c in cases:
        s = jaccard(rt, toks(c["request"]))
        if s > bs:
            best, bs = c, s
    return (best, round(bs, 3)) if best and bs >= MATCH_THRESHOLD else (None, round(bs, 3))


class Monitor:
    def __init__(self, golden, stream=sys.stdout):
        self.cases = golden["cases"]
        self.by_id = {c["id"]: c for c in self.cases}
        self.out = stream
        self.n_events = 0
        self.anomalies = []
        self.counts = {}
        self._buffers = {}   # turn key -> {"case":..., "seen": set(), "args": {tool: [args]}}

    def _log(self, kind, detail):
        self.anomalies.append({"kind": kind, "detail": detail})
        self.counts[kind] = self.counts.get(kind, 0) + 1
        print("  ANOMALY  %-16s %s" % (kind, detail), file=self.out)

    def _turn_key(self, event, case):
        return event.get("case_id") or (case["id"] if case else "?")

    def handle(self, event):
        self.n_events += 1
        request = event.get("request", "")
        case = self.by_id.get(event["case_id"]) if event.get("case_id") in self.by_id else None
        mscore = 1.0
        if case is None:
            case, mscore = match_case(request, self.cases)

        calls = event.get("tool_calls")
        if calls is None and event.get("tool"):
            calls = [{"name": event["tool"], "arguments": event.get("arguments") or {}}]
        calls = calls or []

        if case is None:
            self._log("NO_GOLDEN_MATCH",
                      "request=%r (best overlap %.2f) calls=%s"
                      % (request[:46], mscore, [c.get("name") for c in calls]))
            return

        expected, forbidden = set(case["expect_calls"]), set(case["forbid_calls"])
        key = self._turn_key(event, case)
        buf = self._buffers.setdefault(key, {"case": case, "seen": set()})

        for c in calls:
            name = c.get("name")
            args = c.get("arguments") or {}
            buf["seen"].add(name)

            if name in forbidden:
                self._log("FORBIDDEN_TOOL",
                          "%s on %s: %s%s" % (case["id"], name,
                                              "must never be called here", " args=%s" % args if args else ""))
            elif name not in expected:
                self._log("UNEXPECTED_TOOL",
                          "%s: called %s, not expected (expected=%s)"
                          % (case["id"], name, sorted(expected) or "none"))

            want = (case.get("expect_args") or {}).get(name)
            if want and not all(arg_ok(args, k, v) for k, v in want.items()):
                self._log("ARG_MISMATCH",
                          "%s: %s args=%s, expected %s" % (case["id"], name, args, want))

        # a whole turn arrived at once, or the caller flagged the turn as done
        complete = event.get("tool_calls") is not None or event.get("turn_end")
        if complete:
            missing = expected - buf["seen"]
            if missing and not (case["expect_none"] and not buf["seen"]):
                if not case["expect_none"]:
                    self._log("MISSING_EXPECTED",
                              "%s: never called %s" % (case["id"], sorted(missing)))
            if case["expect_none"] and buf["seen"]:
                extra = sorted(buf["seen"] - forbidden)
                if extra:
                    self._log("UNEXPECTED_TOOL",
                              "%s: expected no call, saw %s" % (case["id"], extra))
            self._buffers.pop(key, None)

    def summary(self):
        rate = (len(self.anomalies) / self.n_events) if self.n_events else 0.0
        print("\n" + "=" * 66)
        print("MONITOR SUMMARY")
        print("  events processed : %d" % self.n_events)
        print("  anomalies        : %d  (%.0f%% of events)" % (len(self.anomalies), rate * 100))
        for k in sorted(self.counts):
            print("    %-16s %d" % (k, self.counts[k]))
        if not self.counts:
            print("    (none)")
        print("=" * 66)


def iter_lines(path, watch):
    """Yield JSON lines from a file; if watch, keep following it."""
    with open(path, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                if not watch:
                    return
                time.sleep(1.0)
                continue
            line = line.strip()
            if line:
                yield line


def main():
    ap = argparse.ArgumentParser(description="Continuously flag anomalies in Oli tool calls.")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--replay", help="process a JSONL file once, then exit")
    ap.add_argument("--watch", help="follow a JSONL file as it grows")
    ap.add_argument("--stdin", action="store_true", help="read JSONL from stdin")
    args = ap.parse_args()

    golden = load(args.golden)
    mon = Monitor(golden)

    if args.replay:
        src = iter_lines(args.replay, watch=False)
    elif args.watch:
        src = iter_lines(args.watch, watch=True)
    elif args.stdin:
        src = (l.strip() for l in sys.stdin if l.strip())
    else:
        ap.error("pass --replay <file>, --watch <file>, or --stdin")

    for line in src:
        try:
            mon.handle(json.loads(line))
        except json.JSONDecodeError:
            print("  skip: not JSON ->", line[:60])
    mon.summary()


if __name__ == "__main__":
    main()
