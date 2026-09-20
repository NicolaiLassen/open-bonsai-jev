#!/usr/bin/env python3
"""
End-to-end example: triage support tickets against a running api.py.

Three decisions per ticket, then a drafted reply only for the tickets judged
both angry and urgent. The interesting part is that this is ordinary control
flow -- `if angry and urgent:` -- where the conditions happen to be semantic.
Nothing parses model prose; each `if` is a probability read off one forward
pass.

Start the server first:
    api.py --model ../bonzi/models/Ternary-Bonsai-2-27B --gpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

TICKETS = [
    {
        "id": "T-1001",
        "from": "dana@acme.example",
        "subject": "billing charged twice",
        "body": "You charged my card TWICE this month and nobody has replied "
                "to my last two emails. This is the third time. Fix it today "
                "or I am disputing the charge.",
    },
    {
        "id": "T-1002",
        "from": "sam@beta.example",
        "subject": "question about dark mode",
        "body": "Hi! Loving the app. Is there a dark mode on the roadmap? No "
                "rush at all, just curious.",
    },
    {
        "id": "T-1003",
        "from": "ops@gamma.example",
        "subject": "production API returning 500",
        "body": "Our production integration has been returning 500 on every "
                "call for the last 40 minutes. This is blocking checkout for "
                "all of our customers.",
    },
]


def call(base: str, path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def decide(base, state, question, options=None) -> dict:
    body = {"state": state, "question": question}
    if options:
        body["options"] = options
    return call(base, "/decide", body)


def main(argv=None):
    ap = argparse.ArgumentParser(description="support-ticket triage demo")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--draft", action="store_true",
                    help="also generate replies (slow; needs /chat)")
    args = ap.parse_args(argv)

    try:
        health = json.loads(
            urllib.request.urlopen(f"{args.api}/health", timeout=10).read()
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"no api at {args.api} ({e}). Start api.py first.")
    print(f"model: {health['model']} via {health['backend']}\n")

    t0 = time.perf_counter()
    calls = 0

    for ticket in TICKETS:
        state = {k: ticket[k] for k in ("from", "subject", "body")}
        print(f"{ticket['id']}  {ticket['subject']}")

        angry = decide(args.api, state, "Is the customer angry?")
        urgent = decide(args.api, state, "Is this blocking the customer's work right now?")
        team = decide(args.api, state,
                      "Which team should handle this?",
                      ["billing", "engineering", "product"])
        calls += 3

        print(f"  angry   {angry['choice']:<12} p={max(angry['probabilities'].values()):.3f}")
        print(f"  urgent  {urgent['choice']:<12} p={max(urgent['probabilities'].values()):.3f}")
        print(f"  team    {team['choice']:<12} p={max(team['probabilities'].values()):.3f}")

        # Semantic conditions, ordinary control flow.
        if angry["choice"] == "true" and urgent["choice"] == "true":
            print("  -> escalate")
            if args.draft:
                reply = call(args.api, "/chat", {
                    "message": (
                        "Write a short, calm, apologetic first reply to this "
                        f"support ticket. Do not promise a refund.\n\n"
                        f"{json.dumps(state, indent=2)}"
                    ),
                    "max_tokens": 200,
                })
                calls += 1
                print(f"  draft:\n    " + reply["reply"].replace("\n", "\n    "))
        else:
            print("  -> normal queue")
        print()

    elapsed = time.perf_counter() - t0
    print(f"{calls} calls in {elapsed:.1f} s on one loaded model "
          f"({elapsed / calls:.2f} s each)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
