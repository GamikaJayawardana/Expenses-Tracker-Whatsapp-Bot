"""
Extraction-accuracy eval for the intent router.

Runs every labelled message in dataset.jsonl through router.parse_message and
reports how often the first extracted action matches the expected intent, type,
amount and category. LLM cost/latency for the run is pulled from the trace table.

Usage:
    python evals/run_eval.py            # run the whole set
    python evals/run_eval.py --limit 20 # run the first 20 rows
    python evals/run_eval.py --delay 0.5
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path

# Make the project root importable when run as `python evals/run_eval.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tracing                    # noqa: E402
from router import parse_message  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"

# Lenient category matching: predicted category is correct if it equals the
# expected label or falls under one of its accepted aliases.
CATEGORY_ALIASES = {
    "Food":          {"food", "dining", "restaurant", "lunch", "meals", "snacks"},
    "Groceries":     {"groceries", "grocery", "food", "supermarket"},
    "Transport":     {"transport", "transportation", "travel", "taxi", "commute"},
    "Fuel":          {"fuel", "petrol", "gas", "diesel", "transport"},
    "Bills":         {"bills", "bill", "utilities", "utility"},
    "Entertainment": {"entertainment", "leisure", "subscription", "subscriptions"},
    "Shopping":      {"shopping", "clothes", "clothing", "retail"},
    "Health":        {"health", "medical", "healthcare", "medicine", "fitness"},
    "Rent":          {"rent", "housing"},
    "Education":     {"education", "school", "tuition", "courses"},
    "Salary":        {"salary", "wages", "pay", "income"},
    "Freelance":     {"freelance", "consulting", "gig", "project"},
    "Bonus":         {"bonus"},
    "Gift":          {"gift", "gifts", "present"},
    "Interest":      {"interest", "dividend", "dividends", "investment"},
    "Sales":         {"sales", "sale", "sold"},
    "Refund":        {"refund", "cashback", "refunds"},
    "Personal Care": {"personal care", "grooming", "self care"},
    "Donation":      {"donation", "charity", "donations"},
    "Insurance":     {"insurance", "premium"},
}


def category_matches(expected: str, predicted) -> bool:
    if not predicted:
        return False
    exp = expected.strip().lower()
    pred = str(predicted).strip().lower()
    if pred == exp:
        return True
    return pred in CATEGORY_ALIASES.get(expected, set())


def load_dataset(limit=None):
    rows = []
    with open(DATASET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


async def run(limit=None, delay=0.0):
    tracing.init_db()
    rows = load_dataset(limit)

    totals = {
        "intent":   {"correct": 0, "n": 0},
        "type":     {"correct": 0, "n": 0},
        "amount":   {"correct": 0, "n": 0},
        "category": {"correct": 0, "n": 0},
    }
    exact = 0
    failures = []

    for i, row in enumerate(rows, 1):
        msg      = row["message"]
        expected = row["expected"]
        try:
            actions = await parse_message(msg, "2026-07-09")
        except Exception as e:
            actions = []
            print(f"[{i}] ERROR parsing '{msg}': {e}")

        pred = actions[0] if actions else {}
        row_ok = True

        # intent — always checked
        totals["intent"]["n"] += 1
        if str(pred.get("intent", "")).upper() == expected["intent"]:
            totals["intent"]["correct"] += 1
        else:
            row_ok = False

        if "type" in expected:
            totals["type"]["n"] += 1
            if str(pred.get("type", "")).lower() == expected["type"]:
                totals["type"]["correct"] += 1
            else:
                row_ok = False

        if "amount" in expected:
            totals["amount"]["n"] += 1
            if pred.get("amount") is not None and abs(float(pred["amount"]) - float(expected["amount"])) < 0.01:
                totals["amount"]["correct"] += 1
            else:
                row_ok = False

        if "category" in expected:
            totals["category"]["n"] += 1
            if category_matches(expected["category"], pred.get("category")):
                totals["category"]["correct"] += 1
            else:
                row_ok = False

        if row_ok:
            exact += 1
        else:
            failures.append((msg, expected, pred))

        if delay:
            await asyncio.sleep(delay)

    _report(len(rows), totals, exact, failures)


def _pct(correct, n):
    return f"{(correct / n * 100):5.1f}%  ({correct}/{n})" if n else "   n/a"


def _report(n_rows, totals, exact, failures):
    print("\n" + "=" * 52)
    print(f" EXTRACTION ACCURACY  —  {n_rows} messages")
    print("=" * 52)
    print(f"  Intent    : {_pct(totals['intent']['correct'],   totals['intent']['n'])}")
    print(f"  Type      : {_pct(totals['type']['correct'],     totals['type']['n'])}")
    print(f"  Amount    : {_pct(totals['amount']['correct'],   totals['amount']['n'])}")
    print(f"  Category  : {_pct(totals['category']['correct'], totals['category']['n'])}")
    print("-" * 52)
    print(f"  Exact row : {_pct(exact, n_rows)}")
    print("=" * 52)

    if failures:
        print(f"\n MISMATCHES ({len(failures)}):")
        for msg, expected, pred in failures:
            print(f"  • {msg}")
            print(f"      expected: {expected}")
            print(f"      got     : intent={pred.get('intent')} type={pred.get('type')} "
                  f"amount={pred.get('amount')} category={pred.get('category')}")

    stats = tracing.summary()
    print("\n LLM COST / LATENCY (this run + history):")
    print(f"  calls          : {stats['calls']}")
    print(f"  total cost USD : {stats['total_cost_usd']}")
    print(f"  total tokens   : {stats['total_tokens']}")
    print(f"  avg latency ms : {stats['avg_latency_ms']}")
    print(f"  success rate   : {stats['success_rate']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only run the first N rows")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between calls")
    args = ap.parse_args()
    asyncio.run(run(limit=args.limit, delay=args.delay))
