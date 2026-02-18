"""
Run check_40 10 times and compare results to detect LLM inconsistency/hallucination.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "yog_checks_oops"))
from extract_data import get_filled_master_json
from check_40 import Actual_BMR_Reconciliation_Protocol

NUM_RUNS = 10
complete_data_path = "/home/softsensor/Desktop/Amneal/challenge_bmr/05jan_AH250076_50Checks 1.json"

print("Loading data...")
filled_master_json = get_filled_master_json(complete_data_path)
print(f"Loaded {len(filled_master_json)} pages\n")

validator = Actual_BMR_Reconciliation_Protocol()

all_results = []
all_debug = []

for run in range(1, NUM_RUNS + 1):
    print(f"=== Run {run}/{NUM_RUNS} ===")
    try:
        results, debug = validator.run(filled_master_json)
        all_results.append(results)
        all_debug.append(debug)
        anomaly_pages = [r for r in results if r["anomaly_status"] == 1]
        print(f"  Anomaly pages: {[r['page_no'] for r in anomaly_pages]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results.append(None)
        all_debug.append(None)

# --- Analysis ---
print("\n" + "=" * 70)
print("CONSISTENCY ANALYSIS")
print("=" * 70)

# Find pages that went through LLM
llm_pages = set()
for debug_run in all_debug:
    if debug_run is None:
        continue
    for entry in debug_run:
        if "llm_output" in entry:
            llm_pages.add(entry["page_no"])

print(f"\nPages processed by LLM: {sorted(llm_pages)}\n")

# Compare anomaly status per page across runs
inconsistent_pages = []
for page_no in sorted(llm_pages):
    statuses = []
    for run_idx, results in enumerate(all_results):
        if results is None:
            statuses.append("ERROR")
            continue
        page_result = next((r for r in results if r["page_no"] == page_no), None)
        status = page_result["anomaly_status"] if page_result else "MISSING"
        statuses.append(status)

    unique_statuses = set(str(s) for s in statuses)
    is_consistent = len(unique_statuses) == 1
    marker = "✓" if is_consistent else "✗ INCONSISTENT"

    print(f"Page {page_no:>4}: statuses={statuses}  {marker}")
    if not is_consistent:
        inconsistent_pages.append(page_no)

print(f"\n{'=' * 70}")
if inconsistent_pages:
    print(f"⚠ INCONSISTENCIES FOUND on pages: {inconsistent_pages}")
    for page_no in inconsistent_pages:
        print(f"\n--- Detailed debug for page {page_no} ---")
        for run_idx, debug in enumerate(all_debug):
            if debug is None:
                continue
            entry = next((d for d in debug if d.get("page_no") == page_no and "llm_output" in d), None)
            if entry:
                details = entry.get("verification_details", [])
                mismatches = [d for d in details if d.get("match") is False]
                print(f"  Run {run_idx+1}: anomaly={entry['anomaly_status']}, mismatches={len(mismatches)}")
                for m in mismatches:
                    print(f"    {m['name']}({m['symbol']}): reported={m['reported_value']}, computed={m['python_computed']}, diff={m['diff']}")
else:
    print(f"✓ ALL {NUM_RUNS} RUNS CONSISTENT — no hallucination detected.")
print("=" * 70)
