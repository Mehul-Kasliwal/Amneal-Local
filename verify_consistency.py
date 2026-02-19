"""
Run check_26.py 10 times on each BMR and compare anomaly results for consistency.
"""
import subprocess, json, sys

PYTHON = "/home/softsensor/Desktop/Amneal/.venv/bin/python"
CHECK_SCRIPT = "/home/softsensor/Desktop/Amneal/check_26.py"

BMRS = {
    "challenge_bmr": "/home/softsensor/Desktop/Amneal/challenge_bmr/05jan_AH250076_50Checks 1.json",
    "complete_data_61": "/home/softsensor/Desktop/Amneal/complete_data_61 1.json",
}

RUNS = 10


def extract_anomalies(debug_output):
    """Pull (page, status) from material_result entries in return_debug."""
    anomalies = []
    for entry in debug_output:
        if "material_result" in entry:
            p = entry.get("page_no")
            s = entry.get("status")
            name = entry["material_result"].get("raw_material_name", "?")
            assay = entry["material_result"].get("lot_1", {}).get("assay")
            lod = entry["material_result"].get("lot_1", {}).get("lod")
            anomalies.append({
                "page": p, "status": s, "name": name,
                "assay": assay, "lod": lod
            })
    return anomalies


def run_once(json_path):
    """Run check_26.py with a specific JSON path and return parsed output."""
    # Temporarily patch JSON_PATH by passing via env or modifying stdin
    # Easier: use sed-like approach or just modify the script inline.
    # Actually, let's just import and call directly.
    
    # We'll load the JSON, import the validator, and call run_validation
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pages = raw.get("steps", {}).get("filled_master_json", [])
    
    # Import the validator
    sys.path.insert(0, "/home/softsensor/Desktop/Amneal")
    from check_26 import PotencyCalculationValidator
    
    v = PotencyCalculationValidator()
    return v.run_validation(pages)


def main():
    for bmr_name, bmr_path in BMRS.items():
        print(f"\n{'='*60}")
        print(f"BMR: {bmr_name}")
        print(f"Path: {bmr_path}")
        print(f"{'='*60}")
        
        all_anomalies = []
        for i in range(1, RUNS + 1):
            print(f"  Run {i}/{RUNS} ...", end=" ", flush=True)
            try:
                debug_output = run_once(bmr_path)
                anomalies = extract_anomalies(debug_output)
                all_anomalies.append(anomalies)
                pages_flagged = [a["page"] for a in anomalies if a["status"] == 1]
                print(f"anomaly pages: {pages_flagged}")
            except Exception as e:
                print(f"ERROR: {e}")
                all_anomalies.append(None)

        # --- Consistency check ---
        valid = [a for a in all_anomalies if a is not None]
        if not valid:
            print(f"\n  ❌ All runs failed for {bmr_name}!")
            continue

        ref = valid[0]
        consistent = all(v == ref for v in valid)

        if consistent:
            print(f"\n  ✅ CONSISTENT across {len(valid)} runs")
            print(f"     Materials detected: {len(ref)}")
            for mat in ref:
                flag = "🔴 ANOMALY" if mat["status"] == 1 else "🟢 OK"
                print(f"       Page {mat['page']}: {mat['name']} "
                      f"(assay={mat['assay']}, lod={mat['lod']}) → {flag}")
        else:
            print(f"\n  ⚠️  INCONSISTENT across {len(valid)} runs!")
            # Show per-run breakdown
            for i, run_data in enumerate(valid, 1):
                pages_flagged = [a["page"] for a in run_data if a["status"] == 1]
                assays = {a["page"]: a["assay"] for a in run_data}
                print(f"     Run {i}: anomaly pages={pages_flagged}, assays={assays}")


if __name__ == "__main__":
    main()
