import subprocess
import json
import sys
from collections import Counter

def run_check():
    try:
        result = subprocess.run(
            [sys.executable, "check_26.py"],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running check: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

def extract_anomalies(results):
    anomalies = {}
    for page in results:
        p_no = page.get("page")
        status = page.get("anomaly_status")
        if status == 1:
            anomalies[p_no] = status
    return anomalies

def main():
    runs = 10
    all_results = []
    
    print(f"Starting {runs} consistency runs for check_26.py...")
    
    for i in range(runs):
        print(f"Run {i+1}...", end=" ", flush=True)
        results = run_check()
        anomalies = extract_anomalies(results)
        all_results.append(anomalies)
        print(f"Found {len(anomalies)} anomalies: {anomalies}")

    # Analysis
    print("\n--- Consistency Analysis ---")
    
    # Check if all runs are identical
    first_run = all_results[0]
    is_consistent = all(res == first_run for res in all_results)
    
    if is_consistent:
        print("✅ All 10 runs produced IDENTICAL results.")
        print(f"Consistent Result: {first_run}")
    else:
        print("❌ Inconsistent results detected.")
        # Count variations
        c = Counter(str(sorted(d.items())) for d in all_results)
        for res_str, count in c.items():
            print(f"Result appearing {count} times: {res_str}")

if __name__ == "__main__":
    main()
