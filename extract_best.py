import json
import sys
from pathlib import Path


def extract_best(log_dir: str):
    path = Path(log_dir)
    if not path.exists():
        print(f"Error: {path} not found")
        return

    # Find all PUCT sampler step files
    sampler_files = sorted(path.glob("puct_sampler_step_*.json"),
                           key=lambda p: int(p.stem.split("_")[-1]))
    if not sampler_files:
        print(f"No PUCT sampler files found in {path}")
        return

    # Read the last PUCT file (contains all discovered states)
    with open(sampler_files[-1]) as f:
        data = json.load(f)

    states = data["states"]
    if not states:
        print("No states found in sampler file")
        return

    # Best state = highest value (value = -c5_bound, so highest value = lowest C5)
    best = max(states, key=lambda s: s["value"])
    best_c5 = -best["value"]
    best_timestep = best["timestep"]
    best_id = best["id"]
    best_construction = best.get("construction", [])
    construction_len = len(best_construction)

    print("=" * 60)
    print(f"Experiment: {log_dir}")
    print(f"Best C₅ bound: {best_c5:.10f}")
    print(f"State ID: {best_id}")
    print(f"Timestep: {best_timestep}")
    print(f"Construction length: {construction_len}")
    print(f"Construction (h values): {best_construction}")
    print("=" * 60)

    # Also show target
    print(f"\nTarget: C₅ ≤ 0.38080 (current record from paper)")
    if best_c5 < 0.38080:
        print(f"✅ NEW RECORD! Beat the paper's best by {0.38080 - best_c5:.6f}")
    else:
        print(f"❌ Need improvement: {best_c5 - 0.38080:.6f} above the record")

    # Show raw_score trend from metrics.jsonl
    metrics_file = path / "metrics.jsonl"
    if metrics_file.exists():
        print(f"\nRaw score (C₅) trend from metrics.jsonl:")
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    print(f"  Step {d['step']}: raw_score = {d['env/all/raw_score']:.6f}  (reward/max = {d['env/all/reward/max']:.4f})")

    # Save best sequence as JSON
    out_path = path / "best_solution.json"
    with open(out_path, "w") as f:
        json.dump({
            "c5_bound": best_c5,
            "construction": best_construction,
            "construction_length": construction_len,
            "state_id": best_id,
            "timestep": best_timestep,
        }, f, indent=2)
    print(f"\nBest solution saved to {out_path}")


if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "tinker_log/test-erdos-min-overlap-local"
    extract_best(log_dir)
