import os
import json

class OutputWriter:
    def write(self, run_id: str, integrated_plan: dict, evals: dict, base_dir: str = "outputs"):
        out_dir = os.path.join(base_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        
        with open(os.path.join(out_dir, "integrated_plan.json"), "w") as f:
            json.dump(integrated_plan, f, indent=2)
            
        with open(os.path.join(out_dir, "evals.json"), "w") as f:
            json.dump(evals, f, indent=2)
            
        print(f"[MASTRA] Wrote outputs to {out_dir}")
