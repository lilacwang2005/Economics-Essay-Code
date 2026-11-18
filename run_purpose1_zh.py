import os, subprocess, sys

SCRIPTS = [
    "model_purpose1_descriptive.py",
    "model_purpose1_eventstudy.py",
    "model_purpose1_survival.py",
    "model_purpose1_robustness.py",
]

def main():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PAPER_LANG"] = "zh"
    for script in SCRIPTS:
        path = os.path.join(code_dir, script)
        print(f"\n=== Running {script} (PAPER_LANG=zh) ===")
        subprocess.run([sys.executable, path], check=True, env=env)
    print("\nAll purpose-1 outputs generated in results_zh.")

if __name__ == "__main__":
    main()
