import os
import subprocess
import sys


SCRIPTS = [
    "descriptive_purpose2.py",
    "model1_dynamic_panel.py",
    "model2_main_regression.py",
    "model3_iv.py",
    "model4_panel_var.py",
    "robustness_purpose2.py",
]


def main():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PAPER_LANG"] = "en"

    for script in SCRIPTS:
        path = os.path.join(code_dir, script)
        print(f"\n=== Running {script} (PAPER_LANG=en) ===")
        subprocess.run([sys.executable, path], check=True, env=env)

    print("\nAll purpose-2 outputs generated in results_en.")


if __name__ == "__main__":
    main()
