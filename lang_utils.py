import os
import re


def get_lang():
    lang = os.getenv("PAPER_LANG", "en").strip().lower()
    if lang not in {"zh", "en"}:
        raise ValueError("PAPER_LANG must be 'zh' or 'en'")
    return lang


def tr(lang: str, zh: str, en: str) -> str:
    return zh if lang == "zh" else en


def get_results_root(project_root: str, lang: str) -> str:
    suffix = "zh" if lang == "zh" else "en"
    return os.path.join(project_root, f"results_{suffix}")


def latex_var_label(label: str) -> str:
    """
    若变量名包含下划线且尚未进入数学模式，则自动包裹为 $...$，
    例如 Stance_i -> $Stance_i$。
    """
    s = str(label)
    if "_" not in s or "$" in s:
        return s
    # 含中文时不要进入数学模式，否则中文常显示为空心方块
    if re.search(r"[\u4e00-\u9fff]", s):
        return s.replace("_", r"\_")
    # 纯变量名（ASCII）进入数学模式，获得正确下标
    if re.fullmatch(r"[A-Za-z0-9_]+", s):
        return f"${s}$"
    # 其余情况在文本模式中转义下划线，避免编译报错
    return s.replace("_", r"\_")
