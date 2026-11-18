"""
表格与图表工具函数（中文经济学期刊格式）
供 model1~model4、robustness 脚本调用
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from lang_utils import latex_var_label

def _pick_installed_font(candidates):
    names = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in names:
            return name
    return None


def setup_lang_font():
    """
    按 PAPER_LANG 设置图表字体：
    - zh: 宋体（优先 SimSun）
    - en: Times New Roman（无则回退 Times）
    """
    lang = os.getenv("PAPER_LANG", "en").strip().lower()
    if lang == "zh":
        cn_candidates = ["SimSun", "Songti SC", "STSong", "Noto Serif CJK SC"]
        picked = _pick_installed_font(cn_candidates) or "SimSun"
        plt.rcParams["font.family"] = [picked]
    else:
        en_candidates = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
        picked = _pick_installed_font(en_candidates) or "Times New Roman"
        plt.rcParams["font.family"] = [picked]
    plt.rcParams["axes.unicode_minus"] = False
    return picked

# ── 经济学期刊图表风格 ─────────────────────────────────────────────────────────
# 颜色方案：沉稳专业，适合中文经济学期刊（黑白打印友好）
ECON_COLORS = {
    "blue":   "#2C5F8A",
    "red":    "#B03A2E",
    "green":  "#1A6B3C",
    "gray":   "#666666",
    "purple": "#5B3080",
    "orange": "#B7770D",
}
ECON_PALETTE = ["#2C5F8A", "#B03A2E", "#1A6B3C", "#666666", "#5B3080", "#B7770D"]

def setup_plot_style():
    """
    设置经济学期刊图表风格（仿AER / 经济研究 风格）：
    - 白底、无上/右边框
    - 浅灰横向网格线
    - 统一字号（标签12pt，刻度10pt）
    - 300 dpi 输出
    """
    setup_lang_font()
    plt.rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.color":         "#DDDDDD",
        "grid.linewidth":     0.6,
        "grid.linestyle":     "--",
        "axes.axisbelow":     True,
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "#CCCCCC",
        "lines.linewidth":    1.8,
        "lines.markersize":   5,
        "axes.linewidth":     0.8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "axes.titlepad":      8,
    })

def style_ax(ax):
    """对单个 axes 应用经济学期刊风格（去掉上/右边框，仅保留横向浅色网格）"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

# ── 显著性星号 ────────────────────────────────────────────────────────────────
def stars(pval):
    if pval < 0.01:  return "***"
    if pval < 0.05:  return "**"
    if pval < 0.10:  return "*"
    return ""

# ── 格式化系数行 ──────────────────────────────────────────────────────────────
def fmt_coef(val, se, pval, decimals=4):
    """返回 (系数字符串, 标准误字符串)，含星号"""
    s = stars(pval)
    return f"{val:.{decimals}f}{s}", f"({se:.{decimals}f})"

# ── LaTeX 表格构建器 ──────────────────────────────────────────────────────────
class LatexTable:
    """
    用法：
        tbl = LatexTable(
            caption="...",
            label="tab:...",
            col_headers=["(1)\\\\Pooled", "(2)\\\\话题FE", ...],
            note="注：括号内为...",
        )
        tbl.add_section_header("主要变量")
        tbl.add_row("早期锚定×时间 $\\delta_1$", fits, "earlyStance_x_t")
        tbl.add_bottom_stats(fits)
        tbl.add_fe_rows([("话题固定效应", ...), ("窗口固定效应", ...)])
        tbl.save(path)
    """
    STANDARD_NOTE = (
        "注：括号内为{se_type}。"
        "***、**、*分别表示在1\\%、5\\%、10\\%的显著水平上显著。"
        "{extra}"
    )

    def __init__(self, caption, label, col_headers, note="", n_cols=None):
        self.caption = caption
        self.label   = label
        self.col_headers = col_headers
        self.note    = note
        self.n_cols  = n_cols if n_cols else len(col_headers) + 1  # +1 for row label col
        self._lines  = []

    def _ncols_str(self):
        return "l" + "c" * (self.n_cols - 1)

    def build_header(self):
        L = self._lines
        L.append(r"\begin{table}[htbp]")
        L.append(r"\centering")
        L.append(f"\\caption{{{self.caption}}}")
        L.append(f"\\label{{{self.label}}}")
        L.append(r"\small")
        L.append(f"\\begin{{tabular}}{{{self._ncols_str()}}}")
        L.append(r"\toprule")
        L.append(" & " + " & ".join(self.col_headers) + r" \\")
        L.append(r"\midrule")

    def add_section_header(self, text):
        self._lines.append(
            f"\\multicolumn{{{self.n_cols}}}{{l}}{{\\textit{{{text}}}}} \\\\"
        )

    def add_row(self, label, fits, varname, decimals=4):
        """
        fits: list of pyfixest fit objects
        varname: coefficient name to extract from each fit
        """
        coef_row = [latex_var_label(label)]
        se_row   = [""]
        for fit in fits:
            coefs = fit.coef()
            ses   = fit.se()
            pvals = fit.pvalue()
            if varname in coefs.index:
                c, s = fmt_coef(coefs[varname], ses[varname], pvals[varname], decimals)
                coef_row.append(c)
                se_row.append(s)
            else:
                coef_row.append("—")
                se_row.append("")
        self._lines.append(" & ".join(coef_row) + r" \\")
        self._lines.append(" & ".join(se_row) + r" \\[3pt]")

    def add_empty_row(self):
        self._lines.append(r"\\[-6pt]")

    def add_midrule(self):
        self._lines.append(r"\midrule")

    def add_fe_rows(self, fe_specs):
        """
        fe_specs: list of (label, list_of_bool_or_str)
        例：[("话题固定效应", [False, True, True, True]),
             ("窗口固定效应", [False, False, True, True])]
        """
        self.add_midrule()
        for label, flags in fe_specs:
            row = [label]
            for f in flags:
                if isinstance(f, bool):
                    row.append(r"\checkmark" if f else "")
                else:
                    row.append(str(f))
            self._lines.append(" & ".join(row) + r" \\")

    def add_bottom_stats(self, fits, show_r2=True, show_within_r2=False):
        """N 和 R²"""
        self.add_midrule()
        n_row  = ["观测值"]
        r2_row = ["$R^2$"]
        for fit in fits:
            n_row.append(f"{fit._N:,}")
            r2_row.append(f"{fit._r2:.3f}")
        self._lines.append(" & ".join(n_row)  + r" \\")
        if show_r2:
            self._lines.append(" & ".join(r2_row) + r" \\")

    def add_cluster_row(self, cluster_labels):
        """聚类层面说明行"""
        row = ["聚类层面"] + list(cluster_labels)
        self._lines.append(" & ".join(row) + r" \\")

    def build_footer(self):
        L = self._lines
        L.append(r"\bottomrule")
        note_width = f"0.{min(98, self.n_cols * 14)}\\textwidth"
        L.append(f"\\multicolumn{{{self.n_cols}}}{{p{{{note_width}}}}}{{\\footnotesize {self.note}}}")
        L.append(r"\end{tabular}")
        L.append(r"\end{table}")

    def save(self, path):
        self.build_header()
        self.build_footer()
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines))
        print(f"  → 已保存：{path}")
        # reset so build_header/footer are not duplicated on re-save
        self._lines = []

    def get_lines(self):
        self.build_header()
        self.build_footer()
        return list(self._lines)

    def write_to(self, path, extra_lines_before_footer=None):
        """一次性写出（带可选额外行）"""
        lines = []
        # header
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{self.caption}}}")
        lines.append(f"\\label{{{self.label}}}")
        lines.append(r"\small")
        lines.append(f"\\begin{{tabular}}{{{self._ncols_str()}}}")
        lines.append(r"\toprule")
        lines.append(" & " + " & ".join(self.col_headers) + r" \\")
        lines.append(r"\midrule")
        # body
        lines.extend(self._lines)
        # extra
        if extra_lines_before_footer:
            lines.extend(extra_lines_before_footer)
        # footer
        lines.append(r"\bottomrule")
        note_width = "0.98\\textwidth"
        lines.append(f"\\multicolumn{{{self.n_cols}}}{{p{{{note_width}}}}}{{\\footnotesize {self.note}}}")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  → 已保存：{path}")


# ── 标准脚注模板 ──────────────────────────────────────────────────────────────
NOTE_CLUSTER_TOPIC = (
    "***、**、*分别表示在1\\%、5\\%、10\\%的显著水平上显著。"
    "括号内为在话题层面聚类的稳健标准误（共820个话题集群）。"
    "因变量为话题-时间窗口内评论的平均支持-反对得分（-100至+100）。"
)

NOTE_CLUSTER_TWOWAY = (
    "***、**、*分别表示在1\\%、5\\%、10\\%的显著水平上显著。"
    "括号内为双向聚类稳健标准误（话题层面 × 时间窗口层面）。"
    "因变量为话题-时间窗口内评论的平均支持-反对得分（-100至+100）。"
)

NOTE_CLUSTER_COMPARE = (
    "***、**、*分别表示在1\\%、5\\%、10\\%的显著水平上显著。"
    "括号内为聚类稳健标准误，聚类层面见表格底部注释行。"
    "因变量为话题-时间窗口内评论的平均支持-反对得分（-100至+100）。"
)
