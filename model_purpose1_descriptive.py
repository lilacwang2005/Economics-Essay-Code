"""
研究目的一：描述统计 + 截面基准回归
输出（按 PAPER_LANG）：
  results_zh/tables/Tab1_描述统计.tex          / results_en/tables/Tab1_SummaryStatistics.tex
  results_zh/figures/Fig1_描述性分布.pdf       / results_en/figures/Fig1_DescriptiveDistributions.pdf
  results_zh/figures/Fig2_按落实程度分组对比.pdf / results_en/figures/Fig2_SolveGroupComparison.pdf
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/wangmanlin/Desktop/citizens-will-and-politics"
sys.path.insert(0, os.path.join(ROOT, "code"))
from lang_utils import get_lang, tr, get_results_root, latex_var_label
from table_utils import setup_plot_style, style_ax, ECON_COLORS

LANG = get_lang()
setup_plot_style()

RESULTS_ROOT = get_results_root(ROOT, LANG)
TABLES  = os.path.join(RESULTS_ROOT, "tables")
FIGURES = os.path.join(RESULTS_ROOT, "figures")
CLEAN   = os.path.join(ROOT, "data", "clean")
os.makedirs(TABLES,  exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ── 0. 加载数据 ────────────────────────────────────────────────────────────────
print(tr(LANG, "加载数据…", "Loading data..."))
tvars = pd.read_csv(os.path.join(CLEAN, "purpose1_topic_vars.csv"), encoding="utf-8-sig")

# 样本筛选：去除高水军话题
data = tvars[tvars["BotRatio_i"].fillna(0) <= 0.5].copy()
data["上榜年份"]  = data["上榜年份"].astype(str)
data["category_L1"] = data["category_L1"].fillna("其他")
data["log_PreBaseline"] = np.log1p(data["PreBaseline_i"].fillna(0))
data["log_NumPosts"]    = np.log1p(data["NumPosts_i"].fillna(0))
print(f"  {tr(LANG, '样本量', 'N')}：{len(data):,}")

# ── 1. 描述统计 Tab 1 ──────────────────────────────────────────────────────────
print(tr(LANG, "生成 Tab 1 描述统计…", "Generating Tab 1 Summary Statistics..."))

stat_vars = {
    "总评论数":              tr(LANG, "总评论数",              "Total Comments"),
    "总点赞数":              tr(LANG, "总点赞数",              "Total Likes"),
    "总转发数":              tr(LANG, "总转发数",              "Total Reposts"),
    "Stance_i":              tr(LANG, "平均支持-反对得分",      "Avg. Stance Score"),
    "Sentiment_i":           tr(LANG, "平均情感得分",           "Avg. Sentiment Score"),
    "AngerRatio_i":          tr(LANG, "愤怒评论占比",           "Anger Comment Ratio"),
    "StanceSD_i":            tr(LANG, "意见分歧（态度SD）",     "Opinion Discord (Stance SD)"),
    "StancePositiveRatio_i": tr(LANG, "支持率（态度>20）",      "Support Rate (Stance>20)"),
    "BotRatio_i":            tr(LANG, "水军帖子占比",           "Bot Post Ratio"),
    "PreBaseline_i":         tr(LANG, "基准期匹配新闻数",       "Pre-period Matched News"),
    "SolveDegree_score":     tr(LANG, "落实程度（0-100连续）",  "Solve Degree (0-100)"),
    "DaysToSolve_i":         tr(LANG, "落实天数（生存分析）",   "Days to Solve (Survival)"),
}

col_var   = tr(LANG, "变量",   "Variable")
col_mean  = tr(LANG, "均值",   "Mean")
col_sd    = tr(LANG, "标准差", "SD")
col_med   = tr(LANG, "中位数", "Median")
col_min   = tr(LANG, "最小值", "Min")
col_max   = tr(LANG, "最大值", "Max")
col_n     = tr(LANG, "观测数", "N")

rows = []
for col, label in stat_vars.items():
    if col not in data.columns:
        continue
    s = data[col].dropna()
    rows.append({
        col_var:  latex_var_label(label),
        col_mean: f"{s.mean():.3f}",
        col_sd:   f"{s.std():.3f}",
        col_med:  f"{s.median():.3f}",
        col_min:  f"{s.min():.3f}",
        col_max:  f"{s.max():.3f}",
        col_n:    f"{len(s):,}",
    })
tab1_df = pd.DataFrame(rows)

caption = tr(LANG, "描述统计", "Summary Statistics")
label   = "tab:summary_stats"
note    = tr(
    LANG,
    "注：样本已剔除水军占比超过50\\%的话题。",
    "Note: Topics with bot ratio above 50\\% are excluded from the sample.",
)

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(f"\\caption{{{caption}}}")
lines.append(f"\\label{{{label}}}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrrrrrr}")
lines.append(r"\toprule")
lines.append(
    f"{col_var} & {col_mean} & {col_sd} & {col_med} & {col_min} & {col_max} & {col_n} \\\\"
)
lines.append(r"\midrule")
for _, row in tab1_df.iterrows():
    lines.append(
        f"{row[col_var]} & {row[col_mean]} & {row[col_sd]} & {row[col_med]}"
        f" & {row[col_min]} & {row[col_max]} & {row[col_n]} \\\\"
    )
lines.append(r"\bottomrule")
lines.append(
    f"\\multicolumn{{7}}{{p{{0.98\\textwidth}}}}{{\\footnotesize {note}}}"
)
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tab1_fname = tr(LANG, "Tab1_描述统计.tex", "Tab1_SummaryStatistics.tex")
with open(os.path.join(TABLES, tab1_fname), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  {tr(LANG, '✓', '✓')} {tab1_fname}")

# ── 2. 图1：热度/态度/落实程度分布 ────────────────────────────────────────────
print(tr(LANG, "生成 图1…", "Generating Fig 1..."))
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

ax = axes[0]
ax.hist(data["总评论数"].dropna(), bins=30, color=ECON_COLORS["blue"], edgecolor="white", linewidth=0.4)
ax.set_xlabel(tr(LANG, "总评论数", "Total Comments"))
ax.set_ylabel(tr(LANG, "话题数", "Number of Topics"))
style_ax(ax)

ax = axes[1]
ax.hist(data["Stance_i"].dropna(), bins=30, color=ECON_COLORS["green"], edgecolor="white", linewidth=0.4)
ax.set_xlabel(tr(LANG, "平均支持-反对得分", "Avg. Stance Score"))
ax.set_ylabel(tr(LANG, "话题数", "Number of Topics"))
style_ax(ax)

ax = axes[2]
ax.hist(data["SolveDegree_score"].dropna(), bins=10, color=ECON_COLORS["red"], edgecolor="white", linewidth=0.4)
ax.set_xlabel(tr(LANG, "最高落实得分（0-100）", "Max Solve Degree (0-100)"))
ax.set_ylabel(tr(LANG, "话题数", "Number of Topics"))
style_ax(ax)

fig.tight_layout()
fig1_fname = tr(LANG, "Fig1_描述性分布.pdf", "Fig1_DescriptiveDistributions.pdf")
fig.savefig(os.path.join(FIGURES, fig1_fname), dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ {fig1_fname}")

# ── 3. 图2：按落实程度分组对比 ───────────────────────────────────────────────
print(tr(LANG, "生成 图2…", "Generating Fig 2..."))
data_with_dv = data[data["SolveDegree_score"].notna()].copy()

group_labels = [
    tr(LANG, "0（无关）",      "0 (Irrelevant)"),
    tr(LANG, "1-20（仅讨论）", "1-20 (Discussion Only)"),
    tr(LANG, "21-60（推进中）","21-60 (In Progress)"),
    tr(LANG, "61-100（已解决）","61-100 (Resolved)"),
]
data_with_dv["solve_group"] = pd.cut(
    data_with_dv["SolveDegree_score"],
    bins=[-1, 0, 20, 60, 100],
    labels=group_labels,
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
plot_specs = [
    ("总评论数",  tr(LANG, "平均评论数",        "Avg. Comments")),
    ("Stance_i",  tr(LANG, "平均支持-反对得分", "Avg. Stance Score")),
    ("Sentiment_i", tr(LANG, "平均情感得分",    "Avg. Sentiment Score")),
]
for ax, (col, ylabel) in zip(axes, plot_specs):
    grp = data_with_dv.groupby("solve_group", observed=True)[col].mean()
    ax.bar(grp.index.astype(str), grp.values, color=ECON_COLORS["blue"], edgecolor="white", linewidth=0.4)
    ax.set_xlabel(tr(LANG, "落实程度", "Implementation Degree"))
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    style_ax(ax)

fig.tight_layout()
fig2_fname = tr(LANG, "Fig2_按落实程度分组对比.pdf", "Fig2_SolveGroupComparison.pdf")
fig.savefig(os.path.join(FIGURES, fig2_fname), dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ {fig2_fname}")

print(f"\n{tr(LANG, '完成！', 'Done!')}")

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
