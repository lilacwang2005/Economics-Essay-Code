"""
模型二（主回归）：早期评论锚定效应
  因变量：stance_it（话题-时间窗口平均支持-反对得分）
  核心变量：earlyStance_2h × 窗口（时间）

表格设计（5列，中文经济学期刊风格）：
  (1) 混合OLS，无固定效应，话题聚类SE
  (2) 话题固定效应，话题聚类SE（主规格）
  (3) 双向固定效应（话题+窗口），话题聚类SE
  (4) 双向固定效应，双向聚类SE（话题×窗口，稳健性）
  (5) 双向固定效应 + 早期情感交互，话题聚类SE

输出：
  results/tables/TabA3_主回归聚合层面.tex
  results/figures/FigA3_锚定效应散点图.pdf
  results/figures/FigA4_锚定效应动态系数图.pdf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from table_utils import setup_plot_style, style_ax, ECON_COLORS, ECON_PALETTE, stars, fmt_coef, NOTE_CLUSTER_COMPARE
from lang_utils import get_lang, tr, get_results_root, latex_var_label

setup_plot_style()

ROOT = "/Users/wangmanlin/Desktop/citizens-will-and-politics"
CLEAN  = os.path.join(ROOT, "data/clean")
LANG = get_lang()
RESULTS_ROOT = get_results_root(ROOT, LANG)
TABLES = os.path.join(RESULTS_ROOT, "tables")
FIG    = os.path.join(RESULTS_ROOT, "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("加载数据...")
panel = pd.read_csv(os.path.join(CLEAN, "panel_topic_window.csv"), encoding="utf-8-sig")
panel = panel[panel["窗口"] >= 0].copy()

panel["log_volume"]         = np.log1p(panel["volume_it"])
panel["log_cum_volume"]     = np.log1p(panel["cum_volume_it"])
panel["t"]                  = panel["窗口"]
panel["log_t"]              = np.log1p(panel["窗口"])
panel["earlyStance_x_t"]    = panel["earlyStance_2h"] * panel["t"]
panel["earlyStance_x_logt"] = panel["earlyStance_2h"] * panel["log_t"]
panel["earlyConsist_x_t"]   = panel["earlyConsistency_2h"].fillna(0) * panel["t"]
panel["earlySentiment_x_t"] = panel["earlySentiment_2h"].fillna(0) * panel["t"]

panel_m = panel.dropna(subset=["stance_it", "earlyStance_2h", "earlyConsistency_2h"]).copy()
print(f"  有效观测：{len(panel_m):,}，话题数：{panel_m['title'].nunique():,}")

import pyfixest as pf

ctrl = "log_volume + log_cum_volume"

# ── 五个规格 ──────────────────────────────────────────────────────────────────
print("\n估计主回归（5个规格）...")

# (1) 混合OLS，话题聚类
m1 = pf.feols(
    f"stance_it ~ earlyStance_2h + earlyStance_x_t"
    f" + earlyConsistency_2h + earlyConsist_x_t + {ctrl}",
    data=panel_m, vcov={"CRV1": "title"}
)

# (2) 话题FE，话题聚类（主规格）
m2 = pf.feols(
    f"stance_it ~ earlyStance_x_t + earlyConsist_x_t + {ctrl} | title",
    data=panel_m, vcov={"CRV1": "title"}
)

# (3) 双向FE，话题聚类
m3 = pf.feols(
    f"stance_it ~ earlyStance_x_t + earlyConsist_x_t + {ctrl} | title + 窗口",
    data=panel_m, vcov={"CRV1": "title"}
)

# (4) 双向FE，双向聚类（话题×窗口）
m4 = pf.feols(
    f"stance_it ~ earlyStance_x_t + earlyConsist_x_t + {ctrl} | title + 窗口",
    data=panel_m, vcov={"CRV1": "title + 窗口"}
)

# (5) 双向FE + 情感交互，话题聚类
m5 = pf.feols(
    f"stance_it ~ earlyStance_x_t + earlySentiment_x_t"
    f" + earlyConsist_x_t + {ctrl} | title + 窗口",
    data=panel_m, vcov={"CRV1": "title"}
)

fits = [m1, m2, m3, m4, m5]

# ── 打印摘要 ──────────────────────────────────────────────────────────────────
print("\n=== 主回归结果摘要 ===")
headers = ["(1)混合OLS", "(2)话题FE（主）", "(3)双向FE", "(4)双向FE双向聚类", "(5)+情感交互"]
key_vars = [
    ("earlyStance_2h",       "早期锚定主效应 γ"),
    ("earlyStance_x_t",      "早期锚定×时间 δ₁"),
    ("earlySentiment_x_t",   "早期情感×时间"),
    ("earlyConsistency_2h",  "早期一致性 β₃"),
    ("earlyConsist_x_t",     "早期一致性×时间 β₄"),
    ("log_volume",           "log(窗口观测量)"),
]
for header, fit in zip(headers, fits):
    c = fit.coef(); s = fit.se(); p = fit.pvalue()
    print(f"\n{header}  N={fit._N:,}  R²={fit._r2:.3f}")
    for vname, vlabel in key_vars:
        if vname in c.index:
            st = stars(p[vname])
            print(f"  {vlabel:25s}  {c[vname]:8.4f}{st}  ({s[vname]:.4f})")

# ── LaTeX 表格 Tab A3 ─────────────────────────────────────────────────────────
print("\n生成 Tab A3...")

col_headers = [
    tr(LANG, r"\shortstack[c]{(1)\\混合OLS\\话题聚类SE}", r"\shortstack[c]{(1)\\Pooled OLS\\Topic-clustered SE}"),
    tr(LANG, r"\shortstack[c]{(2)\\话题固定效应\\话题聚类SE}", r"\shortstack[c]{(2)\\Topic FE\\Topic-clustered SE}"),
    tr(LANG, r"\shortstack[c]{(3)\\双向固定效应\\话题聚类SE}", r"\shortstack[c]{(3)\\Two-way FE\\Topic-clustered SE}"),
    tr(LANG, r"\shortstack[c]{(4)\\双向固定效应\\双向聚类SE}", r"\shortstack[c]{(4)\\Two-way FE\\Two-way clustered SE}"),
    tr(LANG, r"\shortstack[c]{(5)\\双向固定效应\\+情感交互}", r"\shortstack[c]{(5)\\Two-way FE\\+ Sentiment interaction}"),
]

rows = [
    ("earlyStance_2h",       tr(LANG, "早期锚定主效应 $\\hat{\\gamma}$", "Early-anchor main effect $\\hat{\\gamma}$")),
    ("earlyStance_x_t",      tr(LANG, "早期锚定$\\times$时间 $\\hat{\\delta}_1$", "Early anchor $\\times$ time $\\hat{\\delta}_1$")),
    ("earlySentiment_x_t",   tr(LANG, "早期情感$\\times$时间", "Early sentiment $\\times$ time")),
    ("earlyConsistency_2h",  tr(LANG, "早期评论一致性 $\\hat{\\beta}_3$", "Early comment consistency $\\hat{\\beta}_3$")),
    ("earlyConsist_x_t",     tr(LANG, "早期一致性$\\times$时间 $\\hat{\\beta}_4$", "Early consistency $\\times$ time $\\hat{\\beta}_4$")),
    ("log_volume",           tr(LANG, "log（窗口评论量）", "log(window comment volume)")),
    ("log_cum_volume",       tr(LANG, "log（累计评论量）", "log(cumulative comment volume)")),
]

note = tr(
    LANG,
    "***、**、*分别表示在1\\%、5\\%、10\\%的显著水平上显著。"
    "括号内为聚类稳健标准误，规格(1)-(3)(5)聚类至话题层面，规格(4)按话题与时间窗口双向聚类。"
    "因变量为话题-时间窗口内所有评论的平均支持-反对得分（$-100$至$+100$）。"
    "时间变量 $t$ 为窗口编号（以2小时为单位），$t=0$ 对应 $t_0$ 后0-2小时。"
    "规格(1)含截距项；规格(2)-(5)的早期锚定主效应被话题固定效应吸收，仅报告交互项。",
    "Notes: *** $p<0.01$, ** $p<0.05$, * $p<0.10$. Robust clustered standard errors are reported in parentheses. "
    "Columns (1)-(3)(5) use topic-level clustering, and column (4) uses two-way clustering by topic and time window. "
    "Dependent variable is topic-window mean support-opposition score ($-100$ to $+100$). "
    "Time variable $t$ is the 2-hour window index, where $t=0$ is the first 0-2 hours after $t_0$. "
    "Column (1) includes the intercept; in columns (2)-(5), the early-anchor main effect is absorbed by topic fixed effects."
)

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(f"\\caption{{{tr(LANG, '主回归：早期评论锚定效应（聚合层面）', 'Main Regression: Early Comment Anchoring (Aggregated Level)')}\\label{{tab:a3_main_agg}}}}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lcccccc}")
lines.append(r"\toprule")
lines.append(f"{tr(LANG, '变量', 'Variable')} & " + " & ".join(col_headers) + r" \\")
lines.append(r"\cmidrule(lr){2-6}")

for varname, label in rows:
    coef_row = [latex_var_label(label)]
    se_row   = [""]
    for fit in fits:
        c = fit.coef(); s = fit.se(); p = fit.pvalue()
        if varname in c.index:
            cv, sv = fmt_coef(c[varname], s[varname], p[varname])
            coef_row.append(cv); se_row.append(sv)
        else:
            coef_row.append("—"); se_row.append("")
    lines.append(" & ".join(coef_row) + r" \\")
    lines.append(" & ".join(se_row) + r" \\[2pt]")

lines.append(r"\midrule")
fe_rows = [
    (tr(LANG, "话题固定效应", "Topic fixed effects"),   [False, True,  True,  True,  True]),
    (tr(LANG, "时间窗口固定效应", "Window fixed effects"), [False, False, True,  True,  True]),
]
for label, flags in fe_rows:
    row = [label] + [r"\checkmark" if f else "" for f in flags]
    lines.append(" & ".join(row) + r" \\")
lines.append(r"\midrule")
lines.append(f"{tr(LANG, '聚类层面', 'Clustering level')} & {tr(LANG, '话题', 'Topic')} & {tr(LANG, '话题', 'Topic')} & {tr(LANG, '话题', 'Topic')} & {tr(LANG, '话题', 'Topic')}$\\times${tr(LANG, '窗口', 'Window')} & {tr(LANG, '话题', 'Topic')} \\\\")
lines.append(f"{tr(LANG, '观测值', 'Observations')} & " + " & ".join(f"{fit._N:,}" for fit in fits) + r" \\")
lines.append("$R^2$ & " + " & ".join(f"{fit._r2:.3f}" for fit in fits) + r" \\")
lines.append(r"\bottomrule")
lines.append(f"\\multicolumn{{6}}{{p{{0.98\\textwidth}}}}{{\\footnotesize {note}}}")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tab_a3_name = tr(LANG, "TabA3_主回归聚合层面.tex", "TabA3_MainRegression_Aggregated.tex")
tex_a3 = os.path.join(TABLES, tab_a3_name)
with open(tex_a3, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  → 已保存：{tex_a3}")

# ── Fig A3：按窗口散点图 ──────────────────────────────────────────────────────
print("\nGenerating Fig A3 (scatter plots)...")
WINDOWS = [0, 3, 6, 12, 18, 24]
window_labels = {0:"0-2h", 3:"6-8h", 6:"12-14h",
                 12:"24-26h", 18:"36-38h", 24:"48-50h"}

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for idx, win in enumerate(WINDOWS):
    ax = axes[idx]
    sub = panel_m[panel_m["窗口"] == win]
    if len(sub) < 10:
        ax.set_visible(False)
        continue
    x = sub["earlyStance_2h"].values
    y = sub["stance_it"].values
    ax.scatter(x, y, c=np.log1p(sub["volume_it"].values),
               cmap="Blues", alpha=0.5, s=16, edgecolors="none", zorder=3)
    if len(sub) > 5:
        coef = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, np.polyval(coef, xl), color=ECON_COLORS["red"], linewidth=1.8,
                label=tr(LANG, "OLS拟合线", "OLS fit"))
    ax.axhline(0, color="#AAAAAA", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="#AAAAAA", linestyle=":", linewidth=0.8)
    ax.set_xlabel(tr(LANG, "早期评论态度均值", "Early comment mean stance"), fontsize=10)
    ax.set_ylabel(tr(LANG, "窗口平均态度得分", "Window mean stance"), fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, framealpha=0.85)
    style_ax(ax)

plt.tight_layout()
fig_a3_name = tr(LANG, "FigA3_锚定效应散点图.pdf", "FigA3_EarlyStanceScatter.pdf")
out_a3 = os.path.join(FIG, fig_a3_name)
plt.savefig(out_a3, dpi=300, bbox_inches="tight")
plt.close()
print(f"  → 已保存：{out_a3}")

# ── Fig A4：逐窗口系数动态图 ──────────────────────────────────────────────────
print("\nGenerating Fig A4 (dynamic coefficients)...")

window_coefs = []
for win in sorted(panel_m["窗口"].unique()):
    sub = panel_m[panel_m["窗口"] == win].dropna(subset=["stance_it", "earlyStance_2h"])
    if len(sub) < 20:
        continue
    x = sub[["earlyStance_2h", "log_volume"]].assign(const=1.0).values
    y = sub["stance_it"].values
    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        n = len(sub)
        resid = y - x @ coef
        s2 = np.dot(resid, resid) / (n - x.shape[1])
        se = np.sqrt(s2 * np.linalg.inv(x.T @ x)[0, 0])
        window_coefs.append({"window": win, "hour": win*2, "beta": coef[0], "se": se, "n": n})
    except Exception:
        pass

wc = pd.DataFrame(window_coefs)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(wc["hour"], wc["beta"] - 1.96*wc["se"],
                wc["beta"] + 1.96*wc["se"],
                alpha=0.18, color=ECON_COLORS["blue"])
ax.plot(wc["hour"], wc["beta"], color=ECON_COLORS["blue"], linewidth=2,
        marker="o", markersize=4, label=tr(LANG, "截面回归系数", "Cross-sectional coefficient"))
ax.axhline(0, color="#444444", linewidth=0.9, linestyle="--")

for mask, color, label in [
    (wc["hour"] <= 24, ECON_COLORS["red"],   tr(LANG, "前24小时趋势", "Trend: first 24h")),
    (wc["hour"] >  24, ECON_COLORS["green"], tr(LANG, "24小时后趋势", "Trend: after 24h")),
]:
    if mask.sum() > 2:
        c = np.polyfit(wc[mask]["hour"], wc[mask]["beta"], 1)
        xl = wc[mask]["hour"].values
        ax.plot(xl, np.polyval(c, xl), "--", color=color, linewidth=1.5, label=label)

# Shaded area: early comment definition window (0-4h).
from matplotlib.patches import Patch
ax.axvspan(0, 4, alpha=0.12, color=ECON_COLORS["red"], zorder=0)
_early_patch = Patch(facecolor=ECON_COLORS["red"], alpha=0.35, label=tr(LANG, "早期评论定义区间", "Early-comment window"))

ax.set_xlabel(tr(LANG, "上榜后小时数", "Hours after trending"), fontsize=12)
ax.set_ylabel(tr(LANG, "截面回归系数", "Cross-sectional coefficient"), fontsize=12)
ax.legend(handles=[h for h in ax.get_legend_handles_labels()[0]] + [_early_patch],
          fontsize=10, loc="upper right")
ax.set_xlim(left=0)
style_ax(ax)

plt.tight_layout()
fig_a4_name = tr(LANG, "FigA4_锚定效应动态系数图.pdf", "FigA4_DynamicCoefficient.pdf")
out_a4 = os.path.join(FIG, fig_a4_name)
plt.savefig(out_a4, dpi=300, bbox_inches="tight")
plt.close()
print(f"  → 已保存：{out_a4}")

# ── 核心结果摘要 ──────────────────────────────────────────────────────────────
print("\n=== 核心结果摘要 ===")
c2 = m2.coef(); s2 = m2.se(); p2 = m2.pvalue()
d = c2.get("earlyStance_x_t", np.nan)
sd = s2.get("earlyStance_x_t", np.nan)
pd_ = p2.get("earlyStance_x_t", np.nan)
print(f"主规格(2) 话题FE：δ₁ = {d:.4f}{stars(pd_)}  SE={sd:.4f}  p={pd_:.4f}")
print(f"  → 话题内earlyStance每增加1分，随时间推移每窗口额外影响：{d:.4f}分")
print(f"  → 72h（=36窗口）后累积衰减量：{d*36:.2f}分")
print(f"\nCross-sectional effect near t=0: beta = {wc['beta'].iloc[0]:.3f}")
print(f"  A 1-point increase in early stance predicts +{wc['beta'].iloc[0]:.3f} points in the current window.")
print(f"Late-window effect ({wc['hour'].iloc[-1]}h): beta = {wc['beta'].iloc[-1]:.3f}")
print(f"Share of windows with beta > 0: {(wc['beta']>0).mean()*100:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab A4：个体评论层面主回归（含 exposure_ijt）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n[Tab A4] 加载个体评论数据...")
ind = pd.read_csv(os.path.join(CLEAN, "individual_comments.csv"), encoding="utf-8-sig")
ind = ind[ind["窗口"] >= 0].copy()

# 构建交互项
ind["log_volume_i"]     = np.log1p(ind["earlyVolume_2h"].fillna(0))
ind["t"]                = ind["窗口"]
ind["log_t"]            = np.log1p(ind["窗口"])
ind["earlyStance_x_t"]  = ind["earlyStance_2h"] * ind["t"]
ind["earlyConsist_x_t"] = ind["earlyConsistency_2h"].fillna(0) * ind["t"]

# 有效样本：需要 stance_score, earlyStance_2h, exposure_ijt
ind_m = ind.dropna(subset=["stance_score", "earlyStance_2h", "earlyConsistency_2h"]).copy()
print(f"  个体层面有效观测：{len(ind_m):,}，话题数：{ind_m['title'].nunique():,}")

print("估计个体层面回归（3个规格）...")

# (1) 个体层面：话题FE，earlyStance×t（无 exposure）
ia1 = pf.feols(
    "stance_score ~ earlyStance_x_t + earlyConsist_x_t + log_volume_i | title",
    data=ind_m, vcov={"CRV1": "title"}
)

# (2) 个体层面：话题FE，加入 exposure_ijt
ind_exp = ind_m.dropna(subset=["exposure_ijt"]).copy()
ia2 = pf.feols(
    "stance_score ~ earlyStance_x_t + earlyConsist_x_t + exposure_ijt + log_volume_i | title",
    data=ind_exp, vcov={"CRV1": "title"}
)

# (3) 个体层面：双向FE，加入 exposure_ijt
ia3 = pf.feols(
    "stance_score ~ earlyStance_x_t + earlyConsist_x_t + exposure_ijt + log_volume_i | title + 窗口",
    data=ind_exp, vcov={"CRV1": "title"}
)

print("\n=== 个体层面结果 ===")
for name, fit in [("(1)话题FE（无exposure）", ia1),
                  ("(2)话题FE（含exposure）", ia2),
                  ("(3)双向FE（含exposure）", ia3)]:
    coefs = fit.coef(); ses = fit.se(); pvals = fit.pvalue()
    print(f"\n{name}  N={fit._N:,}  R²={fit._r2:.3f}")
    for var in ["earlyStance_x_t", "exposure_ijt", "earlyConsist_x_t"]:
        if var in coefs.index:
            print(f"  {var:30s}  {coefs[var]:8.4f}  ({ses[var]:.4f})  p={pvals[var]:.4f}")

# ── LaTeX Tab A4 ──────────────────────────────────────────────────────────────
print("\n生成 Tab A4...")

fits_a4 = [ia1, ia2, ia3]
col_headers_a4 = [
    tr(LANG, r"\shortstack[c]{(1)\\话题固定效应\\无曝光变量}", r"\shortstack[c]{(1)\\Topic FE\\No exposure variable}"),
    tr(LANG, r"\shortstack[c]{(2)\\话题固定效应\\含热评曝光}", r"\shortstack[c]{(2)\\Topic FE\\With hot-comment exposure}"),
    tr(LANG, r"\shortstack[c]{(3)\\双向固定效应\\含热评曝光}", r"\shortstack[c]{(3)\\Two-way FE\\With hot-comment exposure}"),
]
rows_a4 = [
    ("earlyStance_x_t",  tr(LANG, "早期锚定×时间", "Early anchor × time")),
    ("exposure_ijt",     tr(LANG, "热评曝光态度均值", "Mean stance of hot-comment exposure")),
    ("earlyConsist_x_t", tr(LANG, "早期一致性×时间", "Early consistency × time")),
    ("log_volume_i",     tr(LANG, "log（早期评论量）", "log(early comment volume)")),
]
note_a4 = (
    "***、**、*分别表示在1\\%、5\\%、10\\%的显著水平上显著。"
    "括号内为在话题层面聚类的稳健标准误。"
    "因变量为单条评论的支持-反对得分（$-100$至$+100$）。"
    "热评曝光态度均值（exposure$_{ijt}$）为用户发评前该话题累计热评（按点赞排序前20条）的平均态度得分。"
    "规格(1)含全部个体评论样本，规格(2)(3)限于曝光变量非缺失的样本。"
)

lines_a4 = []
lines_a4.append(r"\begin{table}[htbp]")
lines_a4.append(r"\centering")
lines_a4.append(f"\\caption{{{tr(LANG, '个体评论层面锚定效应主回归', 'Individual-Level Anchoring Regressions')}\\label{{tab:a4_individual}}}}")
lines_a4.append(r"\scriptsize")
lines_a4.append(r"\resizebox{\textwidth}{!}{%")
lines_a4.append(r"\begin{tabular}{p{0.38\textwidth}ccc}")
lines_a4.append(r"\toprule")
lines_a4.append(f"{tr(LANG, '变量', 'Variable')} & " + " & ".join(col_headers_a4) + r" \\")
lines_a4.append(r"\cmidrule(lr){2-4}")

for varname, label in rows_a4:
    coef_row = [latex_var_label(label)]
    se_row   = [""]
    for fit in fits_a4:
        c = fit.coef(); s = fit.se(); p = fit.pvalue()
        if varname in c.index:
            cv, sv = fmt_coef(c[varname], s[varname], p[varname])
            coef_row.append(cv); se_row.append(sv)
        else:
            coef_row.append("—"); se_row.append("")
    lines_a4.append(" & ".join(coef_row) + r" \\")
    lines_a4.append(" & ".join(se_row) + r" \\[2pt]")

lines_a4.append(r"\midrule")
fe_specs = [
    (tr(LANG, "话题固定效应", "Topic fixed effects"),   [True,  True,  True]),
    (tr(LANG, "时间窗口固定效应", "Window fixed effects"), [False, False, True]),
]
for lbl, flags in fe_specs:
    row = [lbl] + [r"\checkmark" if f else "" for f in flags]
    lines_a4.append(" & ".join(row) + r" \\")
lines_a4.append(r"\midrule")
lines_a4.append(f"{tr(LANG, '聚类层面', 'Clustering level')} & {tr(LANG, '话题', 'Topic')} & {tr(LANG, '话题', 'Topic')} & {tr(LANG, '话题', 'Topic')} \\\\")
lines_a4.append(f"{tr(LANG, '观测值', 'Observations')} & " + " & ".join(f"{fit._N:,}" for fit in fits_a4) + r" \\")
lines_a4.append("$R^2$ & " + " & ".join(f"{fit._r2:.3f}" for fit in fits_a4) + r" \\")
lines_a4.append(r"\bottomrule")
lines_a4.append(r"\end{tabular}")
lines_a4.append(r"}")
lines_a4.append(r"\vspace{2pt}")
lines_a4.append(r"\begin{minipage}{0.98\textwidth}")
lines_a4.append(r"\footnotesize " + tr(
    LANG,
    note_a4,
    "Notes: *** $p<0.01$, ** $p<0.05$, * $p<0.10$. Standard errors are clustered at topic level. "
    "Dependent variable is individual comment stance score ($-100$ to $+100$). "
    "Exposure is the mean stance of top-20 liked comments visible before posting. "
    "Column (1) uses all comments; columns (2)-(3) use observations with non-missing exposure."
))
lines_a4.append(r"\end{minipage}")
lines_a4.append(r"\end{table}")

tab_a4_name = tr(LANG, "TabA4_个体层面主回归.tex", "TabA4_IndividualMainRegression.tex")
tex_a4 = os.path.join(TABLES, tab_a4_name)
with open(tex_a4, "w", encoding="utf-8") as f:
    f.write("\n".join(lines_a4))
print(f"  → 已保存：{tex_a4}")

print("\n完成。")
