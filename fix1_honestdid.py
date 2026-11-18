#!/usr/bin/env python3
"""
fix1_honestdid.py — 覆盖全部 10 个事件研究规格的 HonestDiD 敏感性分析
═══════════════════════════════════════════════════════════════════════════
规格清单（5 处理变量 × 2 频率 = 10）：
  热度：总点赞数 / 总转发数 / 总评论数 × 月度 / 季度
  态度：Stance_i / Sentiment_i           × 月度 / 季度

方法：Rambachan & Roth (2023, REStud) 相对幅度法（Delta-RM）

输出：
  results_zh/figures/FigR1_HonestDiD_{label}_{freq}.pdf （10 张单独图）
  results_zh/figures/FigR1_HonestDiD汇总.pdf            （1 张森林图）
  results_zh/tables/TabR1_HonestDiD敏感性.tex            （1 张汇总表）

前置条件：
  Stata MP 已安装；ssc install reghdfe, replace; ssc install ftools, replace
"""

import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy import stats as sp_stats

# ═══════════════════════════════════════════════════════════════════════════════
ROOT       = "/Users/wangmanlin/Desktop/citizens-will-and-politics"
STATA_PATH = "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp"

sys.path.insert(0, os.path.join(ROOT, "code"))
from lang_utils import get_lang, tr, get_results_root
from table_utils import setup_plot_style, style_ax, ECON_COLORS

LANG = get_lang()
setup_plot_style()

RESULTS_ROOT = get_results_root(ROOT, LANG)
TABLES  = os.path.join(RESULTS_ROOT, "tables")
FIGURES = os.path.join(RESULTS_ROOT, "figures")
CLEAN   = os.path.join(ROOT, "data", "clean")
TEMP    = os.path.join(ROOT, "temp_stata")
for d in [TABLES, FIGURES, TEMP]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  1. 数据准备（与 model_purpose1_eventstudy.py 完全一致）
# ═══════════════════════════════════════════════════════════════════════════════
print("加载 Event Study 面板…")
panel = pd.read_csv(os.path.join(CLEAN, "purpose1_event_panel.csv"),
                    encoding="utf-8-sig")
panel = panel[panel["BotRatio_i"].fillna(0) <= 0.5].copy()
panel["article_date"] = pd.to_datetime(panel["article_date"])
panel["hot_date"]     = pd.to_datetime(panel["hot_date"])
panel["rel_month"] = (
    panel["article_date"].dt.to_period("M")
    - panel["hot_date"].dt.to_period("M")
).apply(lambda x: x.n)
panel["calendar_ym"] = panel["article_date"].dt.to_period("M").astype(str)
for col in ["总点赞数", "总转发数", "总评论数"]:
    panel[col] = panel[col].fillna(0)
panel["Stance_i"]    = panel["Stance_i"].fillna(0)
panel["Sentiment_i"] = panel["Sentiment_i"].fillna(0)

monthly = panel.groupby(["title", "rel_month"], as_index=False).agg(
    y_max_score    = ("y_max_score",  "max"),
    total_likes    = ("总点赞数",     "first"),
    total_reposts  = ("总转发数",     "first"),
    total_comments = ("总评论数",     "first"),
    Stance_i       = ("Stance_i",     "first"),
    Sentiment_i    = ("Sentiment_i",  "first"),
    calendar_ym    = ("calendar_ym",  "first"),
)
panel_m = monthly[(monthly["rel_month"] >= -12) & (monthly["rel_month"] <= 24)].copy()

panel_q = monthly.copy()
panel_q["rel_quarter"] = (panel_q["rel_month"] / 3).apply(np.floor).astype(int)
panel_q["calendar_yq"] = (
    panel_q["calendar_ym"].str[:4] + "Q"
    + ((panel_q["calendar_ym"].str[5:7].astype(int) - 1) // 3 + 1).astype(str)
)
panel_q = panel_q[(panel_q["rel_quarter"] >= -4) & (panel_q["rel_quarter"] <= 8)].copy()
panel_q_agg = panel_q.groupby(["title", "rel_quarter"], as_index=False).agg(
    y_max_score    = ("y_max_score",  "max"),
    total_likes    = ("total_likes",  "first"),
    total_reposts  = ("total_reposts","first"),
    total_comments = ("total_comments","first"),
    Stance_i       = ("Stance_i",     "first"),
    Sentiment_i    = ("Sentiment_i",  "first"),
    calendar_yq    = ("calendar_yq",  "first"),
)

DFS = {"m": panel_m, "q": panel_q_agg}
print(f"  月度：{panel_m['title'].nunique():,} 话题 × {len(panel_m):,} 行")
print(f"  季度：{panel_q_agg['title'].nunique():,} 话题 × {len(panel_q_agg):,} 行")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. 全部 10 个规格定义
# ═══════════════════════════════════════════════════════════════════════════════
ALL_SPECS = [
    # (var_col,         label_zh, freq_zh, df_key, rel_col,      base, bpre, bpost, fe)
    ("total_likes",    "点赞数", "月度", "m", "rel_month",   -1, -12, 24, "title + calendar_ym"),
    ("total_reposts",  "转发数", "月度", "m", "rel_month",   -1, -12, 24, "title + calendar_ym"),
    ("total_comments", "评论数", "月度", "m", "rel_month",   -1, -12, 24, "title + calendar_ym"),
    ("Stance_i",       "态度",   "月度", "m", "rel_month",   -1, -12, 24, "title + calendar_ym"),
    ("Sentiment_i",    "情感",   "月度", "m", "rel_month",   -1, -12, 24, "title + calendar_ym"),
    ("total_likes",    "点赞数", "季度", "q", "rel_quarter", -1, -4,   8, "title + calendar_yq"),
    ("total_reposts",  "转发数", "季度", "q", "rel_quarter", -1, -4,   8, "title + calendar_yq"),
    ("total_comments", "评论数", "季度", "q", "rel_quarter", -1, -4,   8, "title + calendar_yq"),
    ("Stance_i",       "态度",   "季度", "q", "rel_quarter", -1, -4,   8, "title + calendar_yq"),
    ("Sentiment_i",    "情感",   "季度", "q", "rel_quarter", -1, -4,   8, "title + calendar_yq"),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  3. HonestDiD 核心：LP 偏误界
# ═══════════════════════════════════════════════════════════════════════════════
def compute_rm_bounds(Mbar, sigma_pre, n_post, l_vec):
    bound = Mbar * sigma_pre
    m = n_post
    A_ub = np.zeros((2 * m, m)); b_ub = np.full(2 * m, bound)
    A_ub[0, 0] = 1.0; A_ub[1, 0] = -1.0
    for j in range(1, m):
        A_ub[2*j,   j] =  1.0; A_ub[2*j,   j-1] = -1.0
        A_ub[2*j+1, j] = -1.0; A_ub[2*j+1, j-1] =  1.0
    res_max = linprog(-l_vec, A_ub=A_ub, b_ub=b_ub, method="highs")
    bias_ub = -res_max.fun if res_max.success else np.nan
    res_min = linprog( l_vec, A_ub=A_ub, b_ub=b_ub, method="highs")
    bias_lb =  res_min.fun if res_min.success else np.nan
    return bias_lb, bias_ub

def run_honestdid_one(df, var_col, rel_col, base, bpre, bpost, fe,
                      Mbar_grid=None, alpha=0.05):
    if Mbar_grid is None:
        Mbar_grid = np.arange(0, 3.25, 0.25)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    dw = df[df[rel_col] != base].copy()
    sd = dw[var_col].std()
    if sd < 1e-8:
        return None, np.nan, np.nan, np.nan, np.nan
    dw["treat_std"] = dw[var_col] / sd
    dw["topic_num"] = pd.Categorical(dw["title"]).codes
    cal_var = [v.strip() for v in fe.split("+") if v.strip() != "title"][0]
    dw["cal_num"]   = pd.Categorical(dw[cal_var]).codes

    ks = sorted(dw[rel_col].unique())
    pre_ks  = [k for k in ks if k < base]
    post_ks = [k for k in ks if k > base]

    tag = f"{var_col}_{rel_col}"
    dta = os.path.join(TEMP, f"h_{tag}.dta")
    dw[["y_max_score","treat_std",rel_col,"topic_num","cal_num"]].to_stata(
        dta, write_index=False, version=118)

    vnames = []
    gens   = []
    for k in ks:
        vn = f"v_m{abs(int(k))}" if k < 0 else f"v_p{int(k)}"
        gens.append(f"gen {vn} = treat_std * ({rel_col} == {int(k)})")
        vnames.append(vn)
    nv = len(vnames)

    do = f"""
clear all
set more off
use "{dta}", clear
{chr(10).join(gens)}
reghdfe y_max_score {' '.join(vnames)}, absorb(topic_num cal_num) cluster(topic_num)
matrix b = e(b)
matrix b = b[1,1..{nv}]
svmat b, names(b)
keep b*
drop if b1==.
export delimited using "{os.path.join(TEMP, f'hb_{tag}.csv')}", replace
clear
use "{dta}", clear
{chr(10).join(gens)}
reghdfe y_max_score {' '.join(vnames)}, absorb(topic_num cal_num) cluster(topic_num)
matrix V = e(V)
matrix V = V[1..{nv},1..{nv}]
svmat V, names(V)
keep V*
drop if V1==.
export delimited using "{os.path.join(TEMP, f'hV_{tag}.csv')}", replace
"""
    do_path = os.path.join(TEMP, f"h_{tag}.do")
    with open(do_path, "w") as f:
        f.write(do)

    try:
        subprocess.run([STATA_PATH, "-b", "do", do_path],
                       cwd=TEMP, capture_output=True, timeout=300)
    except Exception as e:
        print(f"    Stata 错误：{e}")
        return None, np.nan, np.nan, np.nan, np.nan
    try:
        beta = pd.read_csv(os.path.join(TEMP, f"hb_{tag}.csv")).values.flatten()
        V    = pd.read_csv(os.path.join(TEMP, f"hV_{tag}.csv")).values
    except Exception:
        return None, np.nan, np.nan, np.nan, np.nan
    if len(beta) != nv:
        return None, np.nan, np.nan, np.nan, np.nan

    n_pre = len(pre_ks); n_post = len(post_ks)
    bp = beta[:n_pre]; bq_ = beta[n_pre:]
    Vpost = V[n_pre:, n_pre:]

    bfull = np.append(bp, 0.0)
    sigma = np.max(np.abs(np.diff(bfull))) if len(bfull) > 1 else 0.0
    l = np.ones(n_post) / n_post
    th = l @ bq_; se = np.sqrt(l @ Vpost @ l)

    rows = []
    for M in Mbar_grid:
        blb, bub = compute_rm_bounds(M, sigma, n_post, l)
        rows.append({"Mbar": M, "ci_lb": th - bub - z*se, "ci_ub": th - blb + z*se})
    rdf = pd.DataFrame(rows)
    bd = np.nan
    for _, r in rdf.iterrows():
        if r["ci_lb"] <= 0 <= r["ci_ub"]:
            bd = r["Mbar"]; break
    return rdf, th, se, sigma, bd

# ═══════════════════════════════════════════════════════════════════════════════
#  4. 单规格图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_single(rdf, th, bd, lab, freq, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(rdf["Mbar"], rdf["ci_lb"], rdf["ci_ub"],
                    alpha=.18, color=ECON_COLORS["blue"], label="95% 稳健置信区间")
    ax.plot(rdf["Mbar"], rdf["ci_lb"], color=ECON_COLORS["blue"], lw=1.1)
    ax.plot(rdf["Mbar"], rdf["ci_ub"], color=ECON_COLORS["blue"], lw=1.1)
    ax.axhline(th, color=ECON_COLORS["red"], lw=1.3, ls="--",
               label=f"TWFE 事后均值 = {th:.3f}")
    ax.axhline(0, color="#444", lw=.8)
    if not np.isnan(bd):
        ax.axvline(bd, color=ECON_COLORS.get("green","#2ca02c"), lw=1.2, ls=":",
                   label=f"崩溃值 $\\bar{{M}}^*$ = {bd:.2f}")
    ax.set_xlabel("相对幅度界 $\\bar{M}$", fontsize=10)
    ax.set_ylabel("稳健置信区间", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=True, edgecolor="#cccccc")
    style_ax(ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  5. 遍历全部 10 个规格
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" HonestDiD 敏感性分析（10 个事件研究规格）")
print("="*70)

summary = []
for vc, lab, freq, dk, rc, bas, bp, bpo, fe in ALL_SPECS:
    tag = f"{lab}×{freq}"
    print(f"\n── {tag} ──")
    rdf, th, se, sig, bd = run_honestdid_one(DFS[dk], vc, rc, bas, bp, bpo, fe)
    # p-value for θ̂ under standard parallel trends (M̄=0)
    pval = 2 * (1 - sp_stats.norm.cdf(abs(th / se))) if (se and se > 1e-10) else np.nan
    summary.append({"处理变量": lab, "频率": freq,
                    "θ̂": th, "SE": se, "pval": pval, "σ_pre": sig, "M̄*": bd})
    if rdf is not None:
        print(f"  θ̂={th:.4f}  SE={se:.4f}  σ_pre={sig:.4f}  M̄*={bd:.2f}")
        fn = f"FigR1_HonestDiD_{lab}_{freq}.pdf"
        plot_single(rdf, th, bd, lab, freq, os.path.join(FIGURES, fn))
        print(f"  ✓ {fn}")
    else:
        print(f"  ✗ 跳过")

sdf = pd.DataFrame(summary)

# ═══════════════════════════════════════════════════════════════════════════════
#  7. LaTeX 汇总表
# ═══════════════════════════════════════════════════════════════════════════════
print("生成 LaTeX 汇总表…")

def _stars(p):
    if np.isnan(p): return ""
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

def _fmt_row(r):
    """返回两行 LaTeX：系数+星号行、(SE) 行"""
    if np.isnan(r["θ̂"]):
        coef_str = "—"
        se_str   = ""
    else:
        coef_str = f"{r['θ̂']:.4f}{_stars(r['pval'])}"
        se_str   = f"({r['SE']:.4f})"
    sig = f"{r['σ_pre']:.4f}" if not np.isnan(r["σ_pre"]) else "—"
    mb  = f"{r['M̄*']:.2f}" if not np.isnan(r["M̄*"]) else "—"
    line1 = f"\\quad {r['处理变量']} & {r['频率']} & {coef_str} & {sig} & {mb} \\\\"
    line2 = f" & & {se_str} & & \\\\[3pt]"
    return line1, line2

tx = []
tx.append(r"\begin{table}[htbp]"); tx.append(r"\centering")
tx.append(r"\caption{HonestDiD 敏感性分析汇总：全部事件研究规格}")
tx.append(r"\label{tab:honestdid}"); tx.append(r"\small")
tx.append(r"\begin{tabular}{llccc}"); tx.append(r"\toprule")
tx.append(r"处理变量 & 频率 & $\hat{\theta}$ & $\sigma_{pre}$"
          r" & $\bar{M}^*$ \\")
tx.append(r"\midrule")
tx.append(r"\multicolumn{5}{l}{\textit{热度变量}} \\[2pt]")
for _, r in sdf[sdf["处理变量"].isin(["点赞数","转发数","评论数"])].iterrows():
    l1, l2 = _fmt_row(r)
    tx.append(l1); tx.append(l2)
tx.append(r"[2pt]\multicolumn{5}{l}{\textit{态度变量}} \\[2pt]")
for _, r in sdf[sdf["处理变量"].isin(["态度","情感"])].iterrows():
    l1, l2 = _fmt_row(r)
    tx.append(l1); tx.append(l2)
tx.append(r"\bottomrule")
tx.append(
    r"\multicolumn{5}{p{0.88\textwidth}}{\footnotesize "
    r"注：基于 \citet{rambachan2023} 的相对幅度法。"
    r"$\hat{\theta}$ 为 TWFE 事后平均系数（标准平行趋势下的 $\bar{M}=0$ 推断），"
    r"括号内为标准误（话题层面聚类），***、**、*分别表示在1\%、5\%、10\%水平上显著。"
    r"$\sigma_{pre}$ 为事前系数（含 $\beta_{-1}=0$）的最大一阶差分。"
    r"$\bar{M}^*$ 为 95\% 稳健 CI 首次包含零的阈值："
    r"$\leq 1$ 高度敏感，$1$--$2$ 中等稳健，$> 2$ 高度稳健。"
    r"处理变量均按标准差标准化。"
    r"固定效应：话题 + 日历时间。}"
)
tx.append(r"\end{tabular}"); tx.append(r"\end{table}")
tp = os.path.join(TABLES, "TabR1_HonestDiD敏感性.tex")
with open(tp, "w", encoding="utf-8") as f:
    f.write("\n".join(tx))
print(f"  ✓ {tp}")

print("\n=== HonestDiD 全规格摘要 ===")
for _, r in sdf.iterrows():
    t = "敏感" if r["M̄*"] <= 1 else ("中等" if r["M̄*"] <= 2 else "稳健")
    print(f"  {r['处理变量']}×{r['频率']}：θ̂={r['θ̂']:.4f}，M̄*={r['M̄*']:.2f}（{t}）")
print("\n完成。")
