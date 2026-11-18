#!/usr/bin/env python3
"""
fix2_contdid_comparison.py — 覆盖全部 10 个规格的连续处理非参数剂量-反应检验
═══════════════════════════════════════════════════════════════════════════════
规格清单（5 处理变量 × 2 频率 = 10）：
  热度：总点赞数 / 总转发数 / 总评论数 × 月度 / 季度
  态度：Stance_i / Sentiment_i           × 月度 / 季度

思路：将每个连续处理变量离散化为三分位组，以低组为参照，
     分组估计事件研究，与线性 TWFE 预测对比。
     遵循 Callaway, Goodman-Bacon & Sant'Anna (2024) 的建议。

输出：
  results_zh/figures/FigR2_剂量反应_{label}_{freq}.pdf （10 张双面板图）
  results_zh/figures/FigR2_剂量反应汇总.pdf            （1 张汇总偏差图）
  results_zh/tables/TabR2_剂量反应检验.tex              （1 张汇总表）
"""

import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
#  1. 数据准备（与 fix1 一致）
# ═══════════════════════════════════════════════════════════════════════════════
print("加载面板…")
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

ALL_SPECS = [
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
#  2. 对一个规格执行完整分析
# ═══════════════════════════════════════════════════════════════════════════════
def run_contdid_one(df, var_col, rel_col, base, bpre, bpost, fe):
    """
    返回 (all_coefs_df, q33, q67, n_low, n_mid, n_high, dose_medians)
    all_coefs_df 列：rel_k, coef_linear, coef_mid, se_mid, coef_high, se_high
    """
    dw = df.copy()

    # 话题层面三分位
    topic_lv = dw.groupby("title")[var_col].first().reset_index()
    q33, q67 = topic_lv[var_col].quantile([1/3, 2/3]).values
    # 处理态度/情感变量可能有负值
    topic_lv["dg"] = pd.cut(topic_lv[var_col],
                            bins=[-np.inf, q33, q67, np.inf],
                            labels=["low","mid","high"]).astype(str)
    dw = dw.merge(topic_lv[["title","dg"]], on="title", how="left")
    dw["is_mid"]  = (dw["dg"] == "mid").astype(int)
    dw["is_high"] = (dw["dg"] == "high").astype(int)

    sd = dw[var_col].std()
    dw["treat_std"] = dw[var_col] / sd if sd > 1e-8 else 0.0

    dw["topic_num"] = pd.Categorical(dw["title"]).codes
    cal_var = [v.strip() for v in fe.split("+") if v.strip() != "title"][0]
    dw["cal_num"]   = pd.Categorical(dw[cal_var]).codes

    ks = sorted([k for k in dw[rel_col].unique() if k != base])

    # ── Stata do ───────────────────────────────────────────────────────────
    tag = f"{var_col}_{rel_col}"
    dta = os.path.join(TEMP, f"cd_{tag}.dta")
    dw[["y_max_score","treat_std","is_mid","is_high",
        rel_col,"topic_num","cal_num"]].to_stata(dta, write_index=False, version=118)

    gL, gM, gH = [], [], []
    vL, vM, vH = [], [], []
    for k in ks:
        suf = f"m{abs(int(k))}" if k < 0 else f"p{int(k)}"
        gL.append(f"gen iL_{suf} = treat_std * ({rel_col}=={int(k)})")
        gM.append(f"gen iM_{suf} = is_mid    * ({rel_col}=={int(k)})")
        gH.append(f"gen iH_{suf} = is_high   * ({rel_col}=={int(k)})")
        vL.append(f"iL_{suf}"); vM.append(f"iM_{suf}"); vH.append(f"iH_{suf}")
    nv = len(vL)

    do = f"""
clear all
set more off
use "{dta}", clear

// 模型 A：线性 TWFE
{chr(10).join(gL)}
reghdfe y_max_score {' '.join(vL)}, absorb(topic_num cal_num) cluster(topic_num)
matrix bA = e(b)
matrix bA = bA[1,1..{nv}]
svmat bA, names(bA)
preserve
keep bA*
drop if bA1==.
export delimited using "{os.path.join(TEMP, f'cd_bL_{tag}.csv')}", replace
restore
drop bA*

// 模型 B：分组 ES
{chr(10).join(gM)}
{chr(10).join(gH)}
reghdfe y_max_score {' '.join(vM)} {' '.join(vH)}, absorb(topic_num cal_num) cluster(topic_num)

// mid 系数
matrix bM = e(b)
matrix bM = bM[1,1..{nv}]
svmat bM, names(bM)
preserve
keep bM*
drop if bM1==.
export delimited using "{os.path.join(TEMP, f'cd_bM_{tag}.csv')}", replace
restore
drop bM*

// high 系数
matrix bH = e(b)
matrix bH = bH[1,{nv+1}..{2*nv}]
svmat bH, names(bH)
preserve
keep bH*
drop if bH1==.
export delimited using "{os.path.join(TEMP, f'cd_bH_{tag}.csv')}", replace
restore
drop bH*

// mid SE
matrix seAll = vecdiag(e(V))
matrix seM = seAll[1,1..{nv}]
forvalues i = 1/{nv} {{
    matrix seM[1,`i'] = sqrt(seM[1,`i'])
}}
svmat seM, names(seM)
preserve
keep seM*
drop if seM1==.
export delimited using "{os.path.join(TEMP, f'cd_sM_{tag}.csv')}", replace
restore
drop seM*

// high SE
matrix seH = seAll[1,{nv+1}..{2*nv}]
forvalues i = 1/{nv} {{
    matrix seH[1,`i'] = sqrt(seH[1,`i'])
}}
svmat seH, names(seH)
preserve
keep seH*
drop if seH1==.
export delimited using "{os.path.join(TEMP, f'cd_sH_{tag}.csv')}", replace
restore
"""
    do_path = os.path.join(TEMP, f"cd_{tag}.do")
    with open(do_path, "w") as f:
        f.write(do)

    try:
        subprocess.run([STATA_PATH, "-b", "do", do_path],
                       cwd=TEMP, capture_output=True, timeout=300)
    except Exception as e:
        print(f"    Stata 错误：{e}")
        return None, q33, q67, 0, 0, 0, {}

    try:
        bL = pd.read_csv(os.path.join(TEMP, f"cd_bL_{tag}.csv")).values.flatten()
        bM = pd.read_csv(os.path.join(TEMP, f"cd_bM_{tag}.csv")).values.flatten()
        bH = pd.read_csv(os.path.join(TEMP, f"cd_bH_{tag}.csv")).values.flatten()
        sM = pd.read_csv(os.path.join(TEMP, f"cd_sM_{tag}.csv")).values.flatten()
        sH = pd.read_csv(os.path.join(TEMP, f"cd_sH_{tag}.csv")).values.flatten()
    except Exception:
        return None, q33, q67, 0, 0, 0, {}

    # 组装结果
    c = pd.DataFrame({"rel_k": ks, "coef_lin": bL, "coef_mid": bM,
                       "se_mid": sM, "coef_high": bH, "se_high": sH})
    base_row = pd.DataFrame([{"rel_k": base, "coef_lin": 0, "coef_mid": 0,
                               "se_mid": 0, "coef_high": 0, "se_high": 0}])
    c = pd.concat([c, base_row]).sort_values("rel_k").reset_index(drop=True)

    dose_med = topic_lv.groupby("dg")[var_col].median().to_dict()
    n_low  = (topic_lv["dg"]=="low").sum()
    n_mid  = (topic_lv["dg"]=="mid").sum()
    n_high = (topic_lv["dg"]=="high").sum()

    return c, q33, q67, n_low, n_mid, n_high, dose_med


# ═══════════════════════════════════════════════════════════════════════════════
#  3. 单规格双面板图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_contdid_single(coefs, dose_med, treat_sd, lab, freq, path,
                        horizons_m=None, horizons_q=None, is_quarterly=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 分组 Event Study
    ax = axes[0]
    m = coefs
    ax.fill_between(m["rel_k"], m["coef_mid"]-1.96*m["se_mid"],
                    m["coef_mid"]+1.96*m["se_mid"],
                    alpha=.12, color=ECON_COLORS["blue"])
    ax.plot(m["rel_k"], m["coef_mid"], "-o", color=ECON_COLORS["blue"],
            ms=3, lw=1.2, label="中剂量组（vs 低）")
    ax.fill_between(m["rel_k"], m["coef_high"]-1.96*m["se_high"],
                    m["coef_high"]+1.96*m["se_high"],
                    alpha=.12, color=ECON_COLORS["red"])
    ax.plot(m["rel_k"], m["coef_high"], "-s", color=ECON_COLORS["red"],
            ms=3, lw=1.2, label="高剂量组（vs 低）")
    ax.axhline(0, color="#333", lw=.8)
    ax.axvline(-1, color="#333", lw=.8, ls="--")
    rel_label = "相对季度" if is_quarterly else "相对月份"
    ax.set_xlabel(f"{rel_label}（$k$）", fontsize=10)
    ax.set_ylabel("分组交互项系数", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=True, edgecolor="#cccccc")
    ax.text(.02, .98, "(a) 分组事件研究", transform=ax.transAxes,
            fontsize=10, va="top", fontweight="bold")
    style_ax(ax)

    # (b) 剂量-反应曲线
    ax2 = axes[1]
    horizons = horizons_q if is_quarterly else horizons_m
    if horizons is None:
        horizons = [1, 3, 6, 12] if not is_quarterly else [1, 2, 4, 8]
    # 过滤存在的 horizons
    horizons = [h for h in horizons if h in coefs["rel_k"].values]
    palette = plt.cm.viridis(np.linspace(.2, .85, max(len(horizons), 1)))

    x_labels = ["低", "中", "高"]
    x_pos = [0, 1, 2]

    for i, hk in enumerate(horizons):
        row = coefs[coefs["rel_k"] == hk].iloc[0]
        y_np = [0, row["coef_mid"], row["coef_high"]]
        se_np = [0, row["se_mid"], row["se_high"]]

        # 线性预测：根据 dose_med 推算
        low_med  = dose_med.get("low",  0)
        mid_med  = dose_med.get("mid",  0)
        high_med = dose_med.get("high", 0)
        sd = (high_med - low_med) if high_med != low_med else 1
        # 线性 TWFE 系数 × 标准化剂量差
        y_lin = [0,
                 row["coef_lin"] * (mid_med  - low_med) / (sd if sd else 1),
                 row["coef_lin"] * (high_med - low_med) / (sd if sd else 1)]

        offset = (i - len(horizons)/2) * 0.08
        ax2.errorbar([x + offset for x in x_pos], y_np,
                     yerr=[1.96*s for s in se_np],
                     fmt="o", color=palette[i], ms=5, capsize=3,
                     label=f"$k$={hk}（非参数）")
        ax2.plot([x + offset for x in x_pos], y_lin, "--",
                 color=palette[i], lw=1, alpha=.5)

    ax2.axhline(0, color="#333", lw=.8)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("剂量组", fontsize=10)
    ax2.set_ylabel("相对于低剂量组的效应", fontsize=10)
    ax2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, frameon=True, edgecolor="#cccccc")
    ax2.text(.02, .98, "(b) 剂量-反应曲线", transform=ax2.transAxes,
             fontsize=10, va="top", fontweight="bold")
    style_ax(ax2)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  4. 遍历全部 10 个规格
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" 剂量-反应非参数检验（10 个事件研究规格）")
print("="*70)

summary = []
for vc, lab, freq, dk, rc, bas, bp, bpo, fe in ALL_SPECS:
    tag = f"{lab}×{freq}"
    is_q = (dk == "q")
    print(f"\n── {tag} ──")

    coefs, q33, q67, nl, nm, nh, dm = run_contdid_one(
        DFS[dk], vc, rc, bas, bp, bpo, fe)

    if coefs is None:
        print(f"  ✗ 跳过")
        summary.append({"处理变量": lab, "频率": freq,
                        "Q33": q33, "Q67": q67,
                        "偏差_mid": np.nan, "偏差_high": np.nan,
                        "NP_mid": np.nan, "SE_mid": np.nan, "pval_mid": np.nan,
                        "NP_high": np.nan, "SE_high": np.nan, "pval_high": np.nan,
                        "Lin_mid": np.nan, "Lin_high": np.nan,
                        "N_low": 0, "N_mid": 0, "N_high": 0})
        continue

    # 事后平均偏差
    post = coefs[coefs["rel_k"] > bas]
    n_post_k = len(post)
    avg_mid  = post["coef_mid"].mean()
    avg_high = post["coef_high"].mean()
    avg_lin  = post["coef_lin"].mean()

    # 事后平均 SE（忽略协方差的近似：sqrt(mean(se_k^2) / n_post)）
    se_avg_mid  = np.sqrt((post["se_mid"]**2).mean()  / n_post_k) if n_post_k > 0 else np.nan
    se_avg_high = np.sqrt((post["se_high"]**2).mean() / n_post_k) if n_post_k > 0 else np.nan

    # p-value
    from scipy import stats as sp_stats
    pval_mid  = 2*(1-sp_stats.norm.cdf(abs(avg_mid /se_avg_mid)))  if se_avg_mid  > 1e-10 else np.nan
    pval_high = 2*(1-sp_stats.norm.cdf(abs(avg_high/se_avg_high))) if se_avg_high > 1e-10 else np.nan

    # 线性预测的组效应
    low_m  = dm.get("low",  q33 / 2)
    mid_m  = dm.get("mid",  (q33 + q67) / 2)
    high_m = dm.get("high", q67 * 1.5)
    span = high_m - low_m if high_m != low_m else 1
    pred_mid  = avg_lin * (mid_m  - low_m) / span
    pred_high = avg_lin * (high_m - low_m) / span
    dev_mid   = avg_mid  - pred_mid
    dev_high  = avg_high - pred_high

    print(f"  Q33={q33:.2f}  Q67={q67:.2f}  "
          f"偏差：mid={dev_mid:+.4f}  high={dev_high:+.4f}")

    summary.append({"处理变量": lab, "频率": freq,
                    "Q33": q33, "Q67": q67,
                    "偏差_mid": dev_mid, "偏差_high": dev_high,
                    "NP_mid": avg_mid, "SE_mid": se_avg_mid, "pval_mid": pval_mid,
                    "NP_high": avg_high, "SE_high": se_avg_high, "pval_high": pval_high,
                    "Lin_mid": pred_mid, "Lin_high": pred_high,
                    "N_low": nl, "N_mid": nm, "N_high": nh})

    fn = f"FigR2_剂量反应_{lab}_{freq}.pdf"
    sd_ = DFS[dk][vc].std()
    plot_contdid_single(
        coefs, dm, sd_, lab, freq, os.path.join(FIGURES, fn),
        horizons_m=[3, 6, 12, 24], horizons_q=[1, 2, 4, 8],
        is_quarterly=is_q)
    print(f"  ✓ {fn}")

sdf = pd.DataFrame(summary)

# ═══════════════════════════════════════════════════════════════════════════════
#  5. 汇总偏差图
# ═══════════════════════════════════════════════════════════════════════════════
print("\n生成汇总偏差图…")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
v = sdf.dropna(subset=["偏差_mid"]).reset_index(drop=True)
v["label"] = v["处理变量"] + "（" + v["频率"] + "）"
yp = np.arange(len(v))

for i, (col, title, ax) in enumerate([
    ("偏差_mid",  "(a) 中剂量组偏差", axes[0]),
    ("偏差_high", "(b) 高剂量组偏差", axes[1]),
]):
    cols = [ECON_COLORS["blue"] if abs(x) < 0.5
            else (ECON_COLORS.get("orange","#ff7f0e") if abs(x) < 1.5
                  else ECON_COLORS["red"])
            for x in v[col]]
    ax.barh(yp, v[col], color=cols, height=.6, alpha=.8)
    ax.axvline(0, color="#333", lw=.8)
    ax.set_yticks(yp); ax.set_yticklabels(v["label"], fontsize=8)
    ax.set_xlabel("偏差（非参数 − 线性预测）", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.invert_yaxis()
    style_ax(ax)

fig.tight_layout()
fp = os.path.join(FIGURES, "FigR2_剂量反应汇总.pdf")
fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close()
print(f"  ✓ {fp}")

# ═══════════════════════════════════════════════════════════════════════════════
#  6. LaTeX 汇总表
# ═══════════════════════════════════════════════════════════════════════════════
print("生成 LaTeX 汇总表…")

def _stars(p):
    if p is None or np.isnan(p): return ""
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

def _fmt_contdid_rows(r):
    """返回两行 LaTeX：系数+星号行、(SE) 行"""
    if np.isnan(r.get("偏差_mid", np.nan)):
        return (f"\\quad {r['处理变量']} & {r['频率']} "
                f"& — & — & — & — & — & — \\\\"), ""
    # 非参数系数 + 星号
    c_mid  = f"{r['NP_mid']:.3f}{_stars(r['pval_mid'])}"
    c_high = f"{r['NP_high']:.3f}{_stars(r['pval_high'])}"
    # 线性预测（无独立 SE，仅展示数值）
    l_mid  = f"{r['Lin_mid']:.3f}"
    l_high = f"{r['Lin_high']:.3f}"
    # 偏差
    d_mid  = f"{r['偏差_mid']:+.3f}"
    d_high = f"{r['偏差_high']:+.3f}"
    # SE 行
    se_mid  = f"({r['SE_mid']:.3f})" if not np.isnan(r.get("SE_mid", np.nan)) else ""
    se_high = f"({r['SE_high']:.3f})" if not np.isnan(r.get("SE_high", np.nan)) else ""
    line1 = (f"\\quad {r['处理变量']} & {r['频率']} "
             f"& {c_mid} & {c_high} & {l_mid} & {l_high} & {d_mid} & {d_high} \\\\")
    line2 = f" & & {se_mid} & {se_high} & & & & \\\\[3pt]"
    return line1, line2

tx = []
tx.append(r"\begin{table}[htbp]"); tx.append(r"\centering")
tx.append(r"\caption{剂量-反应非参数检验汇总：分组 Event Study 与线性 TWFE 的对比}")
tx.append(r"\label{tab:contdid}"); tx.append(r"\small")
tx.append(r"\begin{tabular}{llcccccc}")
tx.append(r"\toprule")
tx.append(r" & & \multicolumn{2}{c}{非参数（分组 ES）}"
          r" & \multicolumn{2}{c}{线性 TWFE 预测}"
          r" & \multicolumn{2}{c}{偏差} \\")
tx.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
tx.append(r"处理变量 & 频率 & 中组 & 高组 & 中组 & 高组 & 中组 & 高组 \\")
tx.append(r"\midrule")

tx.append(r"\multicolumn{8}{l}{\textit{热度变量}} \\[2pt]")
for _, r in sdf[sdf["处理变量"].isin(["点赞数","转发数","评论数"])].iterrows():
    l1, l2 = _fmt_contdid_rows(r)
    tx.append(l1)
    if l2:
        tx.append(l2)

tx.append(r"[2pt]\multicolumn{8}{l}{\textit{态度变量}} \\[2pt]")
for _, r in sdf[sdf["处理变量"].isin(["态度","情感"])].iterrows():
    l1, l2 = _fmt_contdid_rows(r)
    tx.append(l1)
    if l2:
        tx.append(l2)

tx.append(r"\bottomrule")
tx.append(
    r"\multicolumn{8}{p{0.95\textwidth}}{\footnotesize "
    r"注：各处理变量按话题层面三分位分为低/中/高剂量组，低剂量组为参照。"
    "\u201c非参数\u201d列为分组 Event Study 事后平均系数（$k>0$），"
    r"括号内为事后平均标准误（话题层面聚类），"
    r"***、**、*分别表示在1\%、5\%、10\%水平上显著。"
    "\u201c线性 TWFE 预测\u201d列为基准线性模型在各组中位数处的预测值。"
    "\u201c偏差\u201d列为两者之差，反映线性函数形式的拟合偏差。"
    r"固定效应：话题 + 日历时间。}"
)
tx.append(r"\end{tabular}"); tx.append(r"\end{table}")

tp = os.path.join(TABLES, "TabR2_剂量反应检验.tex")
with open(tp, "w", encoding="utf-8") as f:
    f.write("\n".join(tx))
print(f"  ✓ {tp}")

print("\n=== 剂量-反应非参数检验全规格摘要 ===")
for _, r in sdf.iterrows():
    dm = r.get("偏差_mid", np.nan)
    dh = r.get("偏差_high", np.nan)
    if np.isnan(dm):
        print(f"  {r['处理变量']}×{r['频率']}：跳过")
    else:
        t = "良好" if max(abs(dm), abs(dh)) < 0.5 else \
            ("一定非线性" if max(abs(dm), abs(dh)) < 1.5 else "较大偏差")
        print(f"  {r['处理变量']}×{r['频率']}：偏差 mid={dm:+.3f} high={dh:+.3f}（{t}）")

print("\n完成。")
