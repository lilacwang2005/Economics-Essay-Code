"""
研究目的一：稳健性检验 R1–R8
R1  替代热度指标（点赞/转发/评论分项）
R2  替代因变量（二元落实 / 密集边际）
R3  排除反向因果（PreBaseline 控制 + PreBaseline=0 子样本）
R4  安慰剂检验（时间错位，500次）
R5  排除水军干扰（更严格筛选）
R6  多源因变量一致性（占位，等待其他数据）
R7  Heckman 两步法
R8  不同时间窗口
输出（按 PAPER_LANG）：
  results_{lang}/tables/Tab9_稳健性检验.tex / Tab9_RobustnessChecks.tex
  results_{lang}/figures/Fig8_安慰剂检验.pdf / Fig8_PlaceboTest.pdf
  results_{lang}/figures/Fig9_稳健性系数汇总.pdf / Fig9_RobustnessCoefficients.pdf
"""

import os, sys, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyfixest")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyfixest as pf
from scipy import stats

ROOT = "/Users/wangmanlin/Desktop/citizens-will-and-politics"
sys.path.insert(0, os.path.join(ROOT, "code"))
from lang_utils import get_lang, tr, get_results_root, latex_var_label
from table_utils import setup_plot_style, style_ax, ECON_COLORS, ECON_PALETTE, stars

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

# 原始转评赞填 0
for col in ["总点赞数", "总转发数", "总评论数"]:
    tvars[col] = tvars[col].fillna(0)

data = tvars[tvars["BotRatio_i"].fillna(0) <= 0.5].copy()
data = data[data["SolveDegree_score"].notna()].copy()
data["上榜年份"]     = data["上榜年份"].astype(str)
data["category_L1"] = data["category_L1"].fillna("其他")
data["Stance_i"]     = data["Stance_i"].fillna(0)
data["Sentiment_i"]  = data["Sentiment_i"].fillna(0)
data["AngerRatio_i"] = data["AngerRatio_i"].fillna(0)
data["StanceSD_i"]   = data["StanceSD_i"].fillna(data["StanceSD_i"].median())

print(f"  {tr(LANG, '样本量', 'N')}：{len(data):,}")
FE = "category_L1 + 上榜年份"

# ── 多 FE 规范（稳健性用）────────────────────────────────────────────────────
data["policy_year_cross"] = data["category_L1"].astype(str) + "_" + data["上榜年份"].astype(str)
FE_CONFIGS_CROSS = {
    "Baseline":           "category_L1 + 上榜年份",
    "PolicyYearInteract": "policy_year_cross",
    "YearOnly":           "上榜年份",
    "CategoryOnly":       "category_L1",
}

# 基准规格（所有三个热度指标同时纳入）
def base_fit(df=data, fe=FE):
    return pf.feols(
        f"SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
        f"+ Stance_i + Sentiment_i | {fe}",
        data=df, vcov={"CRV1": "category_L1"},
    )

robustness_rows = []

def extract_primary_coef(fit, label, primary_var="总评论数"):
    """提取主要热度系数（以总评论数为代表）。"""
    try:
        coef_df = fit.coef().reset_index(); coef_df.columns = ["term", "coef"]
        se_df   = fit.se().reset_index();   se_df.columns   = ["term", "se"]
        pv_df   = fit.pvalue().reset_index(); pv_df.columns  = ["term", "pval"]
        res = coef_df.merge(se_df).merge(pv_df)
        row = res[res["term"] == primary_var]
        if not len(row):
            row = res.iloc[[0]]
        c, s, p = row["coef"].values[0], row["se"].values[0], row["pval"].values[0]
        return {"label": label, "coef": c, "se": s, "pval": p,
                "ci_low": c - 1.96*s, "ci_high": c + 1.96*s, "stars": stars(p)}
    except Exception:
        return {"label": label, "coef": np.nan, "se": np.nan, "pval": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "stars": ""}

# ── 基准 ──────────────────────────────────────────────────────────────────────
baseline_label = tr(LANG, "基准规格", "Baseline Specification")
try:
    fit_base = base_fit()
    robustness_rows.append(extract_primary_coef(fit_base, baseline_label))
except Exception as e:
    print(f"  {tr(LANG, '⚠ 基准', '⚠ Baseline')}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# R1  替代热度指标（单个分项）
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'R1 替代热度指标…', 'R1 Alternative heat indicators...')}")
r1_specs = {
    tr(LANG, "R1a 仅点赞", "R1a Likes Only"): (
        f"SolveDegree_score ~ 总点赞数 + Stance_i + Sentiment_i | {FE}",
        "总点赞数",
    ),
    tr(LANG, "R1b 仅转发", "R1b Reposts Only"): (
        f"SolveDegree_score ~ 总转发数 + Stance_i + Sentiment_i | {FE}",
        "总转发数",
    ),
    tr(LANG, "R1c 仅评论", "R1c Comments Only"): (
        f"SolveDegree_score ~ 总评论数 + Stance_i + Sentiment_i | {FE}",
        "总评论数",
    ),
}
for label, (fml, pvar) in r1_specs.items():
    try:
        fit = pf.feols(fml, data=data, vcov={"CRV1": "category_L1"})
        robustness_rows.append(extract_primary_coef(fit, label, primary_var=pvar))
        print(f"  ✓ {label}")
    except Exception as e:
        print(f"  ⚠ {label}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# R2  替代因变量
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'R2 替代因变量…', 'R2 Alternative dependent variables...')}")
data["solved_binary"] = (data["SolveDegree_score"].fillna(0) >= 40).astype(int)
r2_specs = {
    tr(LANG, "R2a 二元落实（≥40分）", "R2a Binary Implementation (≥40)"): "solved_binary",
}
for label, dv_var in r2_specs.items():
    try:
        fit = pf.feols(
            f"{dv_var} ~ 总点赞数 + 总转发数 + 总评论数 "
            f"+ Stance_i + Sentiment_i | {FE}",
            data=data, vcov={"CRV1": "category_L1"},
        )
        robustness_rows.append(extract_primary_coef(fit, label))
        print(f"  ✓ {label}")
    except Exception as e:
        print(f"  ⚠ {label}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# R3  排除反向因果
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'R3 排除反向因果…', 'R3 Excluding reverse causality...')}")
sub_no_pre = data[data["PreBaseline_i"] == 0].copy()
if len(sub_no_pre) >= 30:
    try:
        fit_r3b = pf.feols(
            f"SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
            f"+ Stance_i + Sentiment_i | {FE}",
            data=sub_no_pre, vcov={"CRV1": "category_L1"},
        )
        label_r3 = tr(
            LANG,
            f"R3b 无基准期新闻（N={len(sub_no_pre)}）",
            f"R3b No pre-period news (N={len(sub_no_pre)})",
        )
        robustness_rows.append(extract_primary_coef(fit_r3b, label_r3))
        print(f"  ✓ R3b（N={len(sub_no_pre)}）")
    except Exception as e:
        print(f"  ⚠ R3b: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# R4  安慰剂检验（时间错位，500次）
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'R4 安慰剂检验（500次，耗时较长）…', 'R4 Placebo test (500 permutations, may take time)...')}")
N_PLACEBO = 500
placebo_coefs = []

np.random.seed(42)
for i in range(N_PLACEBO):
    df_perm = data.copy()
    df_perm["总评论数"] = np.random.permutation(df_perm["总评论数"].values)
    try:
        fit_p = pf.feols(
            f"SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
            f"+ Stance_i + Sentiment_i | {FE}",
            data=df_perm, vcov={"CRV1": "category_L1"},
        )
        coef_df = fit_p.coef().reset_index(); coef_df.columns = ["term", "coef"]
        c = coef_df[coef_df["term"] == "总评论数"]["coef"].values
        if len(c):
            placebo_coefs.append(c[0])
    except:
        pass
    if (i + 1) % 100 == 0:
        print(f"  {tr(LANG, '进度', 'Progress')}：{i+1}/{N_PLACEBO}", flush=True)

if placebo_coefs and robustness_rows:
    true_coef = robustness_rows[0]["coef"]
    placebo_arr = np.array(placebo_coefs)
    pct_rank = (placebo_arr < true_coef).mean()
    q95 = np.percentile(placebo_arr, 95)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(placebo_arr, bins=40, color=ECON_COLORS["blue"],
            edgecolor="white", linewidth=0.4,
            label=tr(LANG, "置换零分布", "Permutation null distribution"))
    ax.axvline(true_coef, color=ECON_COLORS["red"], linewidth=2, linestyle="-",
               label=tr(LANG, "真实系数", "True coefficient"))
    ax.axvline(q95, color=ECON_COLORS["gray"], linewidth=1.5, linestyle="--",
               label=tr(LANG, "零分布95%分位", "Null distribution 95th percentile"))
    ax.set_xlabel(tr(LANG, "总评论数 系数", "Total Comments Coefficient"))
    ax.set_ylabel(tr(LANG, "频数", "Count"))
    style_ax(ax)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig8_fname = tr(LANG, "Fig8_安慰剂检验.pdf", "Fig8_PlaceboTest.pdf")
    fig.savefig(os.path.join(FIGURES, fig8_fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fig8_fname}（{tr(LANG, '真实系数排名', 'True coef. percentile')}：{pct_rank:.1%}）")
    label_r4 = tr(
        LANG,
        f"R4 安慰剂（百分位={pct_rank:.1%}）",
        f"R4 Placebo (pct={pct_rank:.1%})",
    )
    robustness_rows.append({
        "label": label_r4,
        "coef": true_coef,
        "se": np.std(placebo_arr),
        "pval": 1 - pct_rank,
        "ci_low": np.nan, "ci_high": np.nan,
        "stars": stars(1 - pct_rank),
    })

# ═══════════════════════════════════════════════════════════════════════════════
# R5  排除水军干扰（更严格：BotRatio < 10%）
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'R5 严格水军筛选…', 'R5 Strict bot filtering...')}")
sub_strict = data[data["BotRatio_i"] < 0.10].copy()
if len(sub_strict) >= 30:
    try:
        fit_r5 = base_fit(sub_strict)
        label_r5 = tr(
            LANG,
            f"R5 水军<10%（N={len(sub_strict)}）",
            f"R5 Bot<10% (N={len(sub_strict)})",
        )
        robustness_rows.append(extract_primary_coef(fit_r5, label_r5))
        print(f"  ✓ R5（N={len(sub_strict)}）")
    except Exception as e:
        print(f"  ⚠ R5: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# R7  Heckman 两步法
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'R7 Heckman 两步法…', 'R7 Heckman two-step...')}")
all_topics = tvars[tvars["BotRatio_i"].fillna(0) <= 0.5].copy()
all_topics["is_covered"]   = all_topics["SolveDegree_score"].notna().astype(int)
all_topics["上榜年份"]    = all_topics["上榜年份"].astype(str)
all_topics["category_L1"] = all_topics["category_L1"].fillna("其他")
all_topics["BotRatio_i"] = all_topics["BotRatio_i"].fillna(0)

try:
    from scipy.stats import norm as sp_norm

    fit_step1 = pf.feols(
        "is_covered ~ 总评论数 + BotRatio_i | category_L1 + 上榜年份",
        data=all_topics, vcov="HC1",
    )
    xb  = fit_step1.predict()
    imr = sp_norm.pdf(xb) / np.where(sp_norm.cdf(xb) > 1e-10, sp_norm.cdf(xb), 1e-10)

    data_heck = data.copy()
    imr_series = pd.Series(imr, index=all_topics.index)
    data_heck["IMR"] = imr_series.reindex(data_heck.index)

    if data_heck["IMR"].notna().sum() > 20:
        fit_r7 = pf.feols(
            f"SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
            f"+ Stance_i + Sentiment_i + IMR | {FE}",
            data=data_heck, vcov={"CRV1": "category_L1"},
        )
        label_r7 = tr(LANG, "R7 Heckman 两步法（含IMR）", "R7 Heckman Two-Step (with IMR)")
        robustness_rows.append(extract_primary_coef(fit_r7, label_r7))
        print(f"  ✓ R7 Heckman")
except Exception as e:
    print(f"  ⚠ R7: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# R_FE  固定效应规范选择稳健性
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'R_FE 固定效应规范稳健性…', 'R_FE Fixed-effect specification robustness...')}")
for fe_label, fe_str in FE_CONFIGS_CROSS.items():
    if fe_label == "Baseline":
        continue  # 已在基准中计算
    try:
        fit_fe = base_fit(fe=fe_str)
        label_fe = f"R_FE {fe_label}"
        robustness_rows.append(extract_primary_coef(fit_fe, label_fe))
        print(f"  ✓ {label_fe}")
    except Exception as e:
        print(f"  ⚠ R_FE {fe_label}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# R8  不同时间窗口（Event Study 面板）
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'R8 不同时间窗口（Event Study 面板）…', 'R8 Different time windows (Event Study panel)...')}")
panel_path = os.path.join(CLEAN, "purpose1_event_panel.csv")
if os.path.exists(panel_path):
    panel = pd.read_csv(panel_path, encoding="utf-8-sig")
    panel = panel[panel["BotRatio_i"].fillna(0) <= 0.5].copy()
    panel["上榜年份"]    = panel["上榜年份"].astype(str)
    panel["category_L1"] = panel["category_L1"].fillna("其他")
    panel["PreBaseline_i"] = panel["PreBaseline_i"].fillna(0)
    panel["总评论数"]    = panel["总评论数"].fillna(0)
    panel["article_date"] = pd.to_datetime(panel["article_date"])
    panel["hot_date"]     = pd.to_datetime(panel["hot_date"])
    panel["rel_month"] = (
        panel["article_date"].dt.to_period("M") - panel["hot_date"].dt.to_period("M")
    ).apply(lambda x: x.n)
    panel["calendar_ym"] = panel["article_date"].dt.to_period("M").astype(str)

    window_specs = [
        (tr(LANG, "R8a [-6,+12]",          "R8a [-6,+12]"),          (-6, 12)),
        (tr(LANG, "R8b [-12,+24]（基准）",  "R8b [-12,+24] (Baseline)"), (-12, 24)),
    ]
    for window_label, (wmin, wmax) in window_specs:
        sub_w = panel[(panel["rel_month"] >= wmin) & (panel["rel_month"] <= wmax)].copy()
        sub_w["post"]          = (sub_w["rel_month"] >= 0).astype(float)
        sub_w["treat_x_post"]  = sub_w["总评论数"] * sub_w["post"]
        try:
            fit_w = pf.feols(
                "y_max_score ~ treat_x_post | title + calendar_ym",
                data=sub_w, vcov={"CRV1": "title"},
            )
            coef_df = fit_w.coef().reset_index(); coef_df.columns = ["term", "coef"]
            se_df   = fit_w.se().reset_index();   se_df.columns   = ["term", "se"]
            pv_df   = fit_w.pvalue().reset_index(); pv_df.columns  = ["term", "pval"]
            res = coef_df.merge(se_df).merge(pv_df)
            r = res[res["term"] == "treat_x_post"]
            if len(r):
                c, s, p = r["coef"].values[0], r["se"].values[0], r["pval"].values[0]
                robustness_rows.append({
                    "label": window_label, "coef": c, "se": s, "pval": p,
                    "ci_low": c - 1.96*s, "ci_high": c + 1.96*s, "stars": stars(p),
                })
                print(f"  ✓ {window_label}")
        except Exception as e:
            print(f"  ⚠ {window_label}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 图9  稳健性系数汇总
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, '生成图9 稳健性汇总…', 'Generating Fig 9 robustness summary...')}")
rob_df = pd.DataFrame([r for r in robustness_rows if not np.isnan(r.get("coef", np.nan))])

if len(rob_df):
    fig, ax = plt.subplots(figsize=(8, max(5, 0.5 * len(rob_df))))
    y = np.arange(len(rob_df))
    colors_rob = [
        ECON_COLORS["red"] if r["label"] == baseline_label
        else ECON_COLORS["blue"]
        for _, r in rob_df.iterrows()
    ]
    ax.barh(
        y, rob_df["coef"],
        xerr=[(rob_df["coef"] - rob_df["ci_low"]).clip(lower=0),
              (rob_df["ci_high"] - rob_df["coef"]).clip(lower=0)],
        color=colors_rob, ecolor="#333333", capsize=3, height=0.6,
        error_kw={"linewidth": 1},
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rob_df["label"], fontsize=9)
    ax.set_xlabel(tr(LANG, "总评论数 回归系数（±95% CI）", "Total Comments Coefficient (±95% CI)"))
    style_ax(ax)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.yaxis.grid(False)
    from matplotlib.patches import Patch
    base_patch = Patch(facecolor=ECON_COLORS["red"],
                       label=tr(LANG, "基准规格", "Baseline"))
    alt_patch  = Patch(facecolor=ECON_COLORS["blue"],
                       label=tr(LANG, "稳健性规格", "Robustness"))
    ax.legend(handles=[base_patch, alt_patch], fontsize=10)
    fig.tight_layout()
    fig9_fname = tr(LANG, "Fig9_稳健性系数汇总.pdf", "Fig9_RobustnessCoefficients.pdf")
    fig.savefig(os.path.join(FIGURES, fig9_fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fig9_fname}")

# ── Tab9 ──────────────────────────────────────────────────────────────────────
print(tr(LANG, "生成 Tab9…", "Generating Tab9..."))

tab9_caption = tr(
    LANG,
    "稳健性检验汇总：热搜热度对政策落实的系数稳定性",
    "Robustness Checks Summary: Coefficient Stability of Hot Search Heat on Policy Implementation",
)
tab9_note = tr(
    LANG,
    "注：因变量均为落实程度（0-100连续）。系数为总评论数（R1除外）。标准误在领域层面聚类。"
    "***、**、*分别表示在1\\%、5\\%、10\\%水平上显著。",
    "Note: Dependent variable is Solve Degree (0-100 continuous) unless noted. "
    "Coefficient reported is for Total Comments (except R1). "
    "Standard errors clustered at the domain level. "
    "***, **, * denote significance at 1\\%, 5\\%, 10\\% levels.",
)

col_spec = tr(LANG, "规格",   "Specification")
col_coef = tr(LANG, "系数",   "Coef.")
col_se   = tr(LANG, "标准误", "SE")
col_pval = tr(LANG, "p值",    "p-value")
col_sig  = tr(LANG, "显著性", "Sig.")

tab9_lines = []
tab9_lines.append(r"\begin{table}[htbp]")
tab9_lines.append(r"\centering")
tab9_lines.append(f"\\caption{{{tab9_caption}}}")
tab9_lines.append(r"\label{tab:robustness}")
tab9_lines.append(r"\small")
tab9_lines.append(r"\begin{tabular}{lc}")
tab9_lines.append(r"\toprule")
tab9_lines.append(f"{col_spec} & {col_coef} \\\\")
tab9_lines.append(r"\midrule")
for _, r in rob_df.iterrows():
    c, s = r['coef'], r['se']
    if not np.isnan(c):
        # Auto-select precision: use enough dp so coefficient is not "0.000"
        dp = 3
        while dp < 7 and abs(c) > 0 and abs(round(c, dp)) < 10**(-dp + 0.5):
            dp += 1
        coef_str = f"{c:.{dp}f}{r['stars']}"
        se_str = f"({s:.{dp}f})" if not np.isnan(s) else ""
    else:
        coef_str = "—"
        se_str = ""
    tab9_lines.append(f"{latex_var_label(r['label'])} & {coef_str} \\\\")
    tab9_lines.append(f" & {se_str} \\\\[3pt]")
tab9_lines.append(r"\bottomrule")
tab9_lines.append(
    f"\\multicolumn{{2}}{{p{{0.98\\textwidth}}}}{{\\footnotesize {tab9_note}}}"
)
tab9_lines.append(r"\end{tabular}")
tab9_lines.append(r"\end{table}")

tab9_fname = tr(LANG, "Tab9_稳健性检验.tex", "Tab9_RobustnessChecks.tex")
with open(os.path.join(TABLES, tab9_fname), "w", encoding="utf-8") as f:
    f.write("\n".join(tab9_lines))
print(f"  ✓ {tab9_fname}")

print(f"\n{tr(LANG, '完成！', 'Done!')}")

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
