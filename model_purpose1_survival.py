"""
研究目的一：补充模型
S1  Cox 比例风险模型 + Kaplan-Meier 生存曲线
S2  愤怒效应（AngerRatio_i）
S3  意见分歧效应（StanceSD_i × 热度）
S4  领域异质性（热度 × 领域哑变量）
输出（按 PAPER_LANG）：
  results_{lang}/tables/Tab6_Cox生存分析.tex      / Tab6_CoxSurvival.tex
  results_{lang}/tables/Tab7_愤怒与分歧效应.tex   / Tab7_AngerDiscordEffects.tex
  results_{lang}/tables/Tab8_领域异质性.tex        / Tab8_DomainHeterogeneity.tex
  results_{lang}/figures/Fig6_KaplanMeier生存曲线.pdf / Fig6_KaplanMeier_Survival.pdf
  results_{lang}/figures/Fig7_领域异质性系数.pdf   / Fig7_DomainHeterogeneity.pdf
"""

import os, sys, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyfixest")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyfixest as pf

ROOT = "/Users/wangmanlin/Desktop/citizens-will-and-politics"
sys.path.insert(0, os.path.join(ROOT, "code"))
from lang_utils import get_lang, tr, get_results_root, latex_var_label
from table_utils import setup_plot_style, style_ax, ECON_COLORS, ECON_PALETTE, stars, fmt_coef

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
data = pd.read_csv(os.path.join(CLEAN, "purpose1_topic_vars.csv"), encoding="utf-8-sig")

# 样本筛选
data = data[data["BotRatio_i"].fillna(0) <= 0.5].copy()
data["上榜年份"]    = data["上榜年份"].astype(str)
data["category_L1"] = data["category_L1"].fillna("其他")

# 原始转评赞填 0
for col in ["总点赞数", "总转发数", "总评论数"]:
    data[col] = data[col].fillna(0)

# 控制变量
data["Stance_i"]     = data["Stance_i"].fillna(0)
data["Sentiment_i"]  = data["Sentiment_i"].fillna(0)
data["AngerRatio_i"] = data["AngerRatio_i"].fillna(0)
data["StanceSD_i"]          = data["StanceSD_i"].fillna(data["StanceSD_i"].median())
data["StanceSD_x_Comments"] = data["StanceSD_i"] * data["总评论数"]   # S3 连续交互项

# 按评论热度高低分组（中位数切分，用于 KM 图）
heat_med = data["总评论数"].median()
high_label = tr(LANG, "高热度", "High Heat")
low_label  = tr(LANG, "低热度", "Low Heat")
data["heat_group"] = (data["总评论数"] >= heat_med).map({True: high_label, False: low_label})

print(f"  {tr(LANG, '样本量', 'N')}：{len(data):,}")

FE = "category_L1 + 上榜年份"

# ═══════════════════════════════════════════════════════════════════════════════
# S1  Kaplan-Meier + Cox
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'S1 生存分析…', 'S1 Survival Analysis...')}")


def kaplan_meier(T, E):
    """返回 (times, survival) 用于绘图。"""
    df = pd.DataFrame({"T": T, "E": E}).sort_values("T").reset_index(drop=True)
    n = len(df)
    times, surv = [0], [1.0]
    S = 1.0
    at_risk = n
    for _, row in df.iterrows():
        if row["E"] == 1:
            S *= (1 - 1 / at_risk)
        at_risk -= 1
        times.append(row["T"])
        surv.append(S)
    return np.array(times), np.array(surv)


# DaysToSolve_i / Censored_i 来自 topic_vars
surv_data = data[data["DaysToSolve_i"].notna()].copy()
surv_data["T_days"] = surv_data["DaysToSolve_i"].clip(lower=1)
surv_data["event"]  = (surv_data["Censored_i"] == 0).astype(int)  # 1=落实

fig, ax = plt.subplots(figsize=(8, 5))
colors = [ECON_COLORS["blue"], ECON_COLORS["red"]]
for grp, color in zip([high_label, low_label], colors):
    sub = surv_data[surv_data["heat_group"] == grp]
    t, s = kaplan_meier(sub["T_days"].values, sub["event"].values)
    ax.step(t / 30, s, where="post", color=color, linewidth=1.8, label=grp)

ax.set_xlabel(tr(LANG, "距上热搜月数", "Months since Hot Search"))
ax.set_ylabel(tr(LANG, "未落实概率",   "Probability of Non-implementation"))
ax.set_xlim(left=0)
ax.set_ylim(0, 1.05)
style_ax(ax)
ax.legend(fontsize=10)
fig.tight_layout()
fig6_fname = tr(LANG, "Fig6_KaplanMeier生存曲线.pdf", "Fig6_KaplanMeier_Survival.pdf")
fig.savefig(os.path.join(FIGURES, fig6_fname), dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ {fig6_fname}")

# ── Cox 比例风险模型 ──────────────────────────────────────────────────────────
cox_rows = []
method_note = ""
try:
    from lifelines import CoxPHFitter
    cox_data = surv_data[["T_days", "event", "总点赞数", "总转发数", "总评论数",
                           "Stance_i", "Sentiment_i", "PreBaseline_i",
                           "category_L1", "上榜年份"]].dropna().copy()
    cox_data["_cluster"] = cox_data["category_L1"]          # 聚类列，get_dummies 前保存
    cox_data = pd.get_dummies(cox_data, columns=["category_L1", "上榜年份"], drop_first=True)
    cph = CoxPHFitter()
    try:
        # cluster_col 传列名字符串，lifelines 会自动从 df 中 pop 该列
        cph.fit(cox_data, duration_col="T_days", event_col="event",
                cluster_col="_cluster")
    except TypeError:                                        # 旧版 lifelines 不支持 cluster_col
        cph.fit(cox_data.drop(columns=["_cluster"]),
                duration_col="T_days", event_col="event")
    summary = cph.summary[["coef", "se(coef)", "p"]].reset_index()
    summary.columns = [
        tr(LANG, "变量", "Variable"),
        tr(LANG, "系数", "Coef."),
        tr(LANG, "标准误", "SE"),
        tr(LANG, "p值", "p-value"),
    ]
    summary[tr(LANG, "显著性", "Sig.")] = summary[tr(LANG, "p值", "p-value")].apply(stars)
    cox_rows = summary.values.tolist()
    method_note = tr(
        LANG,
        "Cox 比例风险模型（lifelines）",
        "Cox Proportional Hazards Model (lifelines)",
    )
    print(f"  {tr(LANG, '使用 lifelines CoxPHFitter', 'Using lifelines CoxPHFitter')}")
except ImportError:
    surv_data["log_T"] = np.log(surv_data["T_days"].clip(lower=1))
    fit_cox = pf.feols(
        f"log_T ~ 总点赞数 + 总转发数 + 总评论数 + Stance_i + Sentiment_i | {FE}",
        data=surv_data[surv_data["event"] == 1],
        vcov={"CRV1": "category_L1"},
    )
    method_note = tr(
        LANG,
        "加速失效时间模型（AFT 近似，仅落实子样本）",
        "Accelerated Failure Time model (AFT approximation, implemented-only subsample)",
    )
    print(f"  {tr(LANG, 'lifelines 未安装，使用 AFT 近似', 'lifelines not installed, using AFT approximation')}")

col_var = tr(LANG, "变量",   "Variable")
col_coe = tr(LANG, "系数",   "Coef.")
col_se  = tr(LANG, "标准误", "SE")
col_sig = tr(LANG, "显著性", "Sig.")

tab6_caption = tr(LANG, "Cox 比例风险模型（政策落实速度）", "Cox Proportional Hazards Model (Policy Implementation Speed)")
tab6_note = tr(
    LANG,
    f"注：{method_note}。***、**、*分别表示在1\\%、5\\%、10\\%水平上显著。",
    f"Note: {method_note}. ***, **, * denote significance at 1\\%, 5\\%, 10\\% levels.",
)

tab6_lines = []
tab6_lines.append(r"\begin{table}[htbp]")
tab6_lines.append(r"\centering")
tab6_lines.append(f"\\caption{{{tab6_caption}}}")
tab6_lines.append(r"\label{tab:cox}")
tab6_lines.append(r"\small")
tab6_lines.append(r"\begin{tabular}{lc}")
tab6_lines.append(r"\toprule")
tab6_lines.append(f"{col_var} & {col_coe} \\\\")
tab6_lines.append(r"\midrule")
if cox_rows:
    for row in cox_rows[:10]:
        # row: [var, coef, se, pval, sig]
        var_str  = latex_var_label(str(row[0]))
        sig_str  = str(row[4]) if len(row) > 4 else ""
        c = row[1] if isinstance(row[1], (int, float)) else np.nan
        s = row[2] if isinstance(row[2], (int, float)) else np.nan
        if not np.isnan(c):
            dp = 3
            while dp < 7 and abs(c) > 0 and abs(round(c, dp)) < 10**(-dp + 0.5):
                dp += 1
            coef_str = f"{c:.{dp}f}{sig_str}"
            se_str = f"({s:.{dp}f})" if not np.isnan(s) else ""
        else:
            coef_str = "—"
            se_str = ""
        tab6_lines.append(f"{var_str} & {coef_str} \\\\")
        tab6_lines.append(f" & {se_str} \\\\[3pt]")
else:
    tab6_lines.append(
        tr(LANG,
           r"参见 AFT 近似结果（log\_T OLS） \\",
           r"See AFT approximation result (log\_T OLS) \\")
    )
tab6_lines.append(r"\bottomrule")
tab6_lines.append(
    f"\\multicolumn{{2}}{{p{{0.85\\textwidth}}}}{{\\footnotesize {tab6_note}}}"
)
tab6_lines.append(r"\end{tabular}")
tab6_lines.append(r"\end{table}")

tab6_fname = tr(LANG, "Tab6_Cox生存分析.tex", "Tab6_CoxSurvival.tex")
with open(os.path.join(TABLES, tab6_fname), "w", encoding="utf-8") as f:
    f.write("\n".join(tab6_lines))
print(f"  ✓ {tab6_fname}")

# ═══════════════════════════════════════════════════════════════════════════════
# S2  愤怒效应  S3  意见分歧效应
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'S2/S3 愤怒与分歧效应…', 'S2/S3 Anger and Discord Effects...')}")
base_fml = (
    f"SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
    f"+ Stance_i + Sentiment_i | {FE}"
)

col_labels_s23 = {
    "(1)": tr(LANG, "(1) 基准",      "(1) Baseline"),
    "(2)": tr(LANG, "(2) +愤怒",     "(2) +Anger"),
    "(3)": tr(LANG, "(3) +分歧",     "(3) +Discord"),
    "(4)": tr(LANG, "(4) +分歧交互", "(4) +Discord×Heat"),
}
fmls_s23 = {
    col_labels_s23["(1)"]: base_fml,
    col_labels_s23["(2)"]: (
        f"SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
        f"+ Stance_i + Sentiment_i + AngerRatio_i | {FE}"
    ),
    col_labels_s23["(3)"]: (
        f"SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
        f"+ Stance_i + Sentiment_i + StanceSD_i | {FE}"
    ),
    col_labels_s23["(4)"]: (
        f"SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
        f"+ Stance_i + Sentiment_i + StanceSD_i + StanceSD_x_Comments | {FE}"
    ),
}

fits_s23 = {}
for name, fml in fmls_s23.items():
    try:
        fits_s23[name] = pf.feols(fml, data=data, vcov={"CRV1": "category_L1"})
    except Exception as e:
        print(f"  {tr(LANG, '⚠', '⚠')} {name}: {e}")

# Build hand-crafted Tab7
# Key variables to show
key_vars_s23 = [
    ("总评论数",   tr(LANG, "总评论数",     "Total Comments")),
    ("总点赞数",   tr(LANG, "总点赞数",     "Total Likes")),
    ("总转发数",   tr(LANG, "总转发数",     "Total Reposts")),
    ("Stance_i",   tr(LANG, "态度得分",     "Stance Score")),
    ("Sentiment_i",tr(LANG, "情感得分",     "Sentiment Score")),
    ("AngerRatio_i",tr(LANG,"愤怒比例",     "Anger Ratio")),
    ("StanceSD_i",          tr(LANG, "意见分歧（SD）",          "Opinion Discord (SD)")),
    ("StanceSD_x_Comments", tr(LANG, "分歧×评论数（交互项）",   "Discord×Comments (interaction)")),
]

col_names_s23 = list(fits_s23.keys())
fits_s23_list = list(fits_s23.values())
ncols_s23 = 1 + len(col_names_s23)

tab7_caption = tr(LANG, "愤怒效应与意见分歧效应（S2, S3）", "Anger Effect and Opinion Discord Effect (S2, S3)")
tab7_label   = "tab:anger_discord"
tab7_note    = tr(
    LANG,
    "注：因变量为落实程度（0-100连续）。括号内为在领域层面聚类的稳健标准误。"
    "固定效应：领域 + 年份。***、**、*分别表示在1\\%、5\\%、10\\%水平上显著。",
    "Note: Dependent variable is Solve Degree (0-100 continuous). "
    "Standard errors clustered at the domain level in parentheses. "
    "Fixed effects: domain + year. ***, **, * denote significance at 1\\%, 5\\%, 10\\% levels.",
)

tab7_lines = []
tab7_lines.append(r"\begin{table}[htbp]")
tab7_lines.append(r"\centering")
tab7_lines.append(f"\\caption{{{tab7_caption}}}")
tab7_lines.append(f"\\label{{{tab7_label}}}")
tab7_lines.append(r"\small")
tab7_lines.append(r"\begin{tabular}{l" + "c" * len(col_names_s23) + "}")
tab7_lines.append(r"\toprule")
tab7_lines.append(" & " + " & ".join(col_names_s23) + r" \\")
tab7_lines.append(r"\midrule")

def extract_coef_se_str(fit, varname):
    """Return (coef_str, se_str) for a given variable from a pyfixest fit."""
    if fit is None:
        return "—", ""
    try:
        coefs = fit.coef()
        ses   = fit.se()
        pvals = fit.pvalue()
        if varname not in coefs.index:
            return "—", ""
        c, s, p = coefs[varname], ses[varname], pvals[varname]
        c_str, se_str = fmt_coef(c, s, p, decimals=4)
        return c_str, se_str
    except Exception:
        return "—", ""

for varname, varlabel in key_vars_s23:
    coef_row = [latex_var_label(varlabel)]
    se_row   = [""]
    for fit in fits_s23_list:
        c_str, se_str = extract_coef_se_str(fit, varname)
        coef_row.append(c_str)
        se_row.append(se_str)
    tab7_lines.append(" & ".join(coef_row) + r" \\")
    tab7_lines.append(" & ".join(se_row)   + r" \\[3pt]")

tab7_lines.append(r"\midrule")
n_row  = [tr(LANG, "观测值", "Observations")]
r2_row = ["$R^2$"]
for fit in fits_s23_list:
    if fit is not None:
        try:
            n_row.append(f"{fit._N:,}")
            r2_row.append(f"{fit._r2:.3f}")
        except Exception:
            n_row.append("—")
            r2_row.append("—")
    else:
        n_row.append("—")
        r2_row.append("—")
tab7_lines.append(" & ".join(n_row)  + r" \\")
tab7_lines.append(" & ".join(r2_row) + r" \\")
tab7_lines.append(r"\midrule")
fe_domain = tr(LANG, "领域固定效应", "Domain FE")
fe_year   = tr(LANG, "年份固定效应", "Year FE")
tab7_lines.append(fe_domain + " & " + " & ".join([r"\checkmark"] * len(col_names_s23)) + r" \\")
tab7_lines.append(fe_year   + " & " + " & ".join([r"\checkmark"] * len(col_names_s23)) + r" \\")
tab7_lines.append(r"\bottomrule")
tab7_lines.append(
    f"\\multicolumn{{{ncols_s23}}}{{p{{0.98\\textwidth}}}}{{\\footnotesize {tab7_note}}}"
)
tab7_lines.append(r"\end{tabular}")
tab7_lines.append(r"\end{table}")

tab7_fname = tr(LANG, "Tab7_愤怒与分歧效应.tex", "Tab7_AngerDiscordEffects.tex")
with open(os.path.join(TABLES, tab7_fname), "w", encoding="utf-8") as f:
    f.write("\n".join(tab7_lines))
print(f"  ✓ {tab7_fname}")

# ═══════════════════════════════════════════════════════════════════════════════
# S4  领域异质性
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{tr(LANG, 'S4 领域异质性…', 'S4 Domain Heterogeneity...')}")
domains = data["category_L1"].value_counts()
domains = domains[(domains >= 10) & (domains.index != "无")].index.tolist()

het_results = []
for domain in domains:
    sub = data[data["category_L1"] == domain].copy()
    if len(sub) < 15:
        continue
    try:
        fit = pf.feols(
            "SolveDegree_score ~ 总点赞数 + 总转发数 + 总评论数 "
            "+ Stance_i + Sentiment_i | 上榜年份",
            data=sub, vcov="HC1",
        )
        coef_df = fit.coef().reset_index(); coef_df.columns = ["term", "coef"]
        se_df   = fit.se().reset_index();   se_df.columns   = ["term", "se"]
        res = coef_df.merge(se_df, on="term")
        heat_row = res[res["term"] == "总评论数"]
        if len(heat_row):
            coef = heat_row["coef"].values[0]
            se   = heat_row["se"].values[0]
            het_results.append({"domain": domain, "coef": coef, "se": se, "n": len(sub)})
    except Exception as e:
        print(f"  {tr(LANG, '⚠ 领域', '⚠ Domain')} {domain}: {e}")

if het_results:
    het_df = pd.DataFrame(het_results).sort_values("coef")
    het_df["ci_low"]  = het_df["coef"] - 1.96 * het_df["se"]
    het_df["ci_high"] = het_df["coef"] + 1.96 * het_df["se"]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.5 * len(het_df))))
    y = np.arange(len(het_df))
    ax.barh(y, het_df["coef"], xerr=1.96 * het_df["se"],
            color=ECON_COLORS["blue"], ecolor="#333333",
            capsize=3, height=0.6, error_kw={"linewidth": 1})
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    n_label = tr(LANG, "N", "N")
    ax.set_yticklabels([f"{r['domain']}（{n_label}={r['n']}）" for _, r in het_df.iterrows()])
    ax.set_xlabel(tr(LANG, "总评论数 回归系数（±95% CI）", "Total Comments Coefficient (±95% CI)"))
    style_ax(ax)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig7_fname = tr(LANG, "Fig7_领域异质性系数.pdf", "Fig7_DomainHeterogeneity.pdf")
    fig.savefig(os.path.join(FIGURES, fig7_fname), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fig7_fname}")

    het_df["显著性"] = (het_df["coef"].abs() / het_df["se"]).apply(
        lambda t: "***" if t > 2.576 else ("**" if t > 1.960 else ("*" if t > 1.645 else ""))
    )

    tab8_caption = tr(
        LANG,
        "领域异质性：热度对政策落实的分领域效应",
        "Domain Heterogeneity: Effect of Heat on Policy Implementation by Domain",
    )
    tab8_note = tr(
        LANG,
        "注：各领域分别估计，系数为总评论数，控制年份固定效应，HC1稳健标准误。"
        "***、**、*分别表示在1\\%、5\\%、10\\%水平上显著。",
        "Note: Estimated separately by domain; coefficient is for Total Comments, "
        "controlling for year fixed effects with HC1 robust SE. "
        "***, **, * denote significance at 1\\%, 5\\%, 10\\% levels.",
    )
    col_domain = tr(LANG, "领域",   "Domain")
    col_coef   = tr(LANG, "系数",   "Coef.")
    col_se     = tr(LANG, "标准误", "SE")
    col_n      = tr(LANG, "观测数", "N")
    col_sig    = tr(LANG, "显著性", "Sig.")

    tab8_lines = []
    tab8_lines.append(r"\begin{table}[htbp]")
    tab8_lines.append(r"\centering")
    tab8_lines.append(f"\\caption{{{tab8_caption}}}")
    tab8_lines.append(r"\label{tab:heterogeneity}")
    tab8_lines.append(r"\small")
    tab8_lines.append(r"\begin{tabular}{lcc}")
    tab8_lines.append(r"\toprule")
    tab8_lines.append(f"{col_domain} & {col_coef} & {col_n} \\\\")
    tab8_lines.append(r"\midrule")
    for _, r in het_df.sort_values("coef", ascending=False).iterrows():
        c, se_val = r['coef'], r['se']
        dp = 3
        while dp < 7 and abs(c) > 0 and abs(round(c, dp)) < 10**(-dp + 0.5):
            dp += 1
        se_str = f"({se_val:.{dp}f})" if np.isfinite(se_val) else ""
        tab8_lines.append(
            f"{r['domain']} & {c:.{dp}f}{r['显著性']} & {int(r['n'])} \\\\"
        )
        tab8_lines.append(f" & {se_str} & \\\\[3pt]")
    tab8_lines.append(r"\bottomrule")
    tab8_lines.append(
        f"\\multicolumn{{3}}{{p{{0.85\\textwidth}}}}{{\\footnotesize {tab8_note}}}"
    )
    tab8_lines.append(r"\end{tabular}")
    tab8_lines.append(r"\end{table}")

    tab8_fname = tr(LANG, "Tab8_领域异质性.tex", "Tab8_DomainHeterogeneity.tex")
    with open(os.path.join(TABLES, tab8_fname), "w", encoding="utf-8") as f:
        f.write("\n".join(tab8_lines))
    print(f"  ✓ {tab8_fname}")

print(f"\n{tr(LANG, '完成！', 'Done!')}")

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
