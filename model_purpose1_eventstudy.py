"""
研究目的一：Event Study（连续型 DID）
标准规范：话题 FE + 日历时间 FE + 相对时间虚拟变量（k=-1 归一化，两端 binning）

热度 × 频率（月度/季度）× 三分项 = 6 张图（Fig3a-c 月度，Fig4a-c 季度）
态度/情感 × 频率（月度/季度）= 4 张图（Fig5a-d）
M4：热度（评论数）× 态度交互（月度 + 季度）

输出（按 PAPER_LANG）：
  results_{lang}/tables/Tab4_热度EventStudy.tex  / Tab4_HeatEventStudy.tex
  results_{lang}/tables/Tab5_态度EventStudy.tex  / Tab5_AttitudeEventStudy.tex
  results_{lang}/tables/Tab5b_热度态度交互.tex   / Tab5b_HeatStanceInteraction.tex
  results_{lang}/figures/Fig3a-c / Fig4a-c / Fig5a-d
"""

import os, sys, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyfixest")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyfixest as pf
from scipy import stats as sp_stats

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

# ── 0. 加载面板并预处理 ──────────────────────────────────────────────────────
print(tr(LANG, "加载 Event Study 面板…", "Loading Event Study panel..."))
panel = pd.read_csv(os.path.join(CLEAN, "purpose1_event_panel.csv"), encoding="utf-8-sig")
panel = panel[panel["BotRatio_i"].fillna(0) <= 0.5].copy()

panel["article_date"] = pd.to_datetime(panel["article_date"])
panel["hot_date"]     = pd.to_datetime(panel["hot_date"])
panel["rel_month"] = (
    panel["article_date"].dt.to_period("M") - panel["hot_date"].dt.to_period("M")
).apply(lambda x: x.n)

# 日历时间 FE 用字段
panel["calendar_ym"] = panel["article_date"].dt.to_period("M").astype(str)

for col in ["总点赞数", "总转发数", "总评论数"]:
    panel[col] = panel[col].fillna(0)
panel["Stance_i"]    = panel["Stance_i"].fillna(0)
panel["Sentiment_i"] = panel["Sentiment_i"].fillna(0)

# 新增 FE 变量（供 Spec 2-5 使用）
if "category_L1" not in panel.columns:
    panel["category_L1"] = "其他"
else:
    panel["category_L1"] = panel["category_L1"].fillna("其他")
panel["year"]        = panel["article_date"].dt.year
panel["policy_year"] = panel["category_L1"].astype(str) + "_" + panel["year"].astype(str)
panel["dow"]         = panel["article_date"].dt.dayofweek.astype(str)

# ── 月度聚合 ──────────────────────────────────────────────────────────────────
monthly = panel.groupby(["title", "rel_month"], as_index=False).agg(
    y_max_score = ("y_max_score",  "max"),
    总点赞数    = ("总点赞数",     "first"),
    总转发数    = ("总转发数",     "first"),
    总评论数    = ("总评论数",     "first"),
    Stance_i    = ("Stance_i",     "first"),
    Sentiment_i = ("Sentiment_i",  "first"),
    BotRatio_i  = ("BotRatio_i",   "first"),
    calendar_ym = ("calendar_ym",  "first"),   # 绝对日历月 FE
    category_L1  = ("category_L1",  "first"),
    policy_year  = ("policy_year",  "first"),
    dow          = ("dow",          "first"),
)
print(
    f"  {tr(LANG,'面板话题数','Panel topics')}：{monthly['title'].nunique():,}，"
    f"{tr(LANG,'月度行数','monthly rows')}：{len(monthly):,}"
)

# Spec5B 用：月度数据 + 季度日历时间 FE
monthly["calendar_yq"] = monthly["calendar_ym"].str[:4] + "Q" + (
    (monthly["calendar_ym"].str[5:7].astype(int) - 1) // 3 + 1
).astype(str)

# 月度窗口 [-12, +24]，两端 bin
panel_m = monthly[(monthly["rel_month"] >= -12) & (monthly["rel_month"] <= 24)].copy()

# ── 季度聚合 ──────────────────────────────────────────────────────────────────
panel_q = monthly.copy()
panel_q["rel_quarter"] = (panel_q["rel_month"] / 3).apply(np.floor).astype(int)
panel_q["calendar_yq"] = panel_q["calendar_ym"].str[:4] + "Q" + (
    (panel_q["calendar_ym"].str[5:7].astype(int) - 1) // 3 + 1
).astype(str)   # e.g. "2021Q2"

panel_q = panel_q[(panel_q["rel_quarter"] >= -4) & (panel_q["rel_quarter"] <= 8)].copy()
panel_q_agg = panel_q.groupby(["title", "rel_quarter"], as_index=False).agg(
    y_max_score  = ("y_max_score",  "max"),
    总点赞数     = ("总点赞数",     "first"),
    总转发数     = ("总转发数",     "first"),
    总评论数     = ("总评论数",     "first"),
    Stance_i     = ("Stance_i",     "first"),
    Sentiment_i  = ("Sentiment_i",  "first"),
    calendar_yq  = ("calendar_yq",  "first"),   # 绝对日历季度 FE
    category_L1  = ("category_L1",  "first"),
    policy_year  = ("policy_year",  "first"),
)
print(f"  {tr(LANG,'季度行数','Quarterly rows')}：{len(panel_q_agg):,}")


# ── 事件研究估计函数（标准规范）───────────────────────────────────────────────
def run_event_study(df, y_var, treat_var, rel_col="rel_month", base=-1,
                    bin_pre=None, bin_post=None,
                    fe="title + calendar_ym"):
    """
    标准 event study：话题 FE + 日历时间 FE + 相对时间交互项。
    bin_pre：k <= bin_pre 的所有期归为左端 bin；None 表示不做 binning。
    bin_post：k >= bin_post 的所有期归为右端 bin。
    基准期 k=base（默认 -1）系数归零（直接从数据中删去该期）。
    """
    df = df.copy()
    df_nb = df[df[rel_col] != base].copy()

    # 标准化处理变量（避免大数值导致矩阵数值奇异；系数单位变为"每1SD"）
    treat_scale = df_nb[treat_var].std()
    if treat_scale > 1e-8:
        df_nb[treat_var] = df_nb[treat_var] / treat_scale

    # 两端 binning
    k_vals = df_nb[rel_col].values.copy().astype(float)
    if bin_pre is not None:
        k_vals = np.where(k_vals <= bin_pre, bin_pre, k_vals)
    if bin_post is not None:
        k_vals = np.where(k_vals >= bin_post, bin_post, k_vals)
    df_nb["k_binned"] = k_vals

    rel_vals = sorted(df_nb["k_binned"].unique())

    def k_to_col(k):
        if bin_pre is not None and k == bin_pre:
            return "inter_pre_bin"
        if bin_post is not None and k == bin_post:
            return "inter_post_bin"
        return f"inter_p{int(k)}" if k >= 0 else f"inter_m{int(abs(k))}"

    new_cols = {k: k_to_col(k) for k in rel_vals}
    extra = {col: df_nb[treat_var] * (df_nb["k_binned"] == k).astype(float)
             for k, col in new_cols.items()}
    df_nb = pd.concat([df_nb, pd.DataFrame(extra, index=df_nb.index)], axis=1)
    inter_cols = list(new_cols.values())

    fml = f"{y_var} ~ {' + '.join(inter_cols)} | {fe}"
    try:
        fit = pf.feols(fml, data=df_nb, vcov={"CRV1": "title"})
    except Exception as e:
        print(f"  {tr(LANG,'⚠ 回归失败','⚠ Regression failed')}：{e}")
        return None, None

    coef_df = fit.coef().reset_index(); coef_df.columns = ["term", "coef"]
    se_df   = fit.se().reset_index();   se_df.columns   = ["term", "se"]
    res = coef_df.merge(se_df, on="term")
    res = res[res["term"].str.startswith("inter_")].copy()

    def parse_k(term):
        if term == "inter_pre_bin":
            return float(bin_pre)
        if term == "inter_post_bin":
            return float(bin_post)
        s = term[len("inter_"):]
        return float(s[1:]) if s.startswith("p") else -float(s[1:])

    res["rel_month"] = res["term"].apply(parse_k)
    res["ci_low"]    = res["coef"] - 1.96 * res["se"]
    res["ci_high"]   = res["coef"] + 1.96 * res["se"]
    res["is_bin"]    = res["term"].isin(["inter_pre_bin", "inter_post_bin"])

    base_row = pd.DataFrame([{
        "term": "inter_base", "coef": 0.0, "se": 0.0,
        "rel_month": float(base), "ci_low": 0.0, "ci_high": 0.0, "is_bin": False,
    }])
    res = pd.concat([res, base_row]).sort_values("rel_month").reset_index(drop=True)
    return res, fit


def plot_event_study(coefs, title, xlabel, save_path, base=-1):
    fig, ax = plt.subplots(figsize=(8, 5))
    color = ECON_COLORS["blue"]

    # 95% CI 阴影（排除 bin 端点，避免折线被拉伸）
    non_bin = coefs[~coefs["is_bin"]]
    ax.fill_between(non_bin["rel_month"], non_bin["ci_low"], non_bin["ci_high"],
                    alpha=0.15, color=color)

    # 整条折线（含 bin 端点）
    ax.plot(coefs["rel_month"], coefs["coef"], "-", color=color, linewidth=1.5)

    # 普通点：圆形
    pts = coefs[~coefs["is_bin"]]
    ax.plot(pts["rel_month"], pts["coef"], "o", color=color, markersize=4)

    # Bin 端点：方形，区分标注
    bins = coefs[coefs["is_bin"]]
    ax.plot(bins["rel_month"], bins["coef"], "s", color=color, markersize=6,
            label=tr(LANG, "binned 端点", "Binned endpoint"))

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(base, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(tr(LANG, "交互项系数 β", "Interaction Coefficient β"))
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(save_path)}")


def get_n_r2(fit):
    if fit is None:
        return "—", "—"
    try:
        n = f"{fit._N:,}"
    except Exception:
        n = "—"
    try:
        r2 = f"{fit._r2:.3f}"
    except Exception:
        r2 = "—"
    return n, r2


def avg_beta(coefs_df, post=True):
    if coefs_df is None:
        return "—"
    sub = coefs_df[coefs_df["rel_month"] > 0] if post else coefs_df[coefs_df["rel_month"] < 0]
    sub = sub[~sub["is_bin"]] if "is_bin" in sub.columns else sub
    if len(sub) == 0:
        return "—"
    return f"{sub['coef'].mean():.4f}"


def _stars_from_pval(pval):
    return "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))


def avg_beta_se(coefs_df, fit, post=True):
    """Return (coef_str_with_stars, se_str_in_parens) for averaged ES coefficients."""
    if coefs_df is None or fit is None:
        return "—", ""
    sub = coefs_df[coefs_df["rel_month"] > 0] if post else coefs_df[coefs_df["rel_month"] < 0]
    sub = sub[~sub["is_bin"]] if "is_bin" in sub.columns else sub
    if len(sub) == 0:
        return "—", ""
    avg = sub["coef"].mean()
    terms = sub["term"].tolist()
    try:
        V = fit.vcov()
        V_sub = V.loc[terms, terms].values
        n = len(terms)
        ones = np.ones(n)
        var_avg = float(ones @ V_sub @ ones) / (n ** 2)
        se_avg = np.sqrt(max(var_avg, 0))
    except Exception:
        se_avg = sub["se"].mean() / np.sqrt(len(sub))
    if se_avg > 0:
        t_stat = abs(avg / se_avg)
        pval = 2 * (1 - sp_stats.norm.cdf(t_stat))
        st = _stars_from_pval(pval)
    else:
        st = ""
    return f"{avg:.4f}{st}", f"({se_avg:.4f})"


# ── 辅助函数：前趋势 Wald F 检验 ─────────────────────────────────────────────
def pretrend_test(fit, coefs_df, rel_col="rel_month"):
    """检验所有事前（k<0，非 bin）系数联合为零。"""
    if fit is None or coefs_df is None:
        return None
    pre = coefs_df[(coefs_df[rel_col] < 0) & (~coefs_df["is_bin"])].copy()
    if len(pre) == 0:
        return None
    try:
        beta = pre["coef"].values
        se   = np.where(pre["se"].values == 0, 1e-10, pre["se"].values)
        wald = float(np.sum((beta / se) ** 2) / len(beta))
        from scipy.stats import f as f_dist
        try:
            nobs = int(fit._N); npar = int(fit._k)
        except Exception:
            nobs, npar = 1000, len(beta) + 2
        pval = float(1 - f_dist.cdf(wald, len(beta), max(1, nobs - npar)))
        return {"F_stat": wald, "p_value": pval, "df": len(beta)}
    except Exception as e:
        print(f"  前趋势检验失败: {e}")
        return None


# ── Imputation Estimator (Borusyak, Jaravel & Spiess 2024) ──────────────────
try:
    from did2s import did2s as _did2s_func
    HAS_DID2S = True
except ImportError:
    HAS_DID2S = False


def run_imputation_estimator(df, y_var, treat_var, rel_col="rel_month",
                              base=-1, bin_pre=None, bin_post=None,
                              fe="title + calendar_ym", n_boot=200):
    """
    Imputation Estimator：
      1. 仅用事前（rel<0）数据估计 y ~ 1 | FE，获得固定效应
      2. 用 FE 对事后数据预测反事实 Ŷ_it
      3. τ_it = y_it − Ŷ_it
      4. 按相对时间 k 分组平均 → β_k
      5. Bootstrap（cluster on title，200次）估计 SE
    """
    df = df.copy()
    fe_vars = [v.strip() for v in fe.split("+")]
    pre_df  = df[df[rel_col] < 0].copy()
    post_df = df[df[rel_col] >= 0].copy()

    if len(pre_df) < 30 or post_df.empty:
        print(f"  ⚠ Imputation: 样本不足（pre={len(pre_df)}, post={len(post_df)}）")
        return None, None

    def _one_pass(data_pre, data_post):
        try:
            fit_ = pf.feols(f"{y_var} ~ 1 | {fe}", data=data_pre)
            fe_dict = fit_.fixef()
        except Exception as e_:
            return None, None
        y_hat = np.zeros(len(data_post))
        for fv in fe_vars:
            if fv in fe_dict:
                y_hat += data_post[fv].map(fe_dict[fv]).fillna(0).values
        tau_post = data_post[y_var].values - y_hat
        k_post   = data_post[rel_col].values.astype(float)
        if bin_post is not None:
            k_post = np.where(k_post >= bin_post, float(bin_post), k_post)

        tau_pre = fit_.resid()
        k_pre   = data_pre[rel_col].values.astype(float)
        if bin_pre is not None:
            k_pre = np.where(k_pre <= bin_pre, float(bin_pre), k_pre)
        mask = data_pre[rel_col].values != base

        # pyfixest 可能因单例删除导致 resid 长度与 data_pre 不一致
        try:
            used_idx = fit_._data.index
            pre_aligned = data_pre.loc[used_idx].copy()
            pre_aligned["_tau"] = tau_pre
            k_pre_a = pre_aligned[rel_col].values.astype(float)
            if bin_pre is not None:
                k_pre_a = np.where(k_pre_a <= bin_pre, float(bin_pre), k_pre_a)
            mask_a = pre_aligned[rel_col].values != base
            rows_pre = pd.DataFrame({
                "k_b": k_pre_a[mask_a], "_tau": pre_aligned["_tau"].values[mask_a],
                "title": pre_aligned["title"].values[mask_a],
            })
        except Exception:
            if len(tau_pre) == len(data_pre):
                rows_pre = pd.DataFrame({
                    "k_b": k_pre[mask], "_tau": tau_pre[mask],
                    "title": data_pre["title"].values[mask],
                })
            else:
                rows_pre = pd.DataFrame(columns=["k_b", "_tau", "title"])
        rows_post = pd.DataFrame({
            "k_b": k_post, "_tau": tau_post,
            "title": data_post["title"].values,
        })
        return pd.concat([rows_pre, rows_post], ignore_index=True), fit_

    result_df, fit_pre = _one_pass(pre_df, post_df)
    if result_df is None:
        return None, None

    coefs_s = result_df.groupby("k_b")["_tau"].mean()
    coefs_s[float(base)] = 0.0
    coefs_s = coefs_s.sort_index()

    # Bootstrap SE（固定 FE 估计步骤，仅对 τ 重采样）
    titles    = result_df["title"].unique()
    title_dfs = {t: result_df[result_df["title"] == t] for t in titles}
    boot_store = {k: [] for k in coefs_s.index}
    np.random.seed(42)
    for _ in range(n_boot):
        boot_t    = np.random.choice(titles, size=len(titles), replace=True)
        boot_rows = pd.concat([title_dfs[t] for t in boot_t], ignore_index=True)
        b_coefs   = boot_rows.groupby("k_b")["_tau"].mean()
        for k in boot_store:
            if k in b_coefs.index:
                boot_store[k].append(b_coefs[k])

    rows = []
    for k in sorted(coefs_s.index):
        c   = float(coefs_s[k])
        bts = boot_store.get(k, [])
        se  = float(np.std(bts)) if len(bts) >= 5 else np.nan
        is_b = ((bin_pre  is not None and k == float(bin_pre)) or
                (bin_post is not None and k == float(bin_post)))
        rows.append({
            "term": f"imp_{k}", "coef": c,
            "se": 0.0 if np.isnan(se) else se,
            "rel_month": k,
            "ci_low":  c - 1.96 * se if not np.isnan(se) else c,
            "ci_high": c + 1.96 * se if not np.isnan(se) else c,
            "is_bin": is_b,
        })
    return pd.DataFrame(rows), fit_pre


# ── 辅助函数：多规范叠加比较图 ───────────────────────────────────────────────
from matplotlib.lines import Line2D

def plot_multi_spec_comparison(results_dict, save_path, title_str):
    """在同一图中叠加多个 Specification 的 Event Study 曲线（不含 CI）。"""
    palette = [ECON_COLORS.get(c, "#555555")
               for c in ["blue", "red", "green", "orange", "purple", "gray"]]
    styles  = ["-", "--", "-.", ":", "-", "--"]
    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    for idx, (spec_name, (coefs, _)) in enumerate(results_dict.items()):
        if coefs is None:
            continue
        col = palette[idx % len(palette)]
        ls  = styles[idx % len(styles)]
        nb  = coefs[~coefs["is_bin"]]
        ax.plot(nb["rel_month"], nb["coef"], linestyle=ls, color=col,
                linewidth=1.5, marker="o", markersize=3)
        handles.append(Line2D([0], [0], color=col, linestyle=ls,
                               linewidth=1.5, label=spec_name))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(-1, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel(tr(LANG, "相对月份（0=上热搜当月）", "Relative Month (0 = Hot Search Month)"))
    ax.set_ylabel(tr(LANG, "交互项系数 β", "Interaction Coefficient β"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(save_path)}")


# ── 辅助函数：多规范汇总 LaTeX 表 ────────────────────────────────────────────
def make_spec_comparison_table(results_dict, caption, label, fname,
                                rel_col="rel_month"):
    col_spec = tr(LANG, "规范",     "Specification")
    col_pre  = tr(LANG, r"事前均值 $\bar{\beta}_{k<0}$",
                        r"Pre-avg $\bar{\beta}_{k<0}$")
    col_post = tr(LANG, r"事后均值 $\bar{\beta}_{k>0}$",
                        r"Post-avg $\bar{\beta}_{k>0}$")
    col_n    = tr(LANG, "观测值", "N")
    note = tr(
        LANG,
        r"注：处理变量为评论数×月度面板。Spec5 为插补估计量（BJS 2024），"
        r"SE 通过 Bootstrap（200次，话题层面聚类）计算；其余规范 SE 在话题层面聚类（CRV1）。",
        r"Note: Treatment variable = total comments, monthly panel. "
        r"Spec5 uses the Imputation Estimator (BJS 2024) with bootstrap SE "
        r"(200 reps, clustered at topic level). Other specs use CRV1 SE.",
    )
    lines = [r"\begin{table}[htbp]", r"\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             f"{col_spec} & {col_pre} & {col_post} & {col_n} & $R^2$ \\\\",
             r"\midrule"]
    for spec_name, (coefs, fit) in results_dict.items():
        pre_c, pre_se   = avg_beta_se(coefs, fit, post=False)
        post_c, post_se = avg_beta_se(coefs, fit, post=True)
        n_v, r2_v = get_n_r2(fit)
        lines.append(f"{spec_name} & {pre_c} & {post_c} & {n_v} & {r2_v} \\\\")
        lines.append(f" & {pre_se} & {post_se} & & \\\\[3pt]")
    lines += [r"\bottomrule",
              f"\\multicolumn{{5}}{{p{{0.98\\textwidth}}}}{{\\footnotesize {note}}}",
              r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TABLES, fname), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  ✓ {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 热度 Event Study：月度（Fig3a-c）+ 季度（Fig4a-c）
# ══════════════════════════════════════════════════════════════════════════════
print(tr(LANG, "\n运行热度 Event Study…", "\nRunning heat Event Study..."))

xlabel_m = tr(LANG, "相对月份（0=上热搜当月）", "Relative Month (0 = Hot Search Month)")
xlabel_q = tr(LANG, "相对季度（0=上热搜当季）", "Relative Quarter (0 = Hot Search Quarter)")

likes_lbl    = tr(LANG, "点赞数", "Likes")
reposts_lbl  = tr(LANG, "转发数", "Reposts")
comments_lbl = tr(LANG, "评论数", "Comments")
stance_lbl   = tr(LANG, "态度",   "Stance")
sent_lbl     = tr(LANG, "情感",   "Sentiment")

# (treat_var, label, freq, fname, title_str, df, rel_col, bin_pre, bin_post, fe, xlabel)
HEAT_SPECS = [
    ("总点赞数",  likes_lbl,    "monthly",
     tr(LANG,"Fig3a_EventStudy月度_点赞.pdf",  "Fig3a_EventStudy_Monthly_Likes.pdf"),
     tr(LANG,"Fig 3a　Event Study：点赞数对政策落实的月度效应",
              "Fig 3a  Event Study: Monthly Effect of Likes on Policy Implementation")),
    ("总转发数",  reposts_lbl,  "monthly",
     tr(LANG,"Fig3b_EventStudy月度_转发.pdf",  "Fig3b_EventStudy_Monthly_Reposts.pdf"),
     tr(LANG,"Fig 3b　Event Study：转发数对政策落实的月度效应",
              "Fig 3b  Event Study: Monthly Effect of Reposts on Policy Implementation")),
    ("总评论数",  comments_lbl, "monthly",
     tr(LANG,"Fig3c_EventStudy月度_评论.pdf",  "Fig3c_EventStudy_Monthly_Comments.pdf"),
     tr(LANG,"Fig 3c　Event Study：评论数对政策落实的月度效应",
              "Fig 3c  Event Study: Monthly Effect of Comments on Policy Implementation")),
    ("总点赞数",  likes_lbl,    "quarterly",
     tr(LANG,"Fig4a_EventStudy季度_点赞.pdf",  "Fig4a_EventStudy_Quarterly_Likes.pdf"),
     tr(LANG,"Fig 4a　Event Study：点赞数对政策落实的季度效应",
              "Fig 4a  Event Study: Quarterly Effect of Likes on Policy Implementation")),
    ("总转发数",  reposts_lbl,  "quarterly",
     tr(LANG,"Fig4b_EventStudy季度_转发.pdf",  "Fig4b_EventStudy_Quarterly_Reposts.pdf"),
     tr(LANG,"Fig 4b　Event Study：转发数对政策落实的季度效应",
              "Fig 4b  Event Study: Quarterly Effect of Reposts on Policy Implementation")),
    ("总评论数",  comments_lbl, "quarterly",
     tr(LANG,"Fig4c_EventStudy季度_评论.pdf",  "Fig4c_EventStudy_Quarterly_Comments.pdf"),
     tr(LANG,"Fig 4c　Event Study：评论数对政策落实的季度效应",
              "Fig 4c  Event Study: Quarterly Effect of Comments on Policy Implementation")),
]

heat_results = {}
for var, label, freq, fname, title_str in HEAT_SPECS:
    print(f"  {label} {freq}…")
    if freq == "monthly":
        c, f = run_event_study(
            panel_m, y_var="y_max_score", treat_var=var,
            rel_col="rel_month", base=-1, bin_pre=-12, bin_post=24,
            fe="title + calendar_ym",
        )
        xl = xlabel_m
    else:
        c, f = run_event_study(
            panel_q_agg, y_var="y_max_score", treat_var=var,
            rel_col="rel_quarter", base=-1, bin_pre=-4, bin_post=8,
            fe="title + calendar_yq",
        )
        xl = xlabel_q
    heat_results[(label, freq)] = (c, f)
    if c is not None:
        plot_event_study(c, title=title_str, xlabel=xl,
                         save_path=os.path.join(FIGURES, fname))


# ══════════════════════════════════════════════════════════════════════════════
# 态度/情感 Event Study：月度（Fig5a/c）+ 季度（Fig5b/d）
# ══════════════════════════════════════════════════════════════════════════════
print(tr(LANG, "\n运行态度/情感 Event Study…", "\nRunning attitude/sentiment Event Study..."))

ATTITUDE_SPECS = [
    ("Stance_i",    stance_lbl, "monthly",
     tr(LANG,"Fig5a_EventStudy月度_态度.pdf",    "Fig5a_EventStudy_Monthly_Stance.pdf"),
     tr(LANG,"Fig 5a　Event Study：公众支持度对政策落实的月度效应",
              "Fig 5a  Event Study: Monthly Effect of Public Stance on Policy Implementation")),
    ("Stance_i",    stance_lbl, "quarterly",
     tr(LANG,"Fig5b_EventStudy季度_态度.pdf",    "Fig5b_EventStudy_Quarterly_Stance.pdf"),
     tr(LANG,"Fig 5b　Event Study：公众支持度对政策落实的季度效应",
              "Fig 5b  Event Study: Quarterly Effect of Public Stance on Policy Implementation")),
    ("Sentiment_i", sent_lbl,   "monthly",
     tr(LANG,"Fig5c_EventStudy月度_情感.pdf",    "Fig5c_EventStudy_Monthly_Sentiment.pdf"),
     tr(LANG,"Fig 5c　Event Study：公众情感对政策落实的月度效应",
              "Fig 5c  Event Study: Monthly Effect of Public Sentiment on Policy Implementation")),
    ("Sentiment_i", sent_lbl,   "quarterly",
     tr(LANG,"Fig5d_EventStudy季度_情感.pdf",    "Fig5d_EventStudy_Quarterly_Sentiment.pdf"),
     tr(LANG,"Fig 5d　Event Study：公众情感对政策落实的季度效应",
              "Fig 5d  Event Study: Quarterly Effect of Public Sentiment on Policy Implementation")),
]

att_results = {}
for var, label, freq, fname, title_str in ATTITUDE_SPECS:
    print(f"  {label} {freq}…")
    if freq == "monthly":
        c, f = run_event_study(
            panel_m, y_var="y_max_score", treat_var=var,
            rel_col="rel_month", base=-1, bin_pre=-12, bin_post=24,
            fe="title + calendar_ym",
        )
        xl = xlabel_m
    else:
        c, f = run_event_study(
            panel_q_agg, y_var="y_max_score", treat_var=var,
            rel_col="rel_quarter", base=-1, bin_pre=-4, bin_post=8,
            fe="title + calendar_yq",
        )
        xl = xlabel_q
    att_results[(label, freq)] = (c, f)
    if c is not None:
        plot_event_study(c, title=title_str, xlabel=xl,
                         save_path=os.path.join(FIGURES, fname))


# ══════════════════════════════════════════════════════════════════════════════
# M4：热度（评论数）× 态度交互（月度 + 季度）
# ══════════════════════════════════════════════════════════════════════════════
print(tr(LANG, "\n运行 M4 热度×态度交互（月度 + 季度）…",
         "\nRunning M4 heat×stance interaction (monthly + quarterly)..."))

def build_m4(df, rel_col, fe, bin_pre, bin_post):
    df = df.copy()
    # 标准化（与 run_event_study 保持一致）
    for _col in ["总评论数", "Stance_i"]:
        _s = df[_col].std()
        if _s > 1e-8:
            df[_col] = df[_col] / _s
    df["HeatXStance"] = df["总评论数"] * df["Stance_i"]
    df_nb = df[df[rel_col] != -1].copy()

    k_vals = df_nb[rel_col].values.astype(float)
    k_vals = np.where(k_vals <= bin_pre, bin_pre, k_vals)
    k_vals = np.where(k_vals >= bin_post, bin_post, k_vals)
    df_nb["k_binned"] = k_vals
    rel_vals = sorted(df_nb["k_binned"].unique())

    def _sfx(k):
        if k == bin_pre:  return "pre"
        if k == bin_post: return "post"
        return f"p{int(k)}" if k >= 0 else f"m{int(abs(k))}"

    extra = {}
    inter_heat, inter_sta, inter_cross = [], [], []
    for k in rel_vals:
        d = (df_nb["k_binned"] == k).astype(float)
        s = _sfx(k)
        extra[f"ih_{s}"] = df_nb["总评论数"]    * d
        extra[f"is_{s}"] = df_nb["Stance_i"]    * d
        extra[f"ix_{s}"] = df_nb["HeatXStance"] * d
        inter_heat.append(f"ih_{s}")
        inter_sta.append(f"is_{s}")
        inter_cross.append(f"ix_{s}")

    df_nb = pd.concat([df_nb, pd.DataFrame(extra, index=df_nb.index)], axis=1)
    fml = ("y_max_score ~ "
           + " + ".join(inter_heat + inter_sta + inter_cross)
           + f" | {fe}")
    try:
        fit = pf.feols(fml, data=df_nb, vcov={"CRV1": "title"})
        return fit, inter_heat, inter_sta, inter_cross
    except Exception as e:
        print(f"  ⚠ M4 失败：{e}")
        return None, [], [], []

fit_m4_m, ih_m, is_m, ix_m = build_m4(
    panel_m, rel_col="rel_month", fe="title + calendar_ym",
    bin_pre=-12, bin_post=24,
)
print(f"  {tr(LANG,'M4 月度','M4 monthly')}：{'✓' if fit_m4_m else '✗'}")

fit_m4_q, ih_q, is_q, ix_q = build_m4(
    panel_q_agg, rel_col="rel_quarter", fe="title + calendar_yq",
    bin_pre=-4, bin_post=8,
)
print(f"  {tr(LANG,'M4 季度','M4 quarterly')}：{'✓' if fit_m4_q else '✗'}")


# ── M4 动态系数图 ─────────────────────────────────────────────────────────────
def extract_m4_group_coefs(fit, prefix, rel_col_name, bin_pre, bin_post, base=-1):
    """从 M4 回归结果中提取某组交互项的逐期系数，构造 plot_event_study 所需 DataFrame。"""
    if fit is None:
        return None
    coef_df = fit.coef().reset_index(); coef_df.columns = ["term", "coef"]
    se_df   = fit.se().reset_index();   se_df.columns   = ["term", "se"]
    merged  = coef_df.merge(se_df, on="term")
    sub = merged[merged["term"].str.startswith(prefix)].copy()
    if len(sub) == 0:
        return None
    def _parse(t):
        s = t[len(prefix):]
        if s == "pre":  return float(bin_pre)
        if s == "post": return float(bin_post)
        return float(s[1:]) if s.startswith("p") else -float(s[1:])
    sub["rel_month"] = sub["term"].apply(_parse)
    sub["ci_low"]  = sub["coef"] - 1.96 * sub["se"]
    sub["ci_high"] = sub["coef"] + 1.96 * sub["se"]
    sub["is_bin"]  = sub["term"].apply(
        lambda t: t.endswith("_pre") or t.endswith("_post"))
    base_row = pd.DataFrame([{
        "term": f"{prefix}base", "coef": 0.0, "se": 0.0,
        "rel_month": float(base), "ci_low": 0.0, "ci_high": 0.0, "is_bin": False,
    }])
    sub = pd.concat([sub, base_row]).sort_values("rel_month").reset_index(drop=True)
    return sub

print(tr(LANG, "\n生成 M4 动态系数图…", "\nGenerating M4 dynamic coefficient plots..."))

m4_group_info = [
    ("ih_", tr(LANG, "热度（评论数）", "Heat (Comments)"),  "heat"),
    ("is_", tr(LANG, "态度（Stance）",  "Stance"),          "stance"),
    ("ix_", tr(LANG, "热度×态度",       "Heat×Stance"),     "cross"),
]

# M4 月度
for prefix, glbl, sfx in m4_group_info:
    c_m4 = extract_m4_group_coefs(fit_m4_m, prefix, "rel_month", -12, 24)
    if c_m4 is not None:
        fname_m4 = tr(LANG,
            f"FigM4_月度_{sfx}.pdf",
            f"FigM4_Monthly_{sfx}.pdf")
        plot_event_study(
            c_m4,
            title=tr(LANG,
                f"M4 月度动态系数：{glbl}",
                f"M4 Monthly Dynamic Coefficients: {glbl}"),
            xlabel=xlabel_m,
            save_path=os.path.join(FIGURES, fname_m4),
        )

# M4 季度
for prefix, glbl, sfx in m4_group_info:
    c_m4 = extract_m4_group_coefs(fit_m4_q, prefix, "rel_quarter", -4, 8)
    if c_m4 is not None:
        fname_m4 = tr(LANG,
            f"FigM4_季度_{sfx}.pdf",
            f"FigM4_Quarterly_{sfx}.pdf")
        plot_event_study(
            c_m4,
            title=tr(LANG,
                f"M4 季度动态系数：{glbl}",
                f"M4 Quarterly Dynamic Coefficients: {glbl}"),
            xlabel=xlabel_q,
            save_path=os.path.join(FIGURES, fname_m4),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab4：全部 Event Study 的平行趋势 Wald F 检验 + 系数摘要（10 列）
# ══════════════════════════════════════════════════════════════════════════════
print(tr(LANG, "\n生成 Tab4（Wald F 检验汇总）…",
         "\nGenerating Tab4 (Wald F-test summary)..."))

# 全部 10 个规格：(col_header, label, freq, freq_short, results_dict)
all_es_specs = [
    (f"(1)",  likes_lbl,    "monthly",   tr(LANG,"月","M"), heat_results),
    (f"(2)",  reposts_lbl,  "monthly",   tr(LANG,"月","M"), heat_results),
    (f"(3)",  comments_lbl, "monthly",   tr(LANG,"月","M"), heat_results),
    (f"(4)",  stance_lbl,   "monthly",   tr(LANG,"月","M"), att_results),
    (f"(5)",  sent_lbl,     "monthly",   tr(LANG,"月","M"), att_results),
    (f"(6)",  likes_lbl,    "quarterly", tr(LANG,"季","Q"), heat_results),
    (f"(7)",  reposts_lbl,  "quarterly", tr(LANG,"季","Q"), heat_results),
    (f"(8)",  comments_lbl, "quarterly", tr(LANG,"季","Q"), heat_results),
    (f"(9)",  stance_lbl,   "quarterly", tr(LANG,"季","Q"), att_results),
    (f"(10)", sent_lbl,     "quarterly", tr(LANG,"季","Q"), att_results),
]

# 收集所有规格的 F 检验结果
wald_rows = []
for col_hdr, label, freq, freq_short, res_dict in all_es_specs:
    coefs_df, fit = res_dict.get((label, freq), (None, None))
    # Wald F 检验
    ftest = pretrend_test(fit, coefs_df, rel_col="rel_month")
    # 事前/事后平均系数
    pre_c, pre_se   = avg_beta_se(coefs_df, fit, post=False)
    post_c, post_se  = avg_beta_se(coefs_df, fit, post=True)
    n_v, r2_v = get_n_r2(fit)
    wald_rows.append({
        "col": col_hdr, "label": label, "freq": freq_short,
        "F": ftest["F_stat"] if ftest else None,
        "p": ftest["p_value"] if ftest else None,
        "df": ftest["df"] if ftest else None,
        "pre_c": pre_c, "pre_se": pre_se,
        "post_c": post_c, "post_se": post_se,
        "N": n_v, "R2": r2_v,
    })

# 打印到终端
print(tr(LANG, "\n全部 Event Study 前趋势 Wald F 检验：",
         "\nAll Event Study pre-trend Wald F-tests:"))
for r in wald_rows:
    f_str = f"F={r['F']:.3f}, p={r['p']:.3f} (df={r['df']})" if r['F'] is not None else "N/A"
    print(f"  {r['col']} {r['label']}×{r['freq']}: {f_str}")

# 生成 LaTeX 表格（纵向排列，每行一个规格）
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
tab4_cap = tr(LANG, "Event Study 平行趋势检验与系数摘要",
                    "Event Study Parallel Trends Tests and Coefficient Summary")
lines.append(f"\\caption{{{tab4_cap}}}")
lines.append(r"\label{tab:eventstudy_heat}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llccccc}")
lines.append(r"\toprule")
header = tr(LANG,
    r" & & 事前 Wald $F$ & $p$值 & 事前均值 $\bar{\beta}_{k<0}$ & 事后均值 $\bar{\beta}_{k>0}$ & $N$ \\",
    r" & & Pre Wald $F$ & $p$-value & Pre-avg $\bar{\beta}_{k<0}$ & Post-avg $\bar{\beta}_{k>0}$ & $N$ \\")
lines.append(header)
lines.append(r"\midrule")

# 热度变量组
lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{tr(LANG,'热度变量','Heat variables')}}}}} \\\\[2pt]")
for r in wald_rows:
    if r["label"] not in [likes_lbl, reposts_lbl, comments_lbl]:
        continue
    f_str = f"{r['F']:.3f}" if r['F'] is not None else "—"
    p_str = f"{r['p']:.3f}" if r['p'] is not None else "—"
    # 系数行
    lines.append(
        f"\\quad {latex_var_label(r['label'])} & {r['freq']}"
        f" & {f_str} & {p_str} & {r['pre_c']} & {r['post_c']} & {r['N']} \\\\")
    # SE 行
    lines.append(
        f" & & & & {r['pre_se']} & {r['post_se']} & \\\\[3pt]")

# 态度变量组
lines.append(f"[2pt]\\multicolumn{{7}}{{l}}{{\\textit{{{tr(LANG,'态度变量','Attitude variables')}}}}} \\\\[2pt]")
for r in wald_rows:
    if r["label"] not in [stance_lbl, sent_lbl]:
        continue
    f_str = f"{r['F']:.3f}" if r['F'] is not None else "—"
    p_str = f"{r['p']:.3f}" if r['p'] is not None else "—"
    lines.append(
        f"\\quad {latex_var_label(r['label'])} & {r['freq']}"
        f" & {f_str} & {p_str} & {r['pre_c']} & {r['post_c']} & {r['N']} \\\\")
    lines.append(
        f" & & & & {r['pre_se']} & {r['post_se']} & \\\\[3pt]")

lines.append(r"\bottomrule")
tab4_note = tr(
    LANG,
    "注：Wald $F$ 统计量检验所有事前系数（$k < 0$，不含 bin 端点）联合为零的原假设，"
    "$p$ 值基于 $F(K, N-p)$ 分布。"
    "事前均值和事后均值为对应时期交互项系数的简单平均，"
    "括号内为基于方差-协方差矩阵计算的平均系数标准误，***、**、*分别表示在1\\%、5\\%、10\\%水平上显著。"
    "基准期 $k=-1$（系数归零）。"
    "月度窗口 $[-12,+24]$，季度窗口 $[-4,+8]$，两端施加 binning。"
    "固定效应：话题 + 日历时间。标准误在话题层面聚类。",
    "Note: Wald $F$ tests the joint null that all pre-period coefficients ($k<0$, excl.\\ bin endpoints) "
    "equal zero; $p$-value from $F(K, N-p)$ distribution. "
    "Pre/post averages are simple means of interaction coefficients; "
    "SE in parentheses computed from the variance-covariance matrix. "
    "*** $p<0.01$, ** $p<0.05$, * $p<0.10$. "
    "Baseline $k=-1$. Monthly $[-12,+24]$, quarterly $[-4,+8]$, tails binned. "
    "FE: topic + calendar time. SE clustered at topic level.",
)
lines.append(f"\\multicolumn{{7}}{{p{{0.95\\textwidth}}}}{{\\footnotesize {tab4_note}}}")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tab4_fname = tr(LANG, "Tab4_热度EventStudy.tex", "Tab4_HeatEventStudy.tex")
with open(os.path.join(TABLES, tab4_fname), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"  ✓ {tab4_fname}")

# Tab5 不再单独输出（已合并进 Tab4）
tab5_fname = tr(LANG, "Tab5_态度EventStudy.tex", "Tab5_AttitudeEventStudy.tex")
# 写一个空文件防止 LaTeX \input 报错（如果论文中仍引用）
with open(os.path.join(TABLES, tab5_fname), "w", encoding="utf-8") as fh:
    fh.write("% 已合并至 Tab4\n")


# ══════════════════════════════════════════════════════════════════════════════
# Tab5b：M4 热度×态度交互（月度 + 季度）
# ══════════════════════════════════════════════════════════════════════════════
print(tr(LANG, "生成 Tab5b…", "Generating Tab5b..."))

def avg_group_post(fit, prefix):
    """平均 k>0 且非 bin 端点的交互项系数。"""
    if fit is None:
        return "—"
    try:
        coef_df = fit.coef().reset_index(); coef_df.columns = ["term", "coef"]
        sub = coef_df[coef_df["term"].str.startswith(prefix)].copy()
        sub = sub[~sub["term"].str.endswith("_pre") & ~sub["term"].str.endswith("_post")]
        def _k(t):
            s = t[len(prefix):]
            return float(s[1:]) if s.startswith("p") else -float(s[1:])
        sub["k"] = sub["term"].apply(_k)
        post = sub[sub["k"] > 0]
        return f"{post['coef'].mean():.4f}" if len(post) else "—"
    except Exception:
        return "—"


def avg_group_post_se(fit, prefix):
    """Return (coef_str_with_stars, se_str_in_parens) for averaged post-period group coefficients."""
    if fit is None:
        return "—", ""
    try:
        coef_df = fit.coef().reset_index(); coef_df.columns = ["term", "coef"]
        sub = coef_df[coef_df["term"].str.startswith(prefix)].copy()
        sub = sub[~sub["term"].str.endswith("_pre") & ~sub["term"].str.endswith("_post")]
        def _k(t):
            s = t[len(prefix):]
            return float(s[1:]) if s.startswith("p") else -float(s[1:])
        sub["k"] = sub["term"].apply(_k)
        post = sub[sub["k"] > 0]
        if len(post) == 0:
            return "—", ""
        avg = post["coef"].mean()
        terms = post["term"].tolist()
        try:
            V = fit.vcov()
            V_sub = V.loc[terms, terms].values
            n = len(terms)
            ones = np.ones(n)
            var_avg = float(ones @ V_sub @ ones) / (n ** 2)
            se_avg = np.sqrt(max(var_avg, 0))
        except Exception:
            se_df = fit.se().reset_index(); se_df.columns = ["term", "se"]
            se_sub = se_df[se_df["term"].isin(terms)]
            se_avg = se_sub["se"].mean() / np.sqrt(len(se_sub))
        if se_avg > 0:
            t_stat = abs(avg / se_avg)
            pval = 2 * (1 - sp_stats.norm.cdf(t_stat))
            st = _stars_from_pval(pval)
        else:
            st = ""
        return f"{avg:.4f}{st}", f"({se_avg:.4f})"
    except Exception:
        return "—", ""

n_m4m, r2_m4m = get_n_r2(fit_m4_m)
n_m4q, r2_m4q = get_n_r2(fit_m4_q)


def m4_group_pretrend_F(fit, prefix, bin_pre):
    """对 M4 中某一组交互项（如 ih_）的事前系数做 Wald F 联合检验。"""
    if fit is None:
        return None
    try:
        coef_s = fit.coef(); se_s = fit.se()
        terms = [t for t in coef_s.index if t.startswith(prefix)]
        # 解析 k 值
        def _k(t):
            s = t[len(prefix):]
            if s == "pre":  return float(bin_pre)
            if s == "post": return np.inf
            return float(s[1:]) if s.startswith("p") else -float(s[1:])
        pre_terms = [t for t in terms if _k(t) < 0
                     and not t.endswith("_pre") and not t.endswith("_post")]
        if len(pre_terms) == 0:
            return None
        beta = np.array([coef_s[t] for t in pre_terms])
        se   = np.array([se_s[t] if se_s[t] > 0 else 1e-10 for t in pre_terms])
        wald = float(np.sum((beta / se) ** 2) / len(beta))
        from scipy.stats import f as f_dist
        try:
            nobs = int(fit._N); npar = int(fit._k)
        except Exception:
            nobs, npar = 1000, len(beta) + 2
        pval = float(1 - f_dist.cdf(wald, len(beta), max(1, nobs - npar)))
        return {"F": wald, "p": pval, "df": len(beta)}
    except Exception:
        return None


def avg_group_pre_se(fit, prefix, bin_pre):
    """事前均值 + SE + 星号，与 avg_group_post_se 对称。"""
    if fit is None:
        return "—", ""
    try:
        coef_df = fit.coef().reset_index(); coef_df.columns = ["term", "coef"]
        sub = coef_df[coef_df["term"].str.startswith(prefix)].copy()
        sub = sub[~sub["term"].str.endswith("_pre") & ~sub["term"].str.endswith("_post")]
        def _k(t):
            s = t[len(prefix):]
            return float(s[1:]) if s.startswith("p") else -float(s[1:])
        sub["k"] = sub["term"].apply(_k)
        pre = sub[sub["k"] < 0]
        if len(pre) == 0:
            return "—", ""
        avg = pre["coef"].mean()
        terms = pre["term"].tolist()
        try:
            V = fit.vcov()
            V_sub = V.loc[terms, terms].values
            n = len(terms)
            ones = np.ones(n)
            var_avg = float(ones @ V_sub @ ones) / (n ** 2)
            se_avg = np.sqrt(max(var_avg, 0))
        except Exception:
            se_df = fit.se().reset_index(); se_df.columns = ["term", "se"]
            se_sub = se_df[se_df["term"].isin(terms)]
            se_avg = se_sub["se"].mean() / np.sqrt(len(se_sub))
        if se_avg > 0:
            t_stat = abs(avg / se_avg)
            pval = 2 * (1 - sp_stats.norm.cdf(t_stat))
            st = _stars_from_pval(pval)
        else:
            st = ""
        return f"{avg:.4f}{st}", f"({se_avg:.4f})"
    except Exception:
        return "—", ""


tab5b_note = tr(
    LANG,
    "注：M4 同时纳入热度（评论数）、态度、热度$\\times$态度三组交互项。"
    "事前 Wald $F$ 检验各组事前系数（$k<0$，不含 bin 端点）联合为零的原假设，$p$ 值基于 $F(K,N-p)$ 分布。"
    "事前均值和事后均值为对应时期交互项系数的简单平均，"
    "括号内为标准误，***、**、*分别表示在1\\%、5\\%、10\\%水平上显著。"
    "月度：话题 FE + 日历月 FE；季度：话题 FE + 日历季度 FE。标准误在话题层面聚类。",
    "Note: M4 includes heat (comments), stance, and heat×stance interaction groups. "
    "Pre Wald $F$ tests the joint null that all pre-period coefficients ($k<0$, excl. bin) equal zero. "
    "Pre/post averages are simple means; SE in parentheses; *** $p<0.01$, ** $p<0.05$, * $p<0.10$. "
    "Monthly: topic FE + calendar-month FE; Quarterly: topic FE + calendar-quarter FE. "
    "SE clustered at topic level.",
)

row_h = tr(LANG, "热度（评论数）",   "Heat (comments)")
row_s = tr(LANG, "态度（Stance）",   "Stance")
row_x = tr(LANG, "热度$\\times$态度", "Heat$\\times$Stance")
col_m = f"(1) {tr(LANG,'月度','Monthly')}"
col_q = f"(2) {tr(LANG,'季度','Quarterly')}"

# 收集数据
m4_groups = [
    (row_h, 'ih_'),
    (row_s, 'is_'),
    (row_x, 'ix_'),
]

tab5b_lines = []
tab5b_lines.append(r"\begin{table}[htbp]")
tab5b_lines.append(r"\centering")
tab5b_lines.append(f"\\caption{{{tr(LANG,'热度×态度交互 Event Study（M4）平行趋势检验与系数摘要','Heat×Stance Interaction Event Study (M4) with Parallel Trends Tests')}}}")
tab5b_lines.append(r"\label{tab:eventstudy_interaction}")
tab5b_lines.append(r"\small")
tab5b_lines.append(r"\begin{tabular}{lcc}")
tab5b_lines.append(r"\toprule")
tab5b_lines.append(f" & {col_m} & {col_q} \\\\")
tab5b_lines.append(r"\midrule")

for row_label, prefix in m4_groups:
    # Wald F
    ft_m = m4_group_pretrend_F(fit_m4_m, prefix, bin_pre=-12)
    ft_q = m4_group_pretrend_F(fit_m4_q, prefix, bin_pre=-4)
    f_m_str = f"{ft_m['F']:.3f}" if ft_m else "—"
    p_m_str = f"[{ft_m['p']:.3f}]" if ft_m else ""
    f_q_str = f"{ft_q['F']:.3f}" if ft_q else "—"
    p_q_str = f"[{ft_q['p']:.3f}]" if ft_q else ""

    # 事前均值
    pre_cm, pre_sem = avg_group_pre_se(fit_m4_m, prefix, bin_pre=-12)
    pre_cq, pre_seq = avg_group_pre_se(fit_m4_q, prefix, bin_pre=-4)

    # 事后均值
    post_cm, post_sem = avg_group_post_se(fit_m4_m, prefix)
    post_cq, post_seq = avg_group_post_se(fit_m4_q, prefix)

    tab5b_lines.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{latex_var_label(row_label)}}}}} \\\\")
    tab5b_lines.append(f"\\quad {tr(LANG,'事前 Wald $F$','Pre Wald $F$')} & {f_m_str} & {f_q_str} \\\\")
    tab5b_lines.append(f"\\quad {tr(LANG,'$p$值','$p$-value')} & {p_m_str} & {p_q_str} \\\\")
    tab5b_lines.append(f"\\quad {tr(LANG,'事前均值','Pre avg')} & {pre_cm} & {pre_cq} \\\\")
    tab5b_lines.append(f" & {pre_sem} & {pre_seq} \\\\")
    tab5b_lines.append(f"\\quad {tr(LANG,'事后均值','Post avg')} & {post_cm} & {post_cq} \\\\")
    tab5b_lines.append(f" & {post_sem} & {post_seq} \\\\[4pt]")

tab5b_lines.append(r"\midrule")
tab5b_lines.append(f"{tr(LANG,'观测值','Observations')} & {n_m4m} & {n_m4q} \\\\")
tab5b_lines.append(f"$R^2$ & {r2_m4m} & {r2_m4q} \\\\")
tab5b_lines.append(r"\bottomrule")
tab5b_lines.append(f"\\multicolumn{{3}}{{p{{0.78\\textwidth}}}}{{\\footnotesize {tab5b_note}}}")
tab5b_lines.append(r"\end{tabular}")
tab5b_lines.append(r"\end{table}")

tab5b_fname = tr(LANG, "Tab5b_热度态度交互.tex", "Tab5b_HeatStanceInteraction.tex")
with open(os.path.join(TABLES, tab5b_fname), "w", encoding="utf-8") as f:
    f.write("\n".join(tab5b_lines))
print(f"  ✓ {tab5b_fname}")

# ══════════════════════════════════════════════════════════════════════════════
# 多规范比较：Spec 1-5 + Imputation Estimator（以评论数×月度为示例）
# ══════════════════════════════════════════════════════════════════════════════
print(tr(LANG, "\n运行多规范比较（评论数×月度）…",
         "\nRunning multi-spec comparison (Comments×Monthly)..."))

SPEC_FE_CONFIGS_M = {
    tr(LANG, "Spec1 基准",       "Spec1 Baseline"):     "title + calendar_ym",
    tr(LANG, "Spec2 政策年FE",   "Spec2 PolicyYearFE"): "title + policy_year",
    tr(LANG, "Spec3 季度时间",   "Spec3 QtrTimeFE"):    "title + calendar_yq",
    tr(LANG, "Spec4 领域FE",     "Spec4 CatL1FE"):      "category_L1 + calendar_ym",
}

multi_spec_results = {}
for spec_name, fe_str in SPEC_FE_CONFIGS_M.items():
    print(f"  {spec_name}…")
    c, f = run_event_study(
        panel_m, y_var="y_max_score", treat_var="总评论数",
        rel_col="rel_month", base=-1, bin_pre=-12, bin_post=24,
        fe=fe_str,
    )
    multi_spec_results[spec_name] = (c, f)
    # 为每个 Spec 单独输出动态系数图
    if c is not None:
        spec_fname = tr(LANG,
            f"Fig_Spec_{spec_name.split()[0]}_评论数月度.pdf",
            f"Fig_Spec_{spec_name.split()[0]}_Comments_Monthly.pdf")
        plot_event_study(
            c,
            title=tr(LANG,
                f"{spec_name}：评论数×月度 Event Study",
                f"{spec_name}: Comments × Monthly Event Study"),
            xlabel=xlabel_m,
            save_path=os.path.join(FIGURES, spec_fname),
        )

# Spec 3：Imputation Estimator
spec5_lbl = tr(LANG, "Spec5 插补", "Spec5 Imputation")
print(f"  {spec5_lbl}（bootstrap×200）…")
c_imp, f_imp = run_imputation_estimator(
    panel_m, y_var="y_max_score", treat_var="总评论数",
    rel_col="rel_month", base=-1, bin_pre=-12, bin_post=24,
    fe="title + calendar_ym", n_boot=200,
)
multi_spec_results[spec5_lbl] = (c_imp, f_imp)
if c_imp is not None:
    spec5_fname = tr(LANG,
        "Fig_Spec_Spec5_评论数月度.pdf",
        "Fig_Spec_Spec5_Comments_Monthly.pdf")
    plot_event_study(
        c_imp,
        title=tr(LANG,
            "Spec5 插补估计量：评论数×月度 Event Study",
            "Spec5 Imputation: Comments × Monthly Event Study"),
        xlabel=xlabel_m,
        save_path=os.path.join(FIGURES, spec5_fname),
    )

# 前趋势 F 检验
print(tr(LANG, "\n前趋势 F 检验：", "\nPre-trend F-tests:"))
for spec_name, (c, f) in multi_spec_results.items():
    res = pretrend_test(f, c)
    if res:
        print(f"  {spec_name}: F={res['F_stat']:.3f}, p={res['p_value']:.3f} (df={res['df']})")

# 多规范比较图
plot_multi_spec_comparison(
    multi_spec_results,
    save_path=os.path.join(FIGURES, tr(
        LANG, "Fig_多规范比较_评论数月度.pdf",
        "Fig_MultiSpec_Comments_Monthly.pdf")),
    title_str=tr(
        LANG, "多规范比较：评论数×月度 Event Study",
        "Multi-Specification Comparison: Comments × Monthly Event Study"),
)

# 多规范比较表
make_spec_comparison_table(
    multi_spec_results,
    caption=tr(LANG, "多规范比较：Event Study 系数稳定性（评论数×月度）",
               "Multi-Specification Robustness: Event Study Coefficient Stability (Comments×Monthly)"),
    label="tab:multispec_comparison",
    fname=tr(LANG, "Tab_多规范比较.tex", "Tab_MultiSpecComparison.tex"),
    rel_col="rel_month",
)

print(f"\n{tr(LANG,'完成！','Done!')}")

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
