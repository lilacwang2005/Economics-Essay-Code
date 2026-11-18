"""
模型四：Panel VAR + 脉冲响应函数（IRF）
  系统：[stance_it, sentiment_it]
  估计 VAR(L) 含话题FE，AIC/BIC 选阶
  IRF 用 bootstrap（999次）计算置信区间

输出：
  results/figures/FigA5_脉冲响应函数.pdf
  results/tables/TabA6_PanelVAR系数.tex
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, warnings
from scipy import stats
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from table_utils import setup_plot_style, style_ax, ECON_COLORS, ECON_PALETTE
from lang_utils import get_lang, tr, get_results_root, latex_var_label

ROOT = "/Users/wangmanlin/Desktop/citizens-will-and-politics"
CLEAN  = os.path.join(ROOT, "data/clean")
LANG = get_lang()
RESULTS_ROOT = get_results_root(ROOT, LANG)
TABLES = os.path.join(RESULTS_ROOT, "tables")
FIG    = os.path.join(RESULTS_ROOT, "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

setup_plot_style()

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("加载数据...")
panel = pd.read_csv(os.path.join(CLEAN, "panel_topic_window.csv"), encoding="utf-8-sig")
panel = panel[panel["窗口"] >= 0].copy()
panel_m = panel.dropna(subset=["stance_it", "sentiment_it"]).copy()

# 数值话题ID（用于矩阵运算）
panel_m["topic_id"] = pd.Categorical(panel_m["title"]).codes
panel_m = panel_m.sort_values(["topic_id", "窗口"]).reset_index(drop=True)

N_TOPICS = panel_m["topic_id"].nunique()
print(f"  话题数：{N_TOPICS:,}，总观测：{len(panel_m):,}")

# ── Within 变换（去除话题固定效应）────────────────────────────────────────────
def within_transform(df, entity_col, vars_):
    """减去话题均值（FE demean）"""
    df = df.copy()
    means = df.groupby(entity_col)[vars_].transform("mean")
    for v in vars_:
        df[f"{v}_dm"] = df[v] - means[v]
    return df

VARS = ["stance_it", "sentiment_it"]
panel_dm = within_transform(panel_m, "topic_id", VARS)

# ── 构建 VAR 滞后矩阵 ─────────────────────────────────────────────────────────
def build_var_data(df, L, vars_):
    """构建含 L 阶滞后的 VAR 设计矩阵（去均值后）"""
    dm_vars = [f"{v}_dm" for v in vars_]
    rows_y, rows_X = [], []
    for topic_id, grp in df.groupby("topic_id"):
        grp = grp.sort_values("窗口").reset_index(drop=True)
        T = len(grp)
        if T <= L:
            continue
        Y = grp[dm_vars].values[L:]       # shape (T-L, k)
        X_lags = []
        for lag in range(1, L+1):
            X_lags.append(grp[dm_vars].values[L-lag:T-lag])  # shape (T-L, k)
        X = np.hstack(X_lags)  # shape (T-L, k*L)
        rows_y.append(Y)
        rows_X.append(X)
    Y_all = np.vstack(rows_y)   # (N_obs, k)
    X_all = np.vstack(rows_X)   # (N_obs, k*L)
    return Y_all, X_all

# ── AIC/BIC 选阶 ──────────────────────────────────────────────────────────────
print("\nAIC/BIC 选阶（L = 1, 2, 3）...")
k = len(VARS)
aic_bic = []
for L in [1, 2, 3]:
    Y, X = build_var_data(panel_dm, L, VARS)
    n_obs = Y.shape[0]
    # OLS equation-by-equation
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]  # (k*L, k)
    resid = Y - X @ coef                           # (n_obs, k)
    sigma = resid.T @ resid / n_obs
    log_det = np.log(np.linalg.det(sigma))
    n_params = k**2 * L
    aic = log_det + 2 * n_params / n_obs
    bic = log_det + np.log(n_obs) * n_params / n_obs
    aic_bic.append((L, aic, bic, n_obs))
    print(f"  L={L}: AIC={aic:.4f}, BIC={bic:.4f}, N_obs={n_obs:,}")

best_L = min(aic_bic, key=lambda x: x[2])[0]  # BIC
print(f"\n最优阶数（BIC）：L = {best_L}")

# ── 估计 VAR(best_L) ──────────────────────────────────────────────────────────
print(f"\n估计 VAR({best_L})...")
Y, X = build_var_data(panel_dm, best_L, VARS)
n_obs = Y.shape[0]

coef_mat = np.linalg.lstsq(X, Y, rcond=None)[0]  # (k*L, k)
resid    = Y - X @ coef_mat
sigma    = resid.T @ resid / (n_obs - k * best_L)

# Compute standard errors for VAR coefficients
XtX_inv = np.linalg.inv(X.T @ X)
# SE for each equation j: sqrt(diag(XtX_inv) * sigma_jj)
se_mat = np.zeros_like(coef_mat)
for j in range(k):
    se_mat[:, j] = np.sqrt(np.diag(XtX_inv) * sigma[j, j])

# 系数表：列为方程（因变量），行为滞后变量
var_names = [tr(LANG, "态度", "Stance"), tr(LANG, "情感", "Sentiment")]
if LANG == "zh":
    lag_names = [f"{v}滞后{l}期" for l in range(1, best_L + 1) for v in var_names]
else:
    lag_names = [f"{v}_lag{l}" for l in range(1, best_L + 1) for v in var_names]

print("\n VAR 系数矩阵（行=解释变量，列=因变量）：")
header = f"{'':30s}" + "  ".join(f"{v:15s}" for v in var_names)
print(header)
for i, rname in enumerate(lag_names):
    row = f"{rname:30s}"
    for j in range(k):
        row += f"  {coef_mat[i, j]:8.4f}      "
    print(row)

# ── Granger 因果检验 ──────────────────────────────────────────────────────────
print("\n格兰杰因果检验（Wald test）：")
def granger_test(coef_mat, X, Y, resid, k, L, cause_idx, effect_idx):
    """检验 cause_idx 是否 Granger-cause effect_idx"""
    # 受约束方程：去掉 cause_idx 的所有滞后
    restrict_cols = [l*k + cause_idx for l in range(L)]
    unrestrict_cols = list(range(k*L))
    X_r = X[:, [c for c in unrestrict_cols if c not in restrict_cols]]
    coef_r = np.linalg.lstsq(X_r, Y[:, effect_idx], rcond=None)[0]
    resid_r = Y[:, effect_idx] - X_r @ coef_r
    resid_u = resid[:, effect_idx]
    n_obs = len(Y)
    n_restrict = len(restrict_cols)
    F = ((resid_r@resid_r - resid_u@resid_u) / n_restrict) / \
        ((resid_u@resid_u) / (n_obs - k*L))
    pval = 1 - stats.f.cdf(F, n_restrict, n_obs - k*L)
    return F, pval

tests = [
    (tr(LANG, "情感 → 态度（格兰杰因果）", "Sentiment -> Stance (Granger causality)"), 1, 0),
    (tr(LANG, "态度 → 情感（格兰杰因果）", "Stance -> Sentiment (Granger causality)"), 0, 1),
]
granger_results = {}
for name, cause, effect in tests:
    F, pval = granger_test(coef_mat, X, Y, resid, k, best_L, cause, effect)
    stars = "***" if pval<0.01 else ("**" if pval<0.05 else ("*" if pval<0.1 else ""))
    print(f"  {name}: F={F:.3f}{stars}, p={pval:.4f}")
    granger_results[name] = (F, pval)

# ── IRF 计算 ─────────────────────────────────────────────────────────────────
print("\n计算脉冲响应函数（IRF）...")
H = 20  # 脉冲后追踪 H 个窗口（= H*2h）

def compute_irf(coef_mat, k, L, H, shock_size=None):
    """计算 Cholesky 分解 IRF"""
    # 构建 VAR 伴随矩阵（companion form）
    A = coef_mat.T  # (k, k*L)
    # 将 VAR(L) 转为 VAR(1)
    F_companion = np.zeros((k*L, k*L))
    F_companion[:k, :] = A
    for l in range(1, L):
        F_companion[l*k:(l+1)*k, (l-1)*k:l*k] = np.eye(k)

    # Sigma（协方差矩阵）
    sigma = (resid.T @ resid) / (len(resid) - k*L)
    # Cholesky 分解（排序：stance first）
    P = np.linalg.cholesky(sigma)  # lower triangular

    irf_all = np.zeros((H+1, k, k))
    Jmat = np.zeros((k, k*L))
    Jmat[:, :k] = np.eye(k)

    Phi = np.eye(k*L)
    for h in range(H+1):
        irf_all[h] = Jmat @ Phi @ Jmat.T @ P
        Phi = Phi @ F_companion

    return irf_all  # shape (H+1, k, k)

irf = compute_irf(coef_mat, k, best_L, H)

# Bootstrap IRF（999次）
print("  Bootstrap IRF（999次）...")
N_BOOT = 999
irf_boot = np.zeros((N_BOOT, H+1, k, k))
rng = np.random.default_rng(42)

for b in range(N_BOOT):
    # Resample residuals
    idx = rng.integers(0, len(resid), size=len(resid))
    resid_b = resid[idx]
    Y_b     = X @ coef_mat + resid_b
    coef_b  = np.linalg.lstsq(X, Y_b, rcond=None)[0]
    # recompute resid for sigma
    resid_b2 = Y_b - X @ coef_b
    orig_resid = resid
    resid = resid_b2
    irf_b = compute_irf(coef_b, k, best_L, H)
    resid = orig_resid
    irf_boot[b] = irf_b

irf_lo = np.percentile(irf_boot, 2.5,  axis=0)
irf_hi = np.percentile(irf_boot, 97.5, axis=0)
print("  Bootstrap 完成。")

# ── Fig A5：IRF 图（2×2 subplots）────────────────────────────────────────────
print("\n生成 Fig A5...")

shock_names    = [tr(LANG, "态度冲击", "Stance shock"), tr(LANG, "情感冲击", "Sentiment shock")]
response_names = [tr(LANG, "对态度响应", "Stance response"), tr(LANG, "对情感响应", "Sentiment response")]
x_axis = np.arange(H+1) * 2  # 转为小时

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

colors_irf = [ECON_COLORS["blue"], ECON_COLORS["red"]]

for shock_i in range(k):
    c = colors_irf[shock_i]
    for resp_j in range(k):
        ax = axes[resp_j][shock_i]
        y_irf = irf[:, resp_j, shock_i]
        y_lo  = irf_lo[:, resp_j, shock_i]
        y_hi  = irf_hi[:, resp_j, shock_i]

        ax.fill_between(x_axis, y_lo, y_hi, alpha=0.18, color=c, label=tr(LANG, "95%置信区间", "95% CI"))
        ax.plot(x_axis, y_irf, color=c, linewidth=2, label=tr(LANG, "点估计", "Point estimate"))
        ax.axhline(0, color="#444444", linewidth=0.9, linestyle="--")
        ax.set_xlabel(tr(LANG, "冲击后小时数", "Hours after shock"), fontsize=11)
        ax.set_ylabel(tr(LANG, "响应幅度", "Response magnitude"), fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_xlim(0, H*2)
        ax.legend(fontsize=9, loc="upper right")
        style_ax(ax)

plt.tight_layout()
fig_a5_name = tr(LANG, "FigA5_脉冲响应函数.pdf", "FigA5_ImpulseResponseFunctions.pdf")
out_a5 = os.path.join(FIG, fig_a5_name)
plt.savefig(out_a5, dpi=300, bbox_inches="tight")
plt.close()
print(f"  → 已保存：{out_a5}")

# ── 输出 LaTeX 表格 Tab A6 ────────────────────────────────────────────────────
print("\n生成 LaTeX 表格 Tab A6...")
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(f"\\caption{{{tr(LANG, 'Panel VAR 系数及格兰杰因果检验', 'Panel VAR Coefficients and Granger Causality Tests')}}}")
lines.append(r"\label{tab:a6_panel_var}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lcc}")
lines.append(r"\toprule")
lines.append(tr(LANG, r"解释变量 & 因变量：态度 & 因变量：情感 \\", r"Regressor & Dep. var.: stance & Dep. var.: sentiment \\"))
lines.append(r"\midrule")

for i, rname in enumerate(lag_names):
    # Compute t-stat and stars for each equation
    cells = []
    for j in range(k):
        c = coef_mat[i, j]
        se = se_mat[i, j]
        if se > 0:
            t = abs(c / se)
            pval = 2 * (1 - stats.t.cdf(t, df=n_obs - k * best_L))
            st = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        else:
            st = ""
        cells.append((f"{c:.4f}{st}", f"({se:.4f})"))
    lines.append(f"{latex_var_label(rname)} & {cells[0][0]} & {cells[1][0]} \\\\")
    lines.append(f" & {cells[0][1]} & {cells[1][1]} \\\\[3pt]")

lines.append(r"\midrule")
lines.append(tr(LANG, f"VAR阶数 & \\multicolumn{{2}}{{c}}{{L = {best_L}（BIC选择）}} \\\\", f"VAR lag order & \\multicolumn{{2}}{{c}}{{L = {best_L} (selected by BIC)}} \\\\"))
lines.append(tr(LANG, f"观测数 & \\multicolumn{{2}}{{c}}{{{n_obs:,}}} \\\\", f"Observations & \\multicolumn{{2}}{{c}}{{{n_obs:,}}} \\\\"))
lines.append(r"\midrule")
lines.append(tr(LANG, r"\multicolumn{3}{l}{\textit{格兰杰因果检验}} \\", r"\multicolumn{3}{l}{\textit{Granger causality tests}} \\"))
for name, (F, pval) in granger_results.items():
    stars = "***" if pval<0.01 else ("**" if pval<0.05 else ("*" if pval<0.1 else ""))
    lines.append(f"{name} & F={F:.3f}{stars} & p={pval:.4f} \\\\")

lines.append(r"\bottomrule")
lines.append(tr(LANG, r"\multicolumn{3}{p{0.7\textwidth}}{\footnotesize 注：Panel VAR 含话题固定效应（within 变换）。括号内为标准误。***、**、*分别表示在1\%、5\%、10\%水平上显著。IRF 使用 Cholesky 分解（态度优先）。Bootstrap 置信区间基于999次重采样。}", r"\multicolumn{3}{p{0.7\textwidth}}{\footnotesize Notes: Panel VAR includes topic fixed effects (within transformation). Standard errors in parentheses. ***, **, * denote significance at 1\%, 5\%, 10\% levels. IRFs use Cholesky decomposition with stance ordered first. Bootstrap confidence intervals are based on 999 replications.}"))
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tab_a6_name = tr(LANG, "TabA6_PanelVAR系数.tex", "TabA6_PanelVAR_Coefficients.tex")
tex_path = os.path.join(TABLES, tab_a6_name)
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  → 已保存：{tex_path}")

print("\n完成。")
