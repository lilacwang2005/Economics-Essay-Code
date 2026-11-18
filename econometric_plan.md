# 完整计量方案：公众舆论与政策落实的因果关系

## 研究问题

人大代表/政协委员的建议登上微博热搜后，舆论热度、公众态度（支持/反对）、公众情感（正面/负面）是否影响该建议被政府落实的概率与速度？

---

## 一、变量构建

### 1.1 因变量（政策落实）

**数据来源**：人民日报匹配结果（经规则粗筛 + LLM二次筛选后），后续叠加新华网、前瞻政策数据库等多源数据。

#### 因变量1：是否被落实（二元）

$$Solved_i = \begin{cases} 1, & \text{若热搜 } i \text{ 的核心诉求在建议提出后有至少一条"已落实/部分落实"的新闻证据} \\ 0, & \text{否则} \end{cases}$$

构建方法：LLM对筛选后的新闻逐条判断 solve_status ∈ {已完全解决, 部分解决, 正在推进, 仅讨论, 无关}。只要有一条 ≥ "部分解决" 即标记为1。

#### 因变量2：落实程度（有序/连续）

$$SolveDegree_i = \max_{j \in \mathcal{N}_i^{post}} \; score(solve\_status_j)$$

其中 $\mathcal{N}_i^{post}$ 是热搜 $i$ 对应的、发布于建议提出之后的所有相关新闻集合。score 映射：无关=0, 仅讨论=1, 正在推进=2, 部分解决=3, 已完全解决=4。取最大值代表该建议截至2026年3月的最高落实程度。

#### 因变量3：落实速度（生存分析用）

$$DaysToSolve_i = \min_{j \in \mathcal{N}_i^{post}: \; score_j \geq 2} \; (date_j - date_i^{proposal})$$

即从建议上热搜到首条"正在推进"及以上新闻的天数。若截至2026年3月仍未落实，则为右删失（right-censored）。

#### 因变量4：Event Study 用面板因变量

对每条热搜 $i$，以建议上热搜的月份（或季度）为 $t=0$，构建相对时间面板：

$$y_{it} = \begin{cases} \text{月度：} & \text{在相对月份 } t \text{ 中，热搜 } i \text{ 匹配到的"落实相关"新闻数量（或布尔值：是否有）} \\ \text{季度：} & \text{同上，以季度为窗口} \end{cases}$$

具体而言，$y_{it}$ 可以是：
- **密集边际**（intensive margin）：相对时期 $t$ 内匹配到的落实相关新闻条数
- **广延边际**（extensive margin）：$\mathbf{1}(y_{it} > 0)$，该时期是否有落实信号
- **累积指标**：截至相对时期 $t$，累计的最高 SolveDegree

推荐主回归使用广延边际（更干净），密集边际作为稳健性。

### 1.2 核心自变量（舆论特征）

**数据来源**：微博热搜posts + comments（BERT全量推理结果）。

以下变量均在热搜 $i$ 层面聚合：

#### 热度指标

$$Heat_i = \text{需要构建一个综合热度指数}$$

推荐做法：对点赞数、转发数、评论数三个维度分别取对数（消除右偏），然后用**主成分分析（PCA）**提取第一主成分作为综合热度指数。理由：

- 点赞 = 被动认同（低成本参与）
- 转发 = 主动扩散（中等成本参与）
- 评论 = 深度参与（高成本参与）

三者捕捉不同维度的公众关注，PCA自动赋权。

具体步骤：
1. 对每条热搜，汇总所有post的总点赞数 $Likes_i$、总转发数 $Reposts_i$、总评论数 $Comments_i$
2. 取对数：$\ln(1+Likes_i)$, $\ln(1+Reposts_i)$, $\ln(1+Comments_i)$
3. 标准化为均值0方差1
4. PCA取第一主成分 → $HeatPC1_i$

同时保留三个分项指标用于异质性分析。

#### 态度指标

$$Stance_i = \frac{1}{|\mathcal{C}_i|} \sum_{j \in \mathcal{C}_i} stance\_score_j$$

其中 $\mathcal{C}_i$ 是热搜 $i$ 下所有评论的集合。$Stance_i \in [-100, +100]$，正值代表公众整体支持该建议，负值代表反对。

同时构建：
- $StanceSD_i$：态度标准差（衡量意见分歧程度）
- $StancePositiveRatio_i$：stance_score > 20 的评论占比（支持率）

#### 情感指标

$$Sentiment_i = \frac{1}{|\mathcal{C}_i|} \sum_{j \in \mathcal{C}_i} sentiment\_score_j$$

正值代表公众情绪正面，负值代表负面。

同时构建：
- $AngerRatio_i$：emotion = "愤怒" 的评论占比（捕捉公众愤怒程度，可能对政府施压更大）

#### 连续型 DID 处理变量

在 event study 框架中，"处理强度"不是二元的（有/无），而是连续的。每条热搜都"被处理了"（都上了热搜），但处理强度不同。因此采用**连续型 DID**：

$$Treatment_i = HeatPC1_i \quad \text{（或 } Stance_i \text{，或二者交互）}$$

这意味着：不是比较"上热搜 vs 没上热搜"，而是比较"高热度热搜 vs 低热度热搜"在事件前后的落实差异。

### 1.3 控制变量

| 变量 | 构建方法 | 理由 |
|------|---------|------|
| $Category_i$ | LLM标注的核心诉求领域（教育/医疗/住房/就业/环保/法治/经济/交通/社会保障/科技/其他） | 不同领域的建议天然有不同的落实概率 |
| $Year_i$ | 建议上热搜的年份 | 控制宏观政策环境变化 |
| $NumPosts_i$ | 热搜下的post数量 | 控制话题的信息供给量 |
| $BotRatio_i$ | 被标记为bot的评论占比 | 控制水军干扰 |
| $PreBaseline_i$ | 建议提出前12个月内匹配到的相关新闻数 | 控制"该问题在建议前就已经在被解决" |
| $IsGovRelated_i$ | LLM判断的 is_government_related | 过滤与政府无关的热搜 |
| $KeywordCount_i$ | 该热搜提取的关键词数量 | 控制匹配到新闻的机械性差异（关键词越多越容易匹配到新闻） |
| $TextLength_i$ | 热搜下所有评论的平均文本长度 | 代理公众讨论深度 |

### 1.4 样本筛选

从1100条热搜中筛选：
1. 保留 is_government_related = True 的热搜
2. 保留 LLM二次筛选后至少匹配到1条相关新闻的热搜（否则因变量无法构建）
3. 去除 BotRatio > 50% 的热搜（严重水军干扰）

最终分析样本量取决于筛选后剩余多少条。

---

## 二、Baseline 模型（横截面回归）

### 模型B1：OLS / Ordered Probit

最简单的横截面回归，建立基本事实。

**OLS（因变量 = SolveDegree）**：

$$SolveDegree_i = \alpha + \beta_1 HeatPC1_i + \beta_2 Stance_i + \beta_3 Sentiment_i + \mathbf{X}_i'\boldsymbol{\gamma} + \delta_{c(i)} + \delta_{y(i)} + \epsilon_i$$

其中：
- $\delta_{c(i)}$：领域固定效应（Category FE）
- $\delta_{y(i)}$：年份固定效应
- $\mathbf{X}_i$：其他控制变量向量

$\beta_1$ 的解释：热度每增加一个标准差，落实程度增加 $\beta_1$ 个单位。

**Ordered Probit（因变量 = SolveDegree 作为有序分类 0-4）**：

$$P(SolveDegree_i \leq k) = \Phi(\mu_k - \mathbf{Z}_i'\boldsymbol{\beta})$$

用于稳健性，因为 SolveDegree 是有序变量，OLS假设等距可能不合理。

**Probit / Logit（因变量 = Solved 二元）**：

$$P(Solved_i = 1) = \Phi(\alpha + \beta_1 HeatPC1_i + \beta_2 Stance_i + \beta_3 Sentiment_i + \mathbf{X}_i'\boldsymbol{\gamma} + \delta_{c(i)} + \delta_{y(i)})$$

### 模型B2：分项拆解

把 HeatPC1 拆回三个分项，看哪种参与形式最能预测落实：

$$SolveDegree_i = \alpha + \beta_L \ln(1+Likes_i) + \beta_R \ln(1+Reposts_i) + \beta_C \ln(1+Comments_i) + \beta_S Stance_i + \beta_E Sentiment_i + \mathbf{X}_i'\boldsymbol{\gamma} + \delta_{c(i)} + \delta_{y(i)} + \epsilon_i$$

如果 $\beta_R > \beta_L > \beta_C$，说明转发（扩散力）对政策落实的边际影响最大。

### 标准误

聚类到领域层面（Category），因为同一领域的建议可能存在相关的未观测因素（如某领域恰好有重大政策窗口）。如果领域类别太少（<20），改用 robust SE。

---

## 三、主回归：Event Study（连续型 DID）

### 模型M1：月度 Event Study

构建热搜 $i$ × 相对月份 $t$ 的面板数据。$t=0$ 为建议上热搜的月份，$t \in [-12, +24]$（前12个月到后24个月）。

$$y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}(t=k) \cdot HeatPC1_i + \mathbf{X}_{it}'\boldsymbol{\theta} + \epsilon_{it}$$

其中：
- $y_{it}$：热搜 $i$ 在相对月份 $t$ 是否有落实信号（广延边际，0/1）
- $\alpha_i$：热搜固定效应（吸收热搜层面所有不随时间变化的特征）
- $\lambda_t$：相对时间固定效应
- $\mathbf{1}(t=k) \cdot HeatPC1_i$：连续型处理变量与时间虚拟变量的交互
- 基准期 $k=-1$（建议上热搜前一个月）

**$\beta_k$ 的解释**：在相对月份 $k$，热度每增加1个标准差，落实信号出现的概率增加 $\beta_k$。

**平行趋势检验**：$k < 0$ 的 $\beta_k$ 应该不显著且接近零——意味着在建议上热搜之前，高热度和低热度的建议在落实方面没有系统差异。

**输出**：Event Study 图，横轴为相对月份 $k$，纵轴为 $\beta_k$ 及95%置信区间。基准期 $k=-1$ 处 $\beta_{-1}=0$。

### 模型M2：季度 Event Study

与M1相同，但 $t$ 以季度为单位，$t \in [-4, +8]$（前4个季度到后8个季度）。

推荐在数据稀疏时使用（如果月度面板中每个 cell 的 $y_{it}$ 变异太小）。

### 模型M3：态度的 Event Study

把处理变量从热度换成态度：

$$y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \gamma_k \cdot \mathbf{1}(t=k) \cdot Stance_i + \mathbf{X}_{it}'\boldsymbol{\theta} + \epsilon_{it}$$

检验：公众越支持的建议，是否在事后落实得越快？$\gamma_k > 0$ 且随 $k$ 递增意味着支持态度加速了落实。

### 模型M4：热度 × 态度交互的 Event Study

$$y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \left[ \beta_k^H \cdot \mathbf{1}(t=k) \cdot HeatPC1_i + \beta_k^S \cdot \mathbf{1}(t=k) \cdot Stance_i + \beta_k^{HS} \cdot \mathbf{1}(t=k) \cdot HeatPC1_i \cdot Stance_i \right] + \epsilon_{it}$$

$\beta_k^{HS}$ 检验：高热度 + 高支持的建议是否比高热度 + 低支持的建议落实更快？这是最核心的检验——"舆论热度只有在方向对的时候才有效"。

### 实现细节

**估计方法**：用 `pyfixest` 的 `feols`，支持高维固定效应和聚类标准误。

```python
import pyfixest as pf

# M1: 月度 Event Study
model = pf.feols(
    "y ~ i(rel_month, HeatPC1, ref=-1) + controls | hotspot_id + rel_month",
    data=panel_df,
    vcov={"CRV1": "category"}
)
```

**标准误**：两种聚类方式都报告：
- 聚类到热搜层面（baseline）
- 聚类到领域层面（更保守）

---

## 四、补充模型

### 模型S1：Cox 比例风险模型（生存分析）

因变量 = DaysToSolve（从上热搜到首次出现落实信号的天数），右删失 = 截至2026年3月仍未落实。

$$h_i(t) = h_0(t) \cdot \exp(\beta_1 HeatPC1_i + \beta_2 Stance_i + \beta_3 Sentiment_i + \mathbf{X}_i'\boldsymbol{\gamma} + \delta_{c(i)})$$

$\exp(\beta_1)$ 的解释：热度每增加1个标准差，落实的"风险率"（hazard rate，此处为正面含义——即落实的概率）增加 $(\exp(\beta_1)-1) \times 100\%$。

**输出**：
- 按热度高低分组的 Kaplan-Meier 生存曲线（中位数分组）
- Cox 回归系数表

**优势**：自然处理右删失（很多建议可能尚未落实），且不需要构建面板。

### 模型S2：愤怒效应（Anger Channel）

检验假说：公众愤怒情绪是否比一般负面情感更能推动政策落实？（因为愤怒可能对政府形成更强的问责压力）

$$SolveDegree_i = \alpha + \beta_1 HeatPC1_i + \beta_2 Stance_i + \beta_3 Sentiment_i + \beta_4 AngerRatio_i + \mathbf{X}_i'\boldsymbol{\gamma} + \delta_{c(i)} + \delta_{y(i)} + \epsilon_i$$

如果 $\beta_4 > 0$ 且显著，说明愤怒情绪有独立于一般情感之外的额外推动作用。

### 模型S3：意见分歧效应

检验假说：意见越一致（标准差小）的热搜，是否更容易推动落实？

$$SolveDegree_i = \alpha + \beta_1 HeatPC1_i + \beta_2 Stance_i + \beta_3 StanceSD_i + \beta_4 HeatPC1_i \times StanceSD_i + \mathbf{X}_i'\boldsymbol{\gamma} + \epsilon_i$$

$\beta_4 < 0$ 意味着：当公众意见越一致时，热度对落实的推动效应越强。

### 模型S4：领域异质性

在 Baseline 模型中加入热度 × 领域虚拟变量的交互项：

$$SolveDegree_i = \alpha + \sum_c \beta_c \cdot HeatPC1_i \times \mathbf{1}(Category_i = c) + \mathbf{X}_i'\boldsymbol{\gamma} + \delta_{c(i)} + \delta_{y(i)} + \epsilon_i$$

哪些领域的舆论压力最能推动政策落实？预期：民生类（教育、医疗、住房）> 经济类 > 外交类。

---

## 五、稳健性检验

### R1：替代热度指标

分别用以下替代 HeatPC1：
- 原始三分项的对数值
- 仅用 $\ln(1+Comments_i)$
- 热搜在榜时长（如果数据中有）
- 分位数虚拟变量（把热度分为4组，检验是否存在非线性）

### R2：替代因变量

- 用密集边际（新闻条数）替代广延边际
- 用 Ordered Probit 替代 OLS
- 用是否在前瞻政策数据库中匹配到正式政策文件作为替代因变量（更"硬"的落实证据）

### R3：排除反向因果

担忧：可能不是"舆论推动落实"，而是"政府本来就打算落实的建议恰好也引发了高热度"。

检验方法：
- 加入 PreBaseline 控制（建议前12个月的相关新闻数），如果结果不变，说明不是因为"该问题已经在被推动"
- 只保留 PreBaseline = 0 的子样本（建议前完全没有相关新闻的"冷启动"建议），重跑主回归

### R4：安慰剂检验（时间错位）

将每条热搜的上热搜时间随机打乱（在样本时间范围内重新抽取），重跑 Event Study。重复500次，看真实 $\beta_k$ 是否在安慰剂分布的极端位置。

### R5：排除水军干扰

- 去除 BotRatio > 10% 的热搜（更严格的筛选）
- 去除 bot 和 spam 标记的评论后重新计算所有舆论指标，重跑全部回归

### R6：多源因变量一致性

分别用以下数据源构建因变量，检验结果是否一致：
- 仅人民日报
- 仅新华网
- 仅前瞻政策数据库
- 三者合并

如果不同数据源的结果方向一致，大大增强可信度。

### R7：样本选择偏误

担忧：只有匹配到新闻的热搜才能构建因变量，匹配不到的被排除 → 样本选择偏误。

检验方法：Heckman 两步法。第一步用 Probit 估计"是否能匹配到至少一条新闻"的概率（选择方程），计算逆米尔斯比率，加入第二步的主回归。

### R8：不同时间窗口

Event Study 中改变时间窗口：
- 缩短到 [-6, +12] 月（更短窗口，检验短期效应）
- 延长到 [-12, +36] 月（如果数据支持的话）

---

## 六、图表规划

### 图

| 编号 | 内容 | 模型 |
|------|------|------|
| Fig 1 | 描述性：热度分布直方图 + 态度分布直方图 + 落实程度分布柱状图 | — |
| Fig 2 | 描述性：按落实程度分组的平均热度/态度/情感对比（条形图 + 误差线） | — |
| Fig 3 | **Event Study 图（主图）**：月度 $\beta_k$ + 95% CI，横轴 $k$，纵轴 $\beta_k$ | M1 |
| Fig 4 | Event Study 图：季度版本 | M2 |
| Fig 5 | Event Study 图：态度版本 | M3 |
| Fig 6 | Kaplan-Meier 生存曲线：按热度高低分组 | S1 |
| Fig 7 | 领域异质性：各领域的热度系数 $\beta_c$ + 95% CI | S4 |
| Fig 8 | 安慰剂检验：500次安慰剂 $\beta_k$ 分布 + 真实值竖线 | R4 |
| Fig 9 | 稳健性汇总：不同设定下的核心系数及CI（coefficient stability plot） | R1-R8 |

### 表

| 编号 | 内容 | 模型 |
|------|------|------|
| Tab 1 | Summary Statistics | — |
| Tab 2 | Baseline OLS + Probit + Ordered Probit | B1 |
| Tab 3 | 分项拆解（点赞/转发/评论分别的系数） | B2 |
| Tab 4 | Event Study 系数表（月度 + 季度） | M1, M2 |
| Tab 5 | 热度 × 态度交互 Event Study | M4 |
| Tab 6 | Cox 比例风险模型 | S1 |
| Tab 7 | 愤怒效应 + 意见分歧效应 | S2, S3 |
| Tab 8 | 领域异质性 | S4 |
| Tab 9 | 稳健性检验汇总 | R1-R8 |

---

## 七、实施优先级

1. **最先做**：数据清洗 + 因变量构建（规则粗筛21G → LLM二次筛选 → 构建 Solved / SolveDegree / DaysToSolve）
2. **然后做**：自变量构建（PCA热度指数 + 聚合态度/情感指标）+ 领域标注
3. **接着做**：Baseline 横截面回归（B1, B2）→ 快速看到初步结果
4. **核心做**：构建面板 → Event Study（M1-M4）+ Event Study 图
5. **补充做**：Cox 生存分析（S1）→ 异质性分析（S2-S4）
6. **最后做**：稳健性检验（R1-R8）

---

## 八、关于识别的说明（论文中需要讨论）

### 识别假设

连续型 DID 的核心假设是：**在控制了领域、年份、PreBaseline 等之后，同一领域内不同热搜的热度差异对落实趋势的影响是外生的。**

通俗地说：两条同属教育领域、同年提出的建议，一条恰好热度很高、一条热度较低，这个热度差异不是因为"政府本来就打算落实第一条所以媒体更关注"，而是因为热搜算法、发布时间、评论互动等准随机因素导致的。

### 支持识别的论证

1. **平行趋势**：Event Study 的事前系数不显著
2. **PreBaseline 控制**：加入建议前的新闻密度，排除"已在推进"的混杂
3. **安慰剂检验**：随机打乱时间后效应消失
4. **机制检验**：高热度 + 高支持的交互效应最强（如果只是反向因果，交互项不应该显著——因为政府推进不取决于公众态度方向）

### 诚实地讨论的局限

- 不能完全排除遗漏变量（如建议的"质量"难以度量）
- 因变量基于新闻报道，而非直接的政策执行数据
- 热搜上榜时间不等于提案正式提交时间
