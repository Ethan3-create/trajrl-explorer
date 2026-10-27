# 具备多尺度时空特征探索的自监督轨迹表示学习 #

Hong Xia $^{1}$, Xiao Zhang $^{1}$, Yuan Cao $^{1}$, Lei Cao $^{2}$, Yanwei Yu $^{1*}$, Junyu Dong $^{1}$

 $^{1}$中国海洋大学信息科学与工程学部，中国青岛

 $^{2}$亚利桑那大学计算机科学系，美国图森

{zhong, zhangxiao4549}@stu.ouc.edu.cn, lcao@csail.mit.edu,

{cy8661, yuyanwei, dongjunyu}@ouc.edu.cn

摘要——轨迹表示学习将轨迹复杂的时空特征转化为稠密的低维嵌入，可支撑轨迹分类、出行时间估计和相似轨迹搜索等多种下游分析任务。现有的轨迹表示学习方法将轨迹仅仅视为一般的点序列，并使用序列模型学习点之间的相关性。然而，轨迹复杂的时空特征是多尺度的，这意味着它们不仅体现在轨迹点之间的相关性上，还体现在轨迹片段之间的相关性上。此外，大多数现有方法未能充分捕获轨迹内部多方面的时序特征。为填补上述空白，我们提出了一种新颖的具备多尺度时空特征探索的自监督轨迹表示学习模型，命名为 TrajRL。具体而言，我们利用轨迹增强生成两个视图，以利用多个自监督信号实现自监督预训练。在每个视图中，我们通过所提出的多尺度轨迹编码器学习轨迹中路段内部及之间、以及路段序列内部及之间的多尺度时空相关性。此外，我们进行了多方面的时序信息编码，特别是利用时间间隔来学习轨迹内部具有多尺度上下文感知的时间模式。大量实验表明，在两个真实数据集上的多个下游任务中，我们的 TrajRL 相较于最先进的基线方法具有优越性。我们模型的源代码可在 https://github.com/Xfc30/TrajRL 获取。

### I. 引言

随着支持 GPS 的设备以及包括地图服务和基于位置的社会网络在内的基于位置服务的发展，产生了大量的轨迹数据。作为最有代表性的时空数据类型之一，轨迹数据为人和物体的移动行为提供了宝贵的洞察。轨迹数据的分析与挖掘 [1] [2] 在诸多领域发挥着至关重要的作用，如城市规划 [3] [4] [5] [6]、智能交通系统 [7] [8] [9] [10] [11] 和传染病防控 [12]。

 $^{*}$通讯作者：Yanwei Yu。

轨迹表示学习（trajectory representation learning，TRL）旨在将轨迹从稀疏的高维空间映射到有意义的、稠密的低维表示，这些表示可用于多种任务。

早期关于 TRL 的研究基于针对特定下游任务设计并以端到端方式训练的序列模型 [13] [14]。这些方法虽然直观，但存在可扩展性和效率方面的局限性 [15] [16]。具体而言，所学到的表示是针对特定任务定制的，使其难以跨任务应用。此外，端到端的嵌入方法依赖于大量任务标签的可用性才能获得令人满意的结果。为了解决这些问题，自监督 TRL 变得越来越流行。首先提出了基于 seq2seq 的方法，这些方法使用重建损失进行训练 [17] [14] [18]。为了避免细粒度 GPS 轨迹中的冗余和噪声，一些方法使用路段序列来替代 GPS 轨迹 [19] [20] [21]。最近，随着图神经网络的兴起，提出了两阶段方法 [21]。这些方法首先以路网图作为输入，通过图神经网络学习路段的嵌入。然后，将路段表示输入到基于序列的深度学习模型中，并使用自监督任务进行训练，从而学习轨迹表示。

尽管如此，自监督 TRL 中仍有两个主要挑战有待解决。(1) 在多个下游任务上泛化能力差。由于每个下游任务都有其独特的特征，现有 TRL 方法通常在某些任务上表现良好，但在其他任务上表现不佳。失败的主要原因是现有方法将轨迹视为一般的点序列，并应用 RNN（循环神经网络）或 Transformer 来捕获点之间的相关性。然而，嵌入在轨迹中的时空特征和出行模式不仅存在于单个轨迹点之间，还存在于轨迹片段之间以及不同轨迹之间。换言之，轨迹天生具有多尺度特征，而先前的工作仅关注点级或片段级特征，忽视了这一特性。重新审视 TRL 的现实下游任务，不同的任务往往与不同尺度的轨迹特征相关。例如，目的地预测与轨迹中单个点的特征高度相关，而轨迹相似度度量则与轨迹的整体结构更相关。因此，开发一种能够捕获多尺度时空特征的轨迹表示学习方法，对于获得高质量的表示并增强各种下游任务的泛化能力至关重要。

(2) 时序信息学习不足。大多数 TRL 方法仅将轨迹视为有序的位置序列，并专注于挖掘空间信息来学习轨迹表示。然而，作为一种时空数据，时序信息同样应被捕获。据我们所知，在我们之前的工作中，只有 Trembr [22] 和 START [21] 在自监督 TRL 中考虑了时序信息。Trembr 是一种 seq2seq 模型，其解码器考虑了轨迹点的时间戳，而 START 在路段表示中融入了时间周期性编码。

除了时间周期性之外，轨迹点之间的时间间隔还包含额外的有价值信息。即使在空间域中，两条完全相同的轨迹也可能具有显著不同的时间间隔分布。这可能反映了轨迹生成时路网的交通状况，以及产生轨迹的个体或物体惯常的移动模式。例如，城市交通的拥堵程度和个体的驾驶习惯都反映在时间间隔信息中。因此，需要将更全面的时序信息提取融入轨迹表示学习方法中。

为了应对上述挑战，我们提出了一种新颖的具备多尺度时空特征探索的自监督轨迹表示学习方法，命名为 TrajRL。所提出的方法探索多方面的时序信息挖掘，旨在通过两个自监督任务学习多尺度的轨迹特征。具体而言，我们利用路网匹配和轨迹增强生成两个视图，以构建自监督预训练框架。在每个视图中，我们首先使用 GAT（图注意力网络）学习路段的初始嵌入，该嵌入隐含了路网连通性和路段特征。然后，我们使用所提出的多尺度轨迹编码器在路网约束下的轨迹中学习不同尺度的时空相关性。此外，我们进一步将时间间隔划分为多个尺度，在捕获上下文感知的时间模式的同时避免细粒度噪声的影响。具体而言，所提出的多尺度轨迹编码器将轨迹划分为多个轨迹片段，并学习这些片段内部和之间的相关性。通过逐层聚合，它有效地捕获了从细粒度到粗粒度的多个尺度上的轨迹特征。此外，所提出的多方面时间编码器同时执行时间周期性编码和时间间隔编码。

为了在多个尺度上预训练我们的 TrajRL，我们采用了片段级轨迹恢复和轨迹对比学习两个任务。前者提供细粒度的自监督信号，后者提供粗粒度的自监督信号。

本文的主要贡献总结如下：

- 我们提出了一种新颖的自监督轨迹表示学习框架，该框架充分探索轨迹数据中的多尺度时空特征，为各种下游挖掘任务学习通用的轨迹表示。为预训练我们的 TrajRL，我们应用了两个在不同尺度上提供自监督信号的自监督任务。

- 我们构建了一个多方面的时序编码器，通过设计的时间周期编码器和时间间隔编码器，分别有效学习轨迹中的时间周期规律和多尺度上下文感知的时间模式。

- 我们设计了一个多尺度轨迹编码器，通过在不同尺度上设计的 InPatch 编码器和 CrossPatch 编码器，同时捕获轨迹中路段与路段序列内部及之间的时空相关性。

- 在两个真实数据集上的三个下游任务上进行的实验验证了我们的模型达到了最佳性能，并展现出一定的效率优势。

### II. 相关工作

我们将从两个方面简要回顾 TRL 的相关工作：轨迹时空信息的学习和轨迹表示的自监督学习。

### A. 轨迹时空信息的学习

1) 轨迹空间信息的利用：轨迹表示学习旨在将原始轨迹映射到一个通用的高维潜在向量空间，可应用于轨迹数据挖掘的各个领域，如相似轨迹搜索、出行时间估计和异常检测 [23] [24] [25] [26] [27]。早期的轨迹表示学习关注自由空间中的轨迹，将其视为简单的序列数据，因此无法在嵌入中保留时空信息。[14] [18] [28] 通常使用 GPS 轨迹作为模型输入，自然忽略了 GPS 轨迹数据与真实路网之间的交互。

近年来，提出了几种两阶段方法。第一步是学习路段的嵌入。Toast [19] 通过引入辅助任务，以自监督方式预测交通相关的上下文特征（如限速），从而训练 skip-gram [29] 模型以获得路网的嵌入。当路网上下文信息不可用时，这种方法可能难以有效运作。更稳健的替代方案是图神经网络（graph neural network，GNN），它能更有效地利用路网的结构信息。JCLRNT [30] 采用更强大的 GAT（图注意力网络）[31] 来处理路网信息。Trembr [22] 利用多任务学习框架，同时考虑路段之间的共现关系和类型关系来学习其嵌入。START [21] 利用路网的性质并考虑车辆转移概率，学习稳健且通用的表示。第二步是将轨迹编码为序列。为了获得轨迹嵌入，上述方法几乎都使用基本的序列模型，如 RNN 或 Transformer。为了增强模型的可扩展性，LightPath [32] 提出在输入 Transformer 编码器之前对轨迹执行稀疏化操作。具体而言，最近的工作 MM-STGED [33] 直接将轨迹建模为图。在本研究中，我们考虑不同尺度下轨迹片段之间的关系，并探索多尺度轨迹编码器的设计。

2) 轨迹时序信息的利用：Trembr 在解码器中引入时间信息用于轨迹重建，但它没有对时间信息进行显式建模。START 通过时间嵌入提取时间周期性，并在 Transformer 编码器的自注意力权重计算中使用时间信息以增强信息聚合。我们设计了一个多方面的时序编码器，对时间周期性和时间间隔进行显式建模。

### B. 轨迹表示的自监督学习

早期轨迹表示学习的工作主要依赖带标签数据 [34] [10] [35] [36] 进行端到端训练，这带来了两个主要挑战。首先，这些模型生成的嵌入往往有偏且缺乏泛化性，难以应用于其他任务 [37] [38] [39]。其次，获取数据的真实标签本身代价高昂。

为了解决这些问题，利用自监督学习预训练轨迹嵌入的研究兴趣日益增长。自监督学习利用数据本身生成的标签或辅助任务进行训练，使模型在缺乏标签的情况下也能获得通用且稳健的嵌入 [40] [41]。在 TRL 领域，基于序列模型的自编码器最初被广泛采用，基于重建损失学习轨迹表示。T2vec [14] 采用基于 RNN 的 seq2seq 模型，通过最大化从低采样率轨迹重建高分辨率轨迹的条件概率，来学习轨迹的潜在表示。Traj2vec [18] 考虑轨迹数据的时空变换，利用连续轨迹点之间一组属性的差异作为输入特征。此外，如前所述，Trembr [22] 旨在通过同时重建输入轨迹的路段和时间戳来学习表示。

Toast [19] 和 PIM [20] 分别使用基于 Transformer 的掩码语言模型（masked language model，MLM）[42] 预测和基于 RNN 的互信息最大化作为自监督任务，以生成通用的轨迹表示。TrajCL [28] 采用对比学习方法学习更具判别性的特征表示。类似地，START [21] 使用对比学习来训练编码器，并额外引入了轨迹恢复的前置任务（pretext task）。

为了提高轨迹嵌入的泛化能力，MMTEC [43] 使用最大熵原理确保同一轨迹两个视图之间的一致性，从而减少使用不同自监督任务时的预训练偏差。我们采用了两个在不同尺度上提供自监督信号的自监督任务。更重要的是，这种自监督预训练方法能够相互增强我们的多尺度轨迹表示学习模型，从而获得更通用的表示。

### III. 预备知识

在本节中，我们首先介绍本文使用的基本预备知识，然后正式定义轨迹表示学习问题。

*表 I：常用符号汇总。*

| 符号 | 定义 |
|---|---|
| p | 原始 GPS 轨迹点。 |
| v | 路段。 |
| t | 轨迹中一个点或一条路段的访问时间 |
| T | 路网约束轨迹 |
| n | 轨迹的长度 |
| $h_{i}^{t}$ | GAT 第 l 层中图 G 的第 i 个节点的表示嵌入。 |
| $\Delta$ | 轨迹的时间间隔序列 |
| $T_{i}^{P}$ | 时间 $t_{i}$ 的时间周期嵌入 |
| $t_{S(i,j)}$ | 第 i 个时间间隔按第 j 个尺度系数缩放后的嵌入 |
| $T_{t_{i}}^{S}$ | 时间 $t_{i}$ 的多尺度时间间隔嵌入 |
| $x_{i}$ | 给定轨迹中一个路段的嵌入。 |
| $\omega$ | 片段掩码层中的窗口大小。 |
| $e_{i}$ | 由 InPatch 编码器获得的第 i 个片段的嵌入 |
| $z^{(l)}$ | 轨迹 T 的第 l 个尺度的轨迹嵌入 |
| z | 轨迹 T 的嵌入 |

定义 1（路网）。路网表示为一个有向图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{F})$，其中 $\mathcal{V}$ 是顶点集合，每个顶点 $v_i \in \mathcal{V}$ 代表一个路段。$\mathcal{E}$ 是边集合，每条边 $e_{i,j} = (v_i, v_j)$ 表示路段 $v_i$ 与 $v_j$ 之间的相交关系。$\mathcal{F} \in \mathbb{R}^{|\mathcal{V}| \times f}$ 是路段的特征矩阵，其中 $f$ 是特征维度。

定义 2（基于 GPS 的轨迹）。基于 GPS 的轨迹 $\mathcal{T}^{raw} = \langle p_1, p_2, \ldots \rangle$ 是空间-时间采样点的序列，每个采样点 $p_i = \langle l_i, t_i \rangle$。

定义 3（路网约束轨迹）。路网约束轨迹 $\mathcal{T} = \langle r_1, r_2, \ldots, r_n \rangle$ 是按时间排序的相邻路段的序列，其中 $r_i = \langle v_i, t_i, \mu_i \rangle$ 表示序列中的第 i 个路段，且 $t_i < t_{i+1}$，$\mu_i \in \mathbb{N}$ 表示沿该路段的（移动）速度。

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0004-00.jpg)
*(a) 多方面时序编码器*

*(b) 多尺度轨迹编码器*

*图 1：所提出的 TrajRL 的总体框架。*

在本工作中，我们主要关注路网约束轨迹。为简洁起见，下文使用"轨迹"指代路网约束轨迹。

接下来，我们正式定义所研究的轨迹表示学习问题如下：

问题 1. 给定一个路网 $\mathcal{G}$ 和一个轨迹数据集 $\mathcal{D} = \{\mathcal{T}_i\}_{i=1}^{|\mathcal{D}|}$，轨迹表示学习旨在为每个轨迹 $\mathcal{T}_i \in \mathcal{D}$ 学习一个保持时空相关性的通用低维向量 $h_i \in \mathbb{R}^d$。

在本工作中，我们的目标是为每条轨迹获取一个稳健且通用的表示向量，使其在多个下游任务上达到最优性能。

### IV. 方法论

在本节中，我们介绍所提出的 TrajRL（轨迹表示学习）框架的细节，如图 1 所示。我们提出的 TrajRL 由五个关键组件构成：（1）路网匹配与轨迹增强（Road-network Matching and Trajectory Augmentation），（2）路段编码器（Road Segment Encoder），（3）多面时间编码器（Multi-faceted Time Encoder），（4）多尺度轨迹编码器（Multi-scale Trajectory Encoder），以及（5）自监督预训练（Self-supervised Pre-training）。

### A. 路网匹配与轨迹增强

首先，我们执行地图匹配 [44]，将原始轨迹（即基于 GPS 的轨迹）中的点映射到路段上，从而得到受路网约束的轨迹。具体而言，基于 GPS 的轨迹被转换为相邻路段序列。对于城市中的移动对象，尤其是车辆，这种轨迹预处理与城市交通场景的实际情况是一致的。

然后，为了捕获轨迹的全局特征和局部特征，我们采用了一种基于轨迹增强的对比学习框架。通过数据增强，对于每条轨迹，我们可以获得两个不同的视图，然后使用后文介绍的综合轨迹编码器来学习对齐的轨迹表示，以进行自监督预训练。

### B. 路段编码器

给定路网的图结构，我们使用图神经网络来学习路段的表示。一方面，真实轨迹受路网拓扑结构的约束；另一方面，路段的属性（如车道数和最高限速）对驾驶员的路线规划有显著影响。因此，我们同时从路段的静态属性和路网的拓扑结构两方面来学习路段的表示。

首先，我们为每条路段考虑四个特征，包括路段类型、长度、车道数和最高行驶速度，以及其在路网中的出度和入度。我们对连续特征值进行归一化处理，对离散特征值进行独热编码。所有编码后的特征被拼接为路段的初始嵌入，即第 $i$ 条路段的 $\vec{v}_i \in \mathbb{R}^d$。

为了灵活地整合邻居节点的嵌入，我们采用 GAT（图注意力网络）[31] 在路网内进行消息传递。图 G 中的节点按如下方式聚合来自其邻居节点的信息：

$$
h_{i}^{(l+1)}=\mathrm{E L U}\left(\sum_{j\in\mathcal{N}_{i}}\alpha_{i j}^{(l)}h_{j}^{(l)}\mathbf{W}_{1}^{(l)}\right),
$$

其中 $h_i^{(0)} = \vec{v}_i$，$\mathbf{W}_1^{(l)} \in \mathbb{R}^{d \times d}$ 是第 $l$ 层的可学习参数，$\mathcal{N}_i$ 是图 $\mathcal{G}$ 中 $v_i$ 的邻居集合，ELU（指数线性单元）是激活函数，$\alpha_{ij}^{(l)}$ 是第 $l$ 层的注意力权重。

受 [21] 的启发，我们还利用历史数据中路段之间的转移概率来增强注意力权重。第 $l$ 层的注意力权重计算如下：

$$
\begin{aligned}\alpha_{ij}^{(l)}&=\frac{\exp(\mathrm{LeakyReLU}(e_{ij}^{(l)}))}{\sum_{k\in\mathcal{N}_{i}}\exp(\mathrm{LeakyReLU}(e_{ik}^{(l)}))},\\e_{ij}^{(l)}&=(h_{i}^{(l)}\mathbf{W}_{2}^{(l)}+h_{j}^{(l)}\mathbf{W}_{3}^{(l)}+p_{ij}\mathbf{W}_{4}^{(l)})\mathbf{W}_{5}^{(l)^{\mathsf{T}}},\\p_{i,j}&=\operatorname{count}(v_{i}\rightarrow v_{j})/\operatorname{count}(v_{i}),\end{aligned}
$$

其中 $h_i^{(l)}$、$h_j^{(l)} \in \mathbb{R}^{d_l}$ 是路段 $v_i$ 和 $v_j$ 在第 $l$ 层的嵌入，$\mathbf{W}_2^{(l)}$、$\mathbf{W}_3^{(l)} \in \mathbb{R}^{d \times d}$，$\mathbf{W}_4^{(l)}$、$\mathbf{W}_5^{(l)} \in \mathbb{R}^{1 \times d}$ 是可学习参数，$d$ 是 GAT 中隐藏向量的维度，LeakyReLU（带泄漏的修正线性单元）是激活函数，$p_{i,j}$ 是 $v_i$ 与 $v_j$ 之间的转移概率。

多头注意力机制可用来进一步强化上述节点信息聚合。最后，路段表示通过拼接多个头的输出来获得。

### C. 多面时间编码器

轨迹中的时间序列数据几乎与相应的空间序列数据同等重要，然而以往的研究并未对这一方面给予充分的考虑。为了捕获轨迹中的各种时间因素，我们设计了一个多面时间编码器，如图 1 的左下角所示。该模块由时间段编码器（time period encoder）和时间间隔编码器（time interval encoder）组成。

1) 时间段编码器（Time Period Encoder）：我们首先提出一个时间段编码器，用于从轨迹时间序列中捕获城市交通的时间周期性模式。为了提取周、日、小时三个层次的周期性，我们提取星期、小时和分钟，并将它们分别转换为星期索引（1 至 7）、小时索引（1 至 24）和分钟索引（1 至 1440）。对于每个访问时间戳 $t_i$，我们使用三个嵌入 $t_i^d \in \mathbb{R}^d$、$t_i^h \in \mathbb{R}^d$ 和 $t_i^m \in \mathbb{R}^d$ 来表示这三种周期性模式。时间 $t_i$ 的时间段嵌入 $T_i^P$ 为：

$$
T_{t_{i}}^{P}=t_{i}^{d}+t_{i}^{h}+t_{i}^{m}.
$$

2) 时间间隔编码器（Time Interval Encoder）：显然，轨迹中某个路段的周期性编码与该轨迹中的其他路段无关，即它是与上下文无关的，这表明它无法表示轨迹在时间维度上的变化模式。

为了解决这一问题，我们接下来设计了一个多尺度时间间隔编码器。具体而言，我们首先获得轨迹 $<t_1, t_2, \ldots, t_n>$ 的时间序列所对应的间隔序列，其定义为：

$$
\begin{align*}\Delta=&<\Delta_{1},\Delta_{2},\ldots,\Delta_{n}>,\\\Delta_{i}=&\begin{cases}0&\text{if}i=1\\t_{i}-t_{i-1}&\text{if}i>1\end{cases}.\end{align*}
$$

注意，我们为时间间隔设置了一个上界，以忽略不常见的较大时间间隔值，从而避免噪声干扰。然后，我们选择 $\lambda$ 个尺度系数 $\{c_1, c_2, \ldots, c_\lambda\}$ 来缩放时间间隔序列 $\Delta$ 中的值 $\Delta_i$。对于每个尺度 $c_j$，我们定义一个缩放函数为：

$$
S(i,j)=\lfloor\Delta_{i}\times\frac{1}{c_{j}}\rfloor.
$$

然后，所有缩放后的时间间隔序列都经过词嵌入 [45] 处理。经系数 $c_{j}$ 缩放的时间间隔序列被编码为 n 个嵌入：

$$
<t_{S(1,j)},t_{S(2,j)},\ldots,t_{S(n,j)}>,
$$

其中 $t_{S(i,j)} \in \mathbb{R}^d$ 是 $S(i,j)$ 对应的缩放时间间隔嵌入。

最终，通过融合多个缩放后的时间间隔嵌入，我们可以获得每个时间 $t_{i}$ 的多尺度时间间隔嵌入：

$$
T_{t_{i}}^{S}=\sum_{j=1}^{\lambda}t_{S(i,j)}.
$$

从上述可以看出，轨迹中每个路段的时间间隔编码构建了该路段与其前一路段之间的关联，因此时间间隔嵌入序列表示了轨迹在时间维度上的变化模式。

在获得时间段嵌入和时间间隔嵌入之后，我们将两者相加，作为时间 $t_{i}$ 的时间嵌入：

$$
T_{t_{i}}=T_{t_{i}}^{P}+T_{t_{i}}^{S}.
$$

对于给定轨迹 T 中的路段 $v_{i}$，其表示由空间嵌入、时间嵌入和位置编码聚合而成：

$$
x_{i}=h_{i}+T_{t_{i}}+p e_{i},
$$

其中 $pe_{i}$ 表示 T 中用于后续模块的位置编码。
### D. 多尺度轨迹编码器（Multi-Scale Trajectory Encoder）

为了在受路网约束的轨迹中更好地学习丰富的交通模式和出行语义，我们设计了一个多尺度轨迹编码器（multi-scale trajectory encoder），以显式建模并捕获轨迹中路段之间的关联与依赖。

我们的多尺度轨迹编码器由 $L_{T}$ 层尺度块（scale block）组成。每个块执行三个步骤：将轨迹划分为片段、编码轨迹片段内部的关联（块内）、编码轨迹片段之间的关联（跨块）。上层尺度块将下层块的输出作为输入，以进一步学习轨迹中更大尺度的出行模式。通过堆叠多个尺度块，我们实现了多尺度时空信息的捕获，从而获得更真实的轨迹表示，服务于下游任务。

接下来，我们给出每个尺度块的前向传播过程。

1) 轨迹分块（Trajectory Patch）：受 PatchTST [46] 提出的时间序列划分方法启发，我们将轨迹划分为不重叠的补丁（patch）。窗口大小记为 $\omega$，则对于输入第一层尺度块的轨迹 $\mathcal{T} = <r_1, r_2, \ldots, r_n>$，分块操作生成一个补丁序列：

$$
<s_{1},s_{2},\ldots,s_{N}>,
$$

其中 $s_i$ 是一个补丁，表示窗口内轨迹中的路段序列，且 $N = \lceil \frac{n}{\omega} \rceil$。对于长度不足 $n$ 的轨迹，我们在将其输入多尺度轨迹编码器之前用填充标记（padding marker）进行填充，以确保它们具有相同的长度。

2) 块内编码器（InPatch Encoder）：提出块内编码器用于捕获补丁内路段之间的相关性。鉴于一个补丁是轨迹中长度为窗口大小的路段序列，我们选择计算高效的 GRU [47] 作为编码器。

设 $\mathbf{X}_i \in \mathbb{R}^{\omega \times d}$ 为第 $i$ 个补丁 $s_i$ 的初始嵌入矩阵，它是由补丁内 $\omega$ 个路段的表示向量组成的矩阵。GRU 的正式公式如下：

$$
\mathbf{H}_{i}=\mathbf{G R U}(\mathbf{X}_{i},\mathbf{H}_{0}),
$$

其中 $\mathbf{H}_0$ 是随机初始化的初始隐藏状态嵌入矩阵，$\mathbf{H}_i$ 是 GRU 的输出。每个补丁的嵌入向量 $e_i \in \mathbb{R}^d$ 通过均值池化（mean pooling）层获得：

$$
e_{i}=MeanPool(\mathbf{H}_{i}).
$$

通过块内编码器的设计，我们不仅能够捕获补丁内路段之间的出行模式，还能学习每个补丁的整体嵌入。这使我们能够将每个补丁视为一个完整的实体，从而有效捕获补丁之间更细粒度的时空语义。

3) 跨块编码器（CrossPatch Encoder）：我们使用一个普通 Transformer 编码器作为跨块编码器，以捕获轨迹片段（即补丁）之间的相关性。现在，一条轨迹被表示为多个补丁组成的有序序列 $<e_1, e_2, \ldots, e_N>$。设 $\mathbf{E} \in \mathbb{R}^{N \times d}$ 表示这些补丁的嵌入矩阵。

然后将这些补丁输入到多头 Transformer 模块中，以捕获补丁之间的相关性：

$$
\mathbf{Z}=M u l t i H e a d T r a n s f o r m e r(\mathbf{E}),
$$

其中 $\mathbf{Z} \in \mathbb{R}^{N \times d}$ 是补丁的缩放嵌入，将被馈送到下一层尺度块。

借助补丁，输入序列的长度可以从 $n$ 减少到 $\lceil \frac{n}{\omega} \rceil$。这大大降低了注意力机制的计算复杂度和内存占用。同时，受训练时间和 GPU 内存的约束，这也使得我们的 TrajRL 能够为更长的轨迹学习表示。

在堆叠 $L_{\mathcal{T}}$ 层尺度块之后，我们获得所有块的输出 $\{\mathbf{Z}^{(1)}, \mathbf{Z}^{(2)}, \ldots, \mathbf{Z}^{(L\tau)}\}$，其中 $\mathbf{Z}^{(i)}$ 是第 $i$ 个块层的输出。请注意，不同尺度块输出的嵌入矩阵中的行是不同的。这是因为块内编码器不仅学习补丁表示，还起到了合并相邻补丁表示的作用。通过这种合并机制，我们可以从较小尺度的表示中不断学习到更大尺度的表示。为了提取整条轨迹的整体表示 $\mathbf{z} \in \mathbb{R}^d$，我们首先对缩放后的补丁嵌入进行均值池化，以获得每个尺度下的轨迹嵌入。第 $l$ 个尺度的轨迹嵌入为：

$$
z^{(l)}={M e a n P o o l}(\mathbf{Z}^{(l)}).
$$

然后，通过自适应调整权重，将这些不同尺度的嵌入在多尺度轨迹编码器中融合：

$$
z=\sum_{l=0}^{L_{\mathcal{T}}}\beta^{(l)}\mathbf{z}^{(l)},\qquad\sum_{l=0}^{L_{\mathcal{T}}}\beta^{(l)}=1
$$

其中 $\beta^{(l)}$ 是可学习的权重系数。

### E. 自监督预训练（Self-supervised Pre-training）

我们的工作旨在学习能够服务于多个下游任务的通用轨迹表示。因此，我们需要利用轨迹自身的自监督信号而非任何监督信号来预训练我们的 TrajRL。为了促使我们的 TrajRL 捕获多尺度时空特征，我们同时采用两个从不同尺度提供自监督信号的自监督任务。

1) 块级轨迹恢复（Patch-level Trajectory Recovery）：轨迹恢复（trajectory recovery）是一个类似于 MLM（掩码语言模型）[42] 的自监督任务。在本工作中，轨迹被表示为连续路段的序列，因此如果轨迹中的某些路段被掩蔽，模型可以轻易推断出这些路段，因为这些路段之间的连接关系隐含在路网中。在所提出的多尺度轨迹编码器中，分块操作同样可以避免这一问题。具体而言，我们使用被掩蔽的补丁（masked patch）而不是被掩蔽的路段。在将轨迹输入多尺度轨迹编码器之前，我们以窗口大小 $\omega$ 对轨迹执行分块和掩蔽操作。请注意，我们使用独立同分布的均匀分布来生成每个补丁的掩蔽概率，并确保被掩蔽的补丁不连续。为了便于评估恢复轨迹的准确性，我们添加了一个窗口大小为 1 的尺度块，以获得路段尺度上的轨迹嵌入。其他尺度块中的窗口大小仍为 $\omega$。

在获得带有掩蔽标记的轨迹 $\mathcal{T}$ 的嵌入 $\mathbf{Z}^{(1)} \in \mathbb{R}^{|\mathcal{V}| \times d}$ 之后，我们使用一个线性分类器来预测被掩蔽补丁中的路段：

$$
\hat{\mathbf{Z}}^{(1)}=\mathbf{Z}^{(1)}\mathbf{W}_{m}+b_{m},
$$

其中 $\mathbf{W}_m \in \mathbb{R}^{d \times |\mathcal{V}|}$ 和 $b_m \in \mathbb{R}^{|\mathcal{V}|}$ 是可学习参数。

最后，我们使用轨迹 $\mathcal{T}$ 中被掩蔽路段与预测值之间的交叉熵损失作为损失函数：

$$
\mathcal{L}_{\mathcal{T}}^{\mathrm{m a s k}}=-\frac{1}{|\mathcal{M}|}\sum_{i=1}^{|\mathcal{M}|}\log\frac{\exp(\hat{\mathbf{Z}}_{i}^{(1)})}{\sum_{j=1}^{|\mathcal{V}|}\exp(\hat{\mathbf{Z}}_{j}^{(1)})},
$$

其中 $\mathcal{M}$ 是被掩蔽补丁中路段的集合。对于一批轨迹，我们对所有轨迹的损失取平均，作为损失 $\mathcal{L}^{mask}$。

2) 轨迹对比学习（Trajectory Contrastive Learning）：轨迹恢复致力于从上下文信息中挖掘轨迹片段之间的相关性，因此它提供的是局部自监督信号。为了提高模型学习全局轨迹表示的能力，我们还引入了从全局视角提供自监督信号的对比学习（contrastive learning）。

路段偏移（Road segment shifting）：给定一条轨迹 $\mathcal{T}$，路段偏移随机改变轨迹内的路段或相应的时间戳。具体而言，对于随机选中的路段，路段偏移将其随机改变为路网中该路段的一跳或两跳邻居之一（不包括轨迹中与它前后相邻的路段）。时间偏移（temporal shifting）则对相应的时间戳施加随机扰动。

- 轨迹截断（Trajectory truncating）：给定一条轨迹 $\mathcal{T}$，轨迹截断按照给定的比例从 $\mathcal{T}$ 中截取一个前缀或后缀，并将剩余部分作为新的视图 $\tilde{\mathcal{T}}$。

轨迹增强（Trajectory Augmentation）。数据增强为输入数据创建不同的视图，使模型能够从这些视图中学习共同特征和区分性特征。[28] 探索了基于 GPS 轨迹的数据增强方法，但这些方法不适用于受路网约束的轨迹数据。在本工作中，我们进一步扩展了面向受路网约束轨迹的现有轨迹增强方法。具体而言，基于先前工作 [28]，我们选择了两种有效的轨迹增强方法来生成合适的轨迹视图。

我们使用 InfoNCE [48] 损失进行模型训练。同一条轨迹的两个不同视图构成一个正样本对，来自不同轨迹的任意两个视图构成一个负样本对。正样本对 $(i,j)$ 的对比训练损失定义如下：

$$
\begin{array}{r l r}&{}&{\mathcal{L}_{i,j}^{c l}=-\log\frac{\exp(\mathrm{S i m}(\mathbf{z}_{i},\mathbf{z}_{j})/\tau)}{\sum_{\lceil k=1\rceil\wedge\lceil k\neq i\rceil}^{2N_{b}}\exp(\mathrm{S i m}(\mathbf{z}_{i},\mathbf{z}_{k})/\tau)},}\\ &{}&{\mathrm{S i m}(\mathbf{z}_{i},\mathbf{z}_{j})=\frac{\mathbf{z}_{i}\cdot\mathbf{z}_{j}}{\|\mathbf{z}_{i}\|\|\mathbf{z}_{j}\|},}\end{array}
$$

其中 $N_b$ 是批量大小，$\tau$ 是温度超参数。我们对所有 $2N_b$ 条轨迹的损失取平均，作为损失 $\mathcal{L}^{cl}$。

我们使用上述两个自监督任务预训练我们的 TrajRL。预训练损失是轨迹恢复损失与对比损失之和：

$$
\mathcal{L}^{p r e}=\gamma\mathcal{L}^{m a s k}+(1-\gamma)\mathcal{L}^{c l},
$$

其中 $\gamma$ 是用于平衡两个自监督任务的权重超参数。

### F. 时间复杂度分析（Time Complexity Analysis）

TrajRL 的时间复杂度由尺度块模块中的自注意力计算决定。第 $i$ 个尺度块的时间复杂度为 $O\left(L_{H}d\sum_{i=1}^{l}\left(\frac{n}{\omega^{i-1}}\right)^{2}\right)$，其中 $L_{H}$ 是每个多头 Transformer 编码器的层数，$n$ 是轨迹的长度，$\omega$ 是多尺度轨迹编码器中的窗口大小。由于块内编码器会合并窗口内的补丁，尺度块的层数 $L_{\mathcal{T}}$ 为 $[\log_{\omega} n+1]$。这里，为简洁起见，我们忽略取整操作。通过将所有尺度块的时间复杂度求和，我们可以推导出 TrajRL 的时间复杂度为 $O\left(\sum_{i=1}^{L_{\tau}}L_{H}dn^{2}\left(\frac{1}{\omega^{2}}\right)^{i}\right)$，即 $O\left(L_{H}dn^{2}\cdot\frac{1-\frac{1}{n^{2}\omega^{2}}}{1-\frac{1}{\omega^{2}}}\right)$，近似为 $O\left(\frac{L_{H}dn^{2}}{1-\frac{1}{\omega^{2}}}\right)$。因此，它与 Transformer 编码器的时间复杂度相当。

### G. 模型微调（Model Fine-tuning）

我们使用最简单的方法将 TrajRL 微调（fine-tune）到下游任务。

1) 轨迹分类（Trajectory Classification）：我们将轨迹表示输入到一个简单的线性分类器中，该分类器由一个线性层和随后的 softmax 激活函数组成。预测值为 $\hat{y}_i = \text{softmax}(FC(\mathbf{z}_i)) \in \mathbb{R}^C$。然后我们使用交叉熵损失微调我们的 TrajRL：

$$
\mathcal{L}^{c l a}=-\frac{1}{N_{t e s t}}\sum_{i=1}^{N_{t e s t}}\sum_{c=1}^{C}y_{i,c}\log(\hat{y}_{i,c}),
$$

其中 $y_{i,c}$ 是第 $i$ 条轨迹的真实标签，$N_{test}$ 是测试集中轨迹的总数，$C$ 是类别数。

2) 轨迹出行时间估计（Trajectory Travel Time Estimation）：类似地，我们将轨迹表示输入到简单的线性回归模型中，得到预测值 $\hat{y}_i = FC(\mathbf{z}_i) \in \mathbb{R}$。然后我们使用均方误差损失微调我们的 TrajRL：

$$
\mathcal{L}^{t t e}=\frac{1}{N_{t e s t}}\sum_{i=1}^{N_{t e s t}}\|y_{i}-\hat{y}_{i}\|^{2},
$$

其中 $y_{i}$ 是第 $i$ 条轨迹的真实值，$N_{test}$ 是测试集中轨迹的总数。

3) 相似轨迹搜索（Similar trajectory search）：给定一条查询轨迹，该任务是找出数据库中与之最相似的轨迹。这里不需要微调，我们直接使用从 TrajRL 获得的表示向量 $z_i$ 进行相似度计算。具体细节请参阅第 V-C2 节。

### V. 实验（EXPERIMENTS）

在本节中，我们进行了大量实验来评估我们的 TrajRL 在轨迹表示学习方面的性能。实验评估包括五个部分：

- **性能对比（Performance Comparison）**。我们在两个真实数据集上，针对三个下游任务，将我们的 TrajRL 与最先进的方法进行性能比较。

- **消融研究（Ablation Study）**。我们设计了 TrajRL 的变体，并将其性能与完整模型进行比较，以验证每个设计模块的有效性。

- **预训练效果研究（Pre-training Effect Study）**。我们研究自监督预训练对模型性能和收敛速度的影响。

- **参数敏感性分析（Parameter Sensitivity Analysis）**。我们评估不同参数设置对我们 TrajRL 性能的影响，以验证其稳定性。

- **效率对比（Efficiency Comparison）**。我们将 TrajRL 的推理时间与所有基线方法进行比较，以证明 TrajRL 在效率方面的优越性。

*表II：预处理后两个数据集的统计数据。*

| 数据集 | Porto | BJ |
|---|---|---|
| 时间跨度 | 07/01/2013-07/01/2014 | 11/01/2015-11/30/2015 |
| 轨迹数量 | 695,085 | 1,018,312 |
| 用户数量 | 435 | 1,677 |
| 路段数量 | 10,903 | 38,479 |

### A. 数据集（Datasets）

我们在两个真实世界的大规模轨迹数据集 Porto 和 BJ 上进行实验。Porto $^{1}$ 是一个开源数据集，包含 2013 年 7 月至 2014 年 6 月期间葡萄牙波尔图的 170 万条出租车轨迹。BJ 是 [21] 最新发布的数据集，收集了 2015 年 11 月北京超过 100 万条出租车轨迹。我们从 OpenStreetMap $^{2}$ 下载 Porto 的地图数据来构建路网。OpenStreetMap 数据包含城市中的全部路段、所有路段之间的连接关系以及路段的特征。对于 BJ 数据集，作者已将轨迹映射到路网，可直接使用。

我们按时间顺序将数据集划分为训练集、验证集和测试集。对于 Porto，我们按月以 7:2:1 的比例划分每月的数据，并将各月数据合并构成训练集、验证集和测试集。对于 BJ，这三个集合分别覆盖 17/6/7 天的数据，最终数据比例约为 7:1.5:1.5。此外，我们将最大轨迹长度设置为 128 个路段。预处理后两个数据集的统计数据总结于表II。

### B. 对比方法（Compared Methods）

为了评估我们 TrajRL 的优越性，我们与九种最先进的预训练轨迹表示学习方法进行了比较。我们选择了为多个下游任务设计的代表性通用模型。此外，我们还选择了若干为特定重要下游任务设计的专用模型，这些模型可以轻松扩展到本文的实验设置中。

- Traj2vec [18] 通过捕获时空不变特征将每条轨迹转换为特征序列，并使用序列到序列自编码器学习轨迹嵌入。

- T2vec [14] 基于去噪自编码器预训练一个 RNN 轨迹编码器，旨在从稀疏轨迹中恢复细粒度轨迹。

- Trembr [22] 是一个 RNN 自编码器，通过路段恢复和时间戳恢复两项任务学习轨迹嵌入。

- PIM [20] 使用 node2vec 生成静态路网的路段表示。它遵循课程学习（curriculum learning）的原则生成负样本，并使用互信息最大化以无监督方式优化 RNN 编码器。

- Toast [19] 采用具有交通上下文感知能力的 node2vec 学习路段表示，然后通过基于 MLM 的任务和判别任务训练 Transformer 编码器以生成轨迹嵌入。

- JCLRNT [30] 分别开发了一个图编码器和一个轨迹编码器来建模路段和轨迹的表示。这些表示被组织起来，通过三个对比任务训练模型。

- LightPath [32] 利用基于 Transformer 的自编码器构建稀疏路径编码器。具体而言，它采用关系推理和知识蒸馏框架来增强模型的可扩展性。

- TrajCL [28] 是轨迹相似性计算的最先进模型。它使用对比学习来预训练所提出的基于空间与结构双特征自注意力的轨迹编码器。

- START [21] 提出了一种时间感知的 Transformer 编码器，并通过路段之间的转移关系优化 GAT（图注意力网络）。该模型由对比学习和 MLM 任务的损失函数联合训练。

### C. 实验设置（Experimental Settings）

1) 实现细节：我们使用 PyTorch 1.12.1 和 Liberty [49] 框架实现我们的 TrajRL，后者是交通预测领域一个全面且可扩展的库。我们在配备 Ubuntu 20.04 和 NVIDIA GeForce 4090 GPU 的机器上运行实验。我们将嵌入维度 $d$ 设置为 256。对于路段编码器（road segment encoder），我们将层数 $L$ 设置为 3，注意力头数设置为 [8,16,1]。对于多面时间编码器（multifaceted time encoder），我们将尺度系数个数 $\lambda$ 设置为 3，其中 Porto 的尺度系数为 $\{10, 30, 60\}$，BJ 的为 $\{15, 60, 150\}$。对于多尺度轨迹编码器，我们将每个 Transformer 编码器中多头注意力模块的层数 $L_H$ 设置为 3，注意力头数为 16，窗口大小 $\omega$ 为 4，尺度块数量 $L_\mathcal{T}$ 为 4。InfoNCE 损失中的温度参数 $\tau$ 为 0.05。用于平衡损失的系数 $\gamma$ 为 0.5。我们使用优化器 Adam [50] 对 TrajRL 进行预训练和微调。批量大小为 64，训练轮数为 30。所有基线方法保持与 TrajRL 相同的嵌入维度，其他设置遵循默认或推荐配置。来自原始论文。此外，所有方法都以一致的方式针对下游任务进行微调，如第 IV-G 节所述。由于 Trembr 和 Toast 未开源，我们参考 [21] 中的实验设置和结果以进行公平比较。

*表III：所有模型在 Porto 数据集上的性能对比。*

| 方法 | 轨迹分类 |  |  | 出行时间估计 |  |  | 最相似轨迹搜索 |  |  |
|---|---|---|---|---|---|---|---|---|---|
| 方法 | Micro-F1 $\uparrow$ | Macro-F1 $\uparrow$ | Recall@5 $\uparrow$ | MAE $\downarrow$ | MAPE $\downarrow$ | RMSE $\downarrow$ | MR $\downarrow$ | HR@1 $\uparrow$ | HR@5 $\uparrow$ |
| traj2vec [IJCNN-17] | 0.066 | 0.041 | 0.181 | 1.549 | 0.241 | 2.346 | 30.69 | 0.543 | 0.732 |
| t2vec [ICDE-18] | 0.064 | 0.053 | 0.189 | 1.521 | 0.237 | 2.324 | 11.54 | 0.769 | 0.858 |
| Trembr [TIST-20] | 0.071 | 0.049 | 0.192 | 1.480 | 0.226 | 2.164 | 4.635 | 0.846 | 0.929 |
| PIM [IJCAL-21] | 0.059 | 0.037 | 0.158 | 1.568 | 0.247 | 2.642 | 20.75 | 0.617 | 0.764 |
| Toast [CIKM-21] | 0.062 | 0.035 | 0.184 | 1.624 | 0.246 | 2.445 | 22.61 | 0.684 | 0.789 |
| JCLRNT [CIKM-22] | 0.069 | 0.046 | 0.191 | 1.647 | 0.238 | 2.334 | 5.017 | 0.852 | 0.931 |
| LightPath [KDD-23] | 0.068 | 0.049 | 0.180 | 1.561 | 0.242 | 2.362 | 13.26 | 0.607 | 0.788 |
| TrajCL [ICDE-23] | 0.056 | 0.033 | 0.148 | 1.960 | 0.269 | 2.963 | 5.461 | 0.834 | 0.915 |
| START [ICDE-23] | 0.084 | 0.061 | 0.238 | 1.374 | 0.219 | 2.185 | 1.927 | 0.921 | 0.974 |
| TrajRL | 0.092 | 0.070 | 0.249 | 1.259 | 0.202 | 2.051 | 1.241 | 0.988 | 0.997 |
| 提升幅度 | 9.52% | 14.75% | 4.62% | 8.36% | 7.76% | 6.13% | 35.6% | 8.36% | 2.36% |

带 "↑" 的指标表示结果越大越好，带 "↓" 的指标表示结果越小越好。

加粗结果为最优，$\underline{\text{underlined}}$（下划线）结果为次优。出行时间估计中的时间单位为分钟。

*表IV：所有模型在 BJ 数据集上的性能对比。*

| 方法 | 轨迹分类 |  |  | 出行时间估计 |  |  | 最相似轨迹搜索 |  |  |
|---|---|---|---|---|---|---|---|---|---|
| 方法 | ACC $\uparrow$ | F1 $\uparrow$ | AUC $\uparrow$ | MAE $\downarrow$ | MAPE $\downarrow$ | RMSE $\downarrow$ | MR $\downarrow$ | HR@1 $\uparrow$ | HR@5 $\uparrow$ |
| traj2vec [IJCNN-17] | 0.802 | 0.841 | 0.869 | 10.14 | 0.399 | 57.28 | 10.26 | 0.573 | 0.758 |
| t2vec [ICDE-18] | 0.799 | 0.859 | 0.861 | 10.27 | 0.372 | 56.15 | 9.684 | 0.763 | 0.905 |
| Trembr [TIST-20] | 0.818 | 0.871 | 0.880 | 9.997 | 0.342 | 36.97 | 2.509 | 0.884 | 0.952 |
| PIM [IJCAL-21] | 0.803 | 0.874 | 0.866 | 11.56 | 0.486 | 35.78 | 26.75 | 0.731 | 0.874 |
| Toast [CIKM-21] | 0.810 | 0.870 | 0.871 | 10.69 | 0.350 | 57.41 | 29.53 | 0.611 | 0.746 |
| JCLRNT [CIKM-22] | 0.825 | 0.867 | 0.883 | 10.73 | 0.406 | 57.84 | 6.382 | 0.802 | 0.927 |
| LightPath [KDD-23] | 0.802 | 0.859 | 0.865 | 10.66 | 0.506 | 55.80 | 6.484 | 0.748 | 0.867 |
| TrajCL [ICDE-23] | 0.764 | 0.722 | 0.730 | 10.99 | 0.523 | 59.97 | 5.524 | 0.827 | 0.948 |
| START [ICDE-23] | 0.842 | 0.881 | 0.904 | 9.134 | 0.336 | 35.31 | 3.081 | 0.969 | 0.993 |
| TrajRL | 0.867 | 0.914 | 0.939 | 8.807 | 0.298 | 33.15 | 2.140 | 0.982 | 0.998 |
| 提升幅度 | 2.96% | 3.74% | 3.87% | 3.58% | 11.31% | 6.11% | 14.7% | 1.34% | 0.53% |

带 "$\uparrow$" 的指标表示结果越大越好，带 "$\downarrow$" 的指标表示结果越小越好。

加粗结果为最优，$\underline{\text{underlined}}$（下划线）结果为次优。出行时间估计中的时间单位为分钟。

2) 下游任务设置：在第 IV-G 节中，我们说明了我们的 TrajRL 如何针对三个挖掘任务进行微调：轨迹分类、出行时间估计和相似轨迹搜索。这三个下游任务代表了直接利用学习到的表示的典型分类、回归和查询任务。轨迹分类是一个典型的分类任务，代表了若干重要任务，如轨迹用户关联、判断出租车是否载客以及识别交通方式。出行时间估计是一个典型的回归任务，在导航软件中有着重要应用。最相似轨迹查询是轨迹相似性学习中的一个重要下游任务，也是构建时空数据库的基础。此外，它们对轨迹的时间方面和空间方面具有不同的敏感性，从而能够更全面地评估所学轨迹表示的质量。实验的具体细节如下：

- 轨迹分类（CLA）：对于 Porto 数据集，使用司机 ID 作为多分类任务的标签。对于 BJ 数据集，使用出租车是否载客作为二分类任务的标签。我们的 TrajRL 和所有基线方法均使用损失函数（式 (20)）进行微调。

- 出行时间估计（TTE）：为防止时间泄漏，我们确保在微调期间不向模型输入任何时间信息，出发时间（轨迹中第一个路段的时间戳）除外。我们的 TrajRL 和所有基线方法均使用损失函数（式 (21)）进行微调。

- 最相似轨迹搜索（STS）：轨迹之间的相似性度量是一个基础研究问题，也是许多基于轨迹的应用的基础。基于学习的轨迹相似性度量方法通常使用最相似轨迹搜索任务来评估模型的有效性。然而，缺乏真实标签使得评估轨迹相似性的准确性变得困难。参照先前的工作 [21] [43]，我们使用绕路（detour）方法 [19] 生成真实标签。具体实现请参阅 [19]。

3) 评估指标：对于轨迹分类任务，我们使用准确率（ACC）、F1 分数（F1）和 ROC 曲线下面积（AUC）来评估二分类任务，使用 Micro-F1、Macro-F1 和 Recall@5 来评估多分类任务。对于出行时间估计任务，我们使用平均绝对误差（MAE）、平均绝对百分比误差（MAPE）和均方根误差（RMSE）来评估回归结果。对于最相似轨迹搜索任务，我们使用平均排名（MR）和命中率（HR@1、HR@5）。
### D. 性能对比

表III和表IV全面比较了不同轨迹表示学习方法在三个任务上的性能。基于上述分析，我们认为学习到的轨迹表示（trajectory representation）质量主要与模型两方面的能力相关：时空建模（spatio-temporal modeling）能力和自监督学习（self-supervised learning）能力。

1) 时空建模能力：Traj2vec、T2vec 和 TrajCL 旨在基于 GPS 坐标对轨迹的时空特征进行建模，但没有考虑底层路网中的语义信息。实验结果表明，它们在涉及路段或出行语义的任务上表现不佳，例如 CLA（轨迹分类）任务，该任务高度依赖轨迹所经过的路段，而不仅仅是轨迹的形状。此外，尽管 TrajCL 是轨迹相似度测量的最先进模型，但由于缺乏时间信息编码，它在出行时间估计任务上的性能明显较差。

PIM、Trembr、Toast、JCLRNT、LightPath 和 START 通过将轨迹映射到路网，并使用轨迹编码器将受路网约束的轨迹编码为嵌入向量，从而融入了语义信息。实验结果表明，总体而言，它们在下游任务上的表现优于使用基于 GPS 轨迹的方法。Trembr 进一步将轨迹点的时间戳作为输入特征融入时间信息，显著提升了在出行时间估计任务上的性能。Toast 采用增强的路段表示学习方法，融入了更丰富的语义信息。JCLRNT 使用图神经网络从路网的拓扑结构中学习路段信息，并使用序列模型获得轨迹表示。然而，它对路网特征的融入不足，且缺乏对轨迹时间信息的建模。LightPath 仍然依赖 node2vec [51] 获取路段嵌入，这影响了它在下游任务上的表现。START 同时编码路段的静态属性以及时间周期性，使其具备明显更强的时空建模能力，与以往方法相比取得了实验上的提升。

相比之下，TrajRL 中的 multi-faceted time encoder（多面时间编码器）和 multi-scale trajectory encoder（多尺度轨迹编码器）进一步增强了时空建模能力，使其在多个下游任务上取得了最佳性能。

2) 自监督学习能力：Traj2vec、T2vec、Trembr 和 LightPath 都采用自编码（auto-encoding）或自回归（auto-regressive）框架，主要侧重于恢复轨迹的详细原始特征。此外，LightPath 引入了因果学习（causal learning）来增强训练效果。实验结果表明，它们在路段级下游任务上表现良好。PIM、TrajCL 和 JCLRMT 使用对比学习（contrastive learning）进行预训练。对比学习通过从轨迹的不同视角进行学习，增强了模型在序列级任务上的性能。Toast 使用 MLM（掩码语言建模，Masked Language Modeling）作为自监督任务，帮助模型更好地理解轨迹中路段之间的相关性。我们的 TrajRL 和 START 同时采用轨迹恢复和对比学习进行自监督训练，使模型在路段级和序列级同时得到优化。此外，由于这两个自监督任务提供不同尺度的信号，我们的多尺度轨迹编码器能够更好地利用预训练的优势，这也增强了我们模型在下游任务上的性能。这同样证明了多尺度轨迹建模的必要性。

上述实验结果和分析表明，我们的 TrajRL 是一种有效的 TRL（轨迹表示学习，trajectory representation learning）方法。它具备更强的时空建模和自监督学习能力，在两个数据集和三个下游任务上取得了最佳性能，与最先进模型相比有显著提升。

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0010-10.jpg)
*(a) Porto 数据集*

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0010-12.jpg)

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0010-13.jpg)

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0010-14.jpg)
*(b) BJ 数据集*

*图2：消融实验结果。*

### E. 消融实验

为了评估 TrajRL 中每个模块的有效性，我们比较了以下变体与完整 TrajRL 的性能：（1）w/o MFTime-Enc（去掉多面时间编码器）：移除整个多面时间编码器（Multi-faceted Time Encoder）。（2）w/o Time Interval（去掉时间间隔嵌入）：移除时间间隔嵌入。（3）w/o MSTraj-Enc（去掉多尺度轨迹编码器）：将整个多尺度轨迹编码器（Multi-scale Trajectory Encoder）替换为 Transformer 编码器。（4）w/o InPatch-Enc（去掉InPatch编码器）：将 InPatch 编码器替换为线性层。（5）w/o TrajRec（去掉轨迹恢复）：移除轨迹恢复自监督任务。（6）w/o CL（去掉对比学习）：移除对比学习自监督任务。

此外，我们将在下一节研究预训练的有效性，其中涉及一个移除所有自监督任务的变体。

1) 模型组件的影响：图2中的实验结果表明了每个模块在两个数据集上的有效性。首先，w/o MFTime-Enc 显著降低了性能，这证明了时间编码在 TRL 中的必要性。此外，w/o Time Interval 与 TrajRL 的对比表明，多尺度时间间隔编码有利于多个下游任务，尤其对出行时间估计有显著影响。因此，多尺度时间间隔编码对丰富时间编码非常有意义。另外，得益于多尺度特征学习能力，TrajRL 的性能显著优于采用 Transformer 编码器的 w/o MSTraj-Enc。这一优势在 STS（最相似轨迹搜索）任务中更为突出。TrajRL 使用 GRU（门控循环单元，Gated Recurrent Unit）聚合一个 patch（块）内子轨迹段的信息，优于使用线性层的 w/o InPatch-Enc。因此，我们认为进一步探索更适合 InPatch 编码器的短序列编码器有助于提升模型性能。

2) 自监督任务的影响：结果表明，两个自监督任务总体上对模型性能都有显著影响。具体而言，w/o TrajRec 和 w/o CL 在 CLA 和 STS 上的性能下降幅度比在 TTE（出行时间估计）上更大。这是因为轨迹恢复和对比学习提供了大量的空间自监督信号，而时间自监督信息较少。此外，w/o CL 在 CLA 任务上性能下降最大，而 w/o TrajRec 在 STS 任务上性能下降最大。考虑到这两个任务的不同尺度，这进一步验证了通过不同的自监督任务、以不同尺度为 TRL 提供自监督信号的必要性。

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0011-05.jpg)
*(a) 平均损失*

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0011-07.jpg)
*(b) 验证准确率*

*图3：预训练+微调 vs. 端到端训练。*

### F. 预训练的有效性

为了探究模型的预训练效果，我们将完整的 TrajRL 与一个仅在训练方式上不同的变体进行比较。端到端（end-to-end）训练方法随机初始化模型参数，并直接使用下游任务的损失函数对模型进行监督训练。此外，我们还研究了模型跨数据集的可迁移性：我们在一个数据集上预训练模型，然后在另一个数据集上进行微调。

*表V：跨两个数据集的模型可迁移性。*

| 数据集 | 模型 | 轨迹分类 |  | 出行时间估计 |  |
|---|---|---|---|---|---|
| 数据集 | 模型 | Micro-F1 $\uparrow$ | Macro-F1 $\uparrow$ | MAE $\downarrow$ | RMSE $\downarrow$ |
| Porto | START | 0.084 | 0.061 | 1.374 | 2.185 |
| Porto | TrajRL | 0.092 | 0.070 | 1.259 | 2.051 |
| BJ $\rightarrow$ Porto | START | 0.076 | 0.055 | 1.444 | 2.385 |
| BJ $\rightarrow$ Porto | TrajRL | 0.081 | 0.057 | 1.346 | 2.153 |
|  |  | ACC $\uparrow$ | F1 $\uparrow$ | MAE $\downarrow$ | RMSE $\downarrow$ |
| BJ | START | 0.842 | 0.881 | 9.134 | 35.31 |
| BJ | TrajRL | 0.867 | 0.914 | 8.807 | 33.15 |
| Porto $\rightarrow$BJ | START | 0.807 | 0.824 | 11.29 | 42.21 |
| Porto $\rightarrow$BJ | TrajRL | 0.826 | 0.827 | 10.71 | 40.98 |

图3报告了 CLA 任务在 Porto 数据集上的平均损失值和验证准确率。结果表明，预训练和微调能够显著提高模型的收敛速度。此外，由于预训练任务中融入了先验知识，预训练还增强了模型在下游任务上的泛化能力。

在表V中，我们报告了 TrajRL 和 START 的跨数据集迁移性能。实验结果表明，TrajRL 达到了直接在目标数据集上训练的模型约 80% 的性能。与最先进的模型 START 相比，TrajRL 也表现出更强的跨数据集迁移能力。尽管不同城市的道路网络具有异质性，我们的 TrajRL 仍能捕捉共同的出行模式。然而，捕捉城市特有的驾驶习惯仍然具有挑战性，这限制了轨迹表示在不同数据集之间的可迁移性。结果还表明，TrajRL 表现出良好的鲁棒性，这主要归功于恰当的预训练方法。

### G. 超参数的影响

我们进一步对关键超参数进行了参数敏感性分析，即嵌入维度 $d$、批大小 $N_b$、InPatch 编码器中的窗口大小 $\omega$，以及时间间隔编码器中的尺度系数个数 $\lambda$。我们在图4中报告了 Porto 数据集上 CLA 和 TTE 任务的结果，其中 CLA 以 Micro-F1（微平均F1）为指标、TTE 以 MAE（平均绝对误差）为指标，BJ 数据集上的结果类似。

从图4(a)可以看出，增大嵌入维度 $d$ 起初会提升模型性能，但当其过大时，由于学习到的轨迹嵌入过于稀疏或模型过拟合，性能会下降。

较大的批大小更有可能帮助模型收敛，但如果批大小过大，会对对比学习的效果产生负面影响，导致模型性能下降。

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0012-00.jpg)
*(a) 嵌入维度 $d$*

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0012-02.jpg)
*(b) 批大小 $N_{b}$*

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0012-05.jpg)
*(c) 窗口大小 $\omega$*

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0012-07.jpg)
*(d) 尺度系数个数 $\lambda$*

*图4：超参数对 Porto 数据集上 CLA 和 TTE 任务的影响。*

性能。图4(b)中关于批大小 $N_{b}$ 的实验结果验证了这一理论分析。

窗口大小决定了多尺度轨迹编码器中每一层的尺度以及尺度表示的数量。窗口大小过小会阻碍模型学习细粒度的轨迹表示，而窗口大小过大会导致模型学习到的尺度表示数量减少。图4(c)中的实验结果表明，当窗口大小 $\omega$ 设为 4 时模型表现最佳。

此外，我们研究了多面时间编码器中尺度系数对模型性能的影响。鉴于我们的 TTE 任务以分钟为单位进行度量，我们在选择尺度系数时优先纳入 60（秒）。因此，我们实验了四种不同的尺度系数组合，即 {60}、{30, 60}、{10, 30, 60} 和 {10, 30, 60, 90}。从图4(d)可以看出，使用更多的尺度系数能够显著提升模型性能，因为尺度系数越多，时间间隔编码的粒度越细。理想情况下，尺度系数越多，模型性能越好。然而，在实际中，轨迹中路段之间的时间间隔不会过大，因此过大的尺度系数不仅无效，还可能引入噪声。此外，出行时间估计任务尤其受到尺度系数选择的影响，这进一步验证了时间间隔编码的必要性。

### H. 效率对比

我们还在两个数据集上将 TrajRL 与所有基线方法的效率进行了比较。由于我们的 TrajRL 是自监督预训练模型，可以方便地进行离线训练，因此我们主要关注将轨迹编码为嵌入向量的时间成本。我们在图5中报告了对 200,000 条轨迹进行编码的时间成本。

总体而言，我们的 TrajRL 的推理速度处于中等水平。更重要的是，除了专门为快速轨迹查询而设计的 TrajCL [28] 之外，我们的 TrajRL 在通用的

![](media/Self-Supervised_Trajectory_Representation_Learning_with_Multi-Scale_Spatio-Temporal_Feature_Exploration.pdf-0012-17.jpg)
*图5：推理时间对比（秒）。*

轨迹表示学习方法中表现出相对先进的推理速度。我们的模型编码 200,000 条轨迹大约需要 50 秒，证明了其对大规模轨迹数据集的可扩展性。与目前最先进的 TRL 方法（即 START）相比，TrajRL 也展现出一些优势。值得注意的是，尽管我们的多尺度轨迹编码器是更深的模型，但第IV-F节中的时间复杂度分析和实验结果都表明，与其他基于自注意力的模型相比，TrajRL 的推理速度并未下降。这主要得益于我们多尺度轨迹编码器的两个关键优势。首先，CrossPatch 编码器中的自注意力计算在 patch（块）上进行而非在点上进行，这显著降低了时间复杂度。其次，由于 InPatch 编码器会在窗口内合并 patch（块），较深层尺度块中的输入序列长度和参数数量相比较浅层呈指数级减少。这两个优势极大地促进了更深层模型的构建和训练。总之，我们的 TrajRL 以更低的时间成本实现了更深的架构，并展现出很强的可扩展性。

### VI. 结论与未来工作

在本工作中，我们提出了 TrajRL，一个新颖的自监督轨迹表示学习框架。TrajRL 通过所设计的多面时间编码器和多尺度轨迹编码器有效地学习通用轨迹嵌入。多面时间编码器同时对时间段和时间间隔进行建模，捕捉轨迹全面的时间特征。多尺度轨迹编码器通过 InPatch 和 CrossPatch 编码器学习轨迹的多尺度时空特征。在两个真实数据集上三个下游任务的实验证明了我们提出的 TrajRL 相对于最先进基线的优越性。未来，我们将结合大语言模型，探索面向大规模多源异构轨迹的统一表示学习模型。

### VII. 致谢

本工作部分受国家自然科学基金项目（批准号：62176243 和 62202438）以及国家重点研发计划项目（批准号：2022ZD0117201）资助。

[21] J. Jiang, D. Pan, H. Ren, X. Jiang, C. Li, and J. Wang, "Self-supervised trajectory representation learning with temporal regularities and travel semantics," in 2023 IEEE 39th international conference on data engineering (ICDE). IEEE, 2023, pp. 843–855.

[22] T.-Y. Fu and W.-C. Lee, "Trembr: Exploring road networks for trajectory representation learning," ACM Transactions on Intelligent Systems and Technology (TIST), vol. 11, no. 1, pp. 1–25, 2020.

[23] D. Yao, H. Hu, L. Du, G. Cong, S. Han, and J. Bi, "Trajgat: A graph-based long-term dependency modeling approach for trajectory similarity computation," in Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, 2022, pp. 2275–2285.

[24] S. Zhou, J. Li, H. Wang, S. Shang, and P. Han, "Gristm: trajectory similarity computation with graph-based residual lsm," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 4, 2023, pp. 4972–4980.

[25] A. Liu, Y. Zhang, X. Zhang, G. Liu, Y. Zhang, Z. Li, L. Zhao, Q. Li, and X. Zhou, "Representation learning with multi-level attention for activity trajectory similarity computation," IEEE Transactions on Knowledge and Data Engineering, vol. 34, no. 5, pp. 2387–2400, 2020.

[26] L. Deng, Y. Zhao, Z. Fu, H. Sun, S. Liu, and K. Zheng, "Efficient trajectory similarity computation with contrastive learning," in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022, pp. 365–374.

[27] Y. Cao, L. Li, X. Chen, X. Xu, Z. Huang, and Y. Yu, "Hypergraph hash learning for efficient trajectory similarity computation," in Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, 2024, pp. 175–186.

[28] Y. Chang, J. Qi, Y. Liang, and E. Tanin, "Contrastive trajectory similarity learning with dual-feature attention," in 2023 IEEE 39th International Conference on Data Engineering (ICDE). IEEE, 2023, pp. 2933–2945.

[29] T. Mikołow, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, "Distributed representations of words and phrases and their compositionality," Advances in neural information processing systems, vol. 26, 2013.

[30] Z. Mao, Z. Li, D. Li, L. Bai, and R. Zhao, "Jointly contrastive representation learning on road network and trajectory," in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022, pp. 1501–1510.

[31] P. Veliković, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, "Graph attention networks," arXiv preprint arXiv:1710.10903, 2017.

[32] S. B. Yang, J. Hu, C. Guo, B. Yang, and C. S. Jensen, "Lightpath: Lightweight and scalable path representation learning," in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 2999–3010.

[33] T. Wei, Y. Lin, Y. Lin, S. Guo, L. Zhang, and H. Wan, "Micro-macro spatial-temporal graph-based encoder-decoder for map-constrained trajectory recovery," IEEE Transactions on Knowledge and Data Engineering, 2024.

[34] C. Feng, Z. Pan, J. Fang, J. Xu, P. Zhao, and L. Zhao, "Aries: Accurate metric-based representation learning for fast top-k trajectory similarity query," in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022, pp. 499–508.

[35] P. Yang, H. Wang, Y. Zhang, L. Qin, W. Zhang, and X. Lin, "T3s: Effective representation learning for trajectory similarity computation," in 2021 IEEE 37th International Conference on Data Engineering (ICDE). IEEE, 2021, pp. 2183–2188.

[36] H. Zhang, X. Zhang, Q. Jiang, B. Zheng, Z. Sun, W. Sun, and C. Wang, "Trajectory similarity learning with auxiliary supervision and optimal matching," in Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, 2021.

[37] S. B. Yang, C. Guo, J. Hu, B. Yang, J. Tang, and C. S. Jensen, "Weakly-supervised temporal path representation learning with contrastive curriculum learning," in 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE, 2022, pp. 2873–2885.

[38] B. Yan, G. Zhao, L. Song, Y. Yu, and J. Dong, "Preclin: Pretrained-based contrastive learning network for vehicle trajectory prediction," World Wide Web, vol. 26, no. 4, pp. 1853–1875, 2023.

[39] K. Luo, Y. Zhu, W. Chen, K. Wang, Z. Zhou, S. Ruan, and Y. Liang, "Towards robust trajectory representations: Isolating environmental confounders with causal learning," arXiv preprint arXiv:2404.14073, 2024.

[40] X. Qiu, T. Sun, Y. Xu, Y. Shao, N. Dai, and X. Huang, "Pre-trained models for natural language processing: A survey," Science China Technological Sciences, Vol. 63, no. 10, pp. 1872–1897, 2020.

[1] L. Chen, Y. Gao, X. Li, C. S. Jensen, and G. Chen, "Efficient metric indexing for similarity search and similarity joins," IEEE Transactions on Knowledge and Data Engineering, vol. 29, no. 3, pp. 556–571, 2017.

[2] X. Ding, L. Chen, Y. Gao, C. S. Jensen, and H. Bao, "Ultraman: A unified platform for big trajectory data management and analytics," Proceedings of the VLDB Endowment, vol. 11, no. 7, pp. 787–799, 2018.

[3] J. Bao, T. He, S. Ruan, Y. Li, and Y. Zheng, "Planning bike lanes based on sharing-bikes' trajectories," in Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, 2017, pp. 1377–1386.

[4] L. Chen, Q. Zhong, X. Xiao, Y. Gao, P. Jin, and C. S. Jensen, "Price-and-time-aware dynamic ridesharing," in 2018 IEEE 34th International conference on data engineering (ICDE). IEEE, 2018, pp. 1061–1072.

[5] W. Chen, S. Li, C. Huang, Y. Yu, Y. Jiang, and J. Dong, "Mutual distillation learning network for trajectory-user linking," arXiv preprint arXiv:2205.03773, 2022.

[6] S. Dai, Y. Yu, H. Fan, and J. Dong, "Spatio-temporal representation learning with social tie for personalized poi recommendation," Data Science and Engineering, vol. 7, no. 1, pp. 44–56, 2022.

[7] T. Feng, H. Yan, H. Wang, W. Huang, Y. Han, H. Liao, J. Hao, and Y. Li, "Iltroute: A graph-based imitation learning method to unveil riders' routing strategies in food delivery service," in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 4024–4034.

[8] Y. Liang, K. Ouyang, Y. Wang, X. Liu, H. Chen, J. Zhang, Y. Zheng, and R. Zimmermann, "Trafformer: Efficient trajectory classification with transformers," in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022, pp. 1229–1237.

[9] Z. Wang, K. Fu, and J. Ye, "Learning to estimate the travel time," in Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, 2018, pp. 858–866.

[10] D. Yao, G. Cong, C. Zhang, and J. Bi, "Computing trajectory similarity in linear time: A generic seed-guided neural metric learning approach," in 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 2019, pp. 1358–1369.

[11] L. Chen, Y. Gao, Z. Fang, X. Miao, C. S. Jensen, and C. Guo, "Real-time distributed co-movement pattern detection on streaming trajectories," Proceedings of the VLDB Endowment, vol. 12, no. 10, pp. 1028–1220, 2019.

[12] L. Alessandretti, "What human mobility data tell us about covid-19 spread," Nature Reviews Physics, vol. 4, no. 1, pp. 12–13, 2022.

[13] H. Liu, H. Wu, W. Sun, and I. Lee, "Spatio-temporal gru for trajectory classification," in 2019 IEEE International Conference on Data Mining (ICDM). IEEE, 2019, pp. 1228–1233.

[14] X. Li, K. Zhao, G. Cong, C. S. Jensen, and W. Wei, "Deep representation learning for trajectory similarity computation," in 2018 IEEE 34th International Conference on Data Engineering (ICDE). IEEE, 2018, pp. 617–628.

[15] A. Ashukha, A. Lyzhov, D. Molchanov, and D. Vetrov, "Pitfalls of domain uncertainty estimation and ensembling in deep learning," arXiv preprint arXiv:2002.06470, 2020.

[16] T. Ishida, I. Yamane, T. Sakai, G. Niu, and M. Sugiyama, "Do we need zero training loss after achieving zero training error?" arXiv preprint arXiv:2002.08709, 2020.

[17] Z. Fang, Y. Du, L. Chen, Y. Hu, Y. Gao, and G. Chen, "E2dtc: An end to end deep trajectory clustering framework via self-training," in 2021 IEEE 37th International Conference on Data Engineering (ICDE). IEEE, 2021, pp. 696–707.

[18] D. Yao, C. Zhang, Z. Zhu, J. Huang, and J. Bi, "Trajectory clustering via deep representation learning," in 2017 International Joint Conference on neural networks (IJCNN). IEEE, 2017, pp. 3880–3887.

[19] Y. Chen, X. Li, G. Cong, Z. Bao, C. Long, Y. Liu, A. K. Chandran, and R. Ellison, "Robust road network representation learning: When traffic patterns meet traveling semantics," in Proceedings of the 30th ACM International Conference on Information & Knowledge Management, 2021, pp. 211–220.

[20] S. B. Yang, C. Guo, J. Hu, J. Tang, and B. Yang, "Unsupervised path representation learning with curriculum negative sampling," in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021, pp. 3286–3292.

[41] J. Yu, H. Yin, X. Xia, T. Chen, J. Li, and Z. Huang, "Self-supervised learning for recommender systems: A survey," IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 1, pp. 335–355, 2023.

[42] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," arXiv preprint arXiv:1810.04805, 2018.

[43] Y. Lin, H. Wan, S. Guo, J. Hu, C. S. Jensen, and Y. Lin, "Pre-training general trajectory embeddings with maximum multi-view entropy coding," IEEE Transactions on Knowledge and Data Engineering, 2023.

[44] C. Yang and G. Gidofalvi, "Fast map matching, an algorithm integrating hidden markov model with precomputation," International Journal of Geographical Information Science, vol. 32, no. 3, pp. 547–570, 2018.

[45] T. Mikolov, K. Chen, C. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," arXiv preprint arXiv:1301.3781, 2013.

[46] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," arXiv preprint arXiv:2211.14730, 2022.

[47] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, "Empirical evaluation of gated recurrent neural networks on sequence modeling," arXiv preprint arXiv:1412.3555, 2014.

[48] F. Wang and H. Liu, "Understanding the behaviour of contrastive loss," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2495–2504.

[49] J. Jiang, C. Han, W. Jiang, W. X. Zhao, and J. Wang, "Libcity: A unified library towards efficient and comprehensive urban spatial-temporal prediction," arXiv preprint arXiv:2304.14343, 2023.

[50] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

[51] A. Grover and J. Leskovec, "node2vec: Scalable feature learning for networks," in Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, 2016, pp. 855–864.