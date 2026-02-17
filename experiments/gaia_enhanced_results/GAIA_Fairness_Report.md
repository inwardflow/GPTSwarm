# GAIA 实验公平性深度分析报告

**作者**: Manus AI
**日期**: 2026年2月17日

## 1. 执行摘要 (Executive Summary)

本次分析旨在回应您提出的核心问题：**在 GAIA 基准测试上，之前的实验是否对不同 Agent 架构（如 AgentGroup）构成了公平的比较？**

**核心结论是：不，之前的比较是严重不公平的。**

我们发现，观察到的 `Enhanced ReAct` Agent (35.2% 准确率) 与 `GPTSwarm` 基线 (14.5% 准确率) 之间的 **+20.7pp** 性能差距，并非单纯由 Agent 架构优越性导致。相反，它是由至少四个混杂因素（Confounding Factors）共同作用的结果：

1.  **工具套件差异 (Tool Suite)**: `Enhanced ReAct` 配备了文件解析、代码执行和图像分析工具，而 `GPTSwarm` 基线完全不具备这些能力。**这直接导致了约 9.7pp 的性能差距**，因为 23% 的 GAIA 问题需要这些高级工具。
2.  **搜索质量差异 (Search Quality)**: `Enhanced ReAct` 使用高质量的 Google 搜索，而 `GPTSwarm` 基线使用质量较低的 DuckDuckGo + Wikipedia API。
3.  **模型差异 (Model)**: `Enhanced ReAct` 使用了稳定且强大的 `gpt-4.1-mini`，而 `GPTSwarm` 基线使用的 `gpt-5.1-codex-mini` 表现不稳定，可能存在问题。
4.  **架构差异 (Architecture)**: 这是我们希望测量的变量（例如，单 Agent vs. 多 Agent），但其效果被前三个因素完全掩盖。

**本报告建议，为了在 AgentGroup 论文中进行科学严谨的比较，所有参与对比的 Agent 必须在相同的模型、相同的工具套件和相同的搜索引擎下运行，唯一的变量应该是其核心架构。**

---

## 2. 背景：一个“不公平”的赛场

您敏锐地指出，GAIA 基准测试 [1] 不仅仅是语言理解的测试，更是对一个 Agent 系统综合能力的考验，这些能力包括网页浏览、文件处理（PDF, Excel, 图片等）和代码执行。然而，在之前的实验中，`GPTSwarm` 仓库提供的基线方法（如 Direct, WebSearch, ToolTOT）仅配备了基础的网页搜索工具，这使得它们在面对需要高级工具的 GAIA 问题时束手无策。

本报告将量化这种“不公平”带来的影响，并为 AgentGroup 论文的实验设计提供清晰、可执行的建议。

## 3. 性能差距分解：+20.7pp 的增益从何而来？

我们将 `Enhanced ReAct` 相对于 `WebSearch` 基线的 **+20.7pp** 整体性能提升分解为两个主要部分：

1.  **“相对公平”的提升 (+10.9pp)**: 在 **不需要** 文件处理的 127 个问题上，`Enhanced ReAct` 的表现优于基线。这部分增益主要归因于 **更好的模型、更优的架构和更高质量的搜索**。
2.  **“工具优势”的提升 (+9.7pp)**: 在 **需要** 文件处理的 38 个问题上，`Enhanced ReAct` 凭借其工具优势取得了巨大领先。这部分增益主要来自 **工具套件的不对等**。

下图直观地展示了这一分解过程：

![图1：GAIA 性能提升分解](fig3_improvement_decomposition.png)

> **图1解读**: 瀑布图清晰地显示，超过一半的性能提升（10.9pp，占总提升的 53%）来自在“无文件问题”上的架构、模型和搜索质量优势，而另一半（9.7pp，占总提升的 47%）则直接来源于基线所不具备的工具能力。

## 4. 深入分析：两种赛道，两种结果

为了更清晰地展示工具不公平性的影响，我们将 GAIA 的 165 个问题分为两个赛道：

*   **“公平”赛道 (n=127)**: 不含文件附件的问题。所有 Agent 至少都有网页搜索能力。
*   **“不公平”赛道 (n=38)**: 包含文件附件的问题。只有 `Enhanced ReAct` 具备处理能力。

![图2：公平 vs 不公平赛道对比](fig2_fair_vs_unfair.png)

> **图2解读**: 
> *   **左图（公平赛道）**: 即使在相对公平的条件下，`Enhanced ReAct` (32.3%) 仍然显著优于最好的基线 `WebSearch` (18.1%)。这 **+14.2pp** 的差距主要体现了架构、模型和搜索质量的综合优势。
> *   **右图（不公平赛道）**: 差距是戏剧性的。`Enhanced ReAct` 达到了 44.7% 的准确率，而所有基线几乎为零。这清晰地证明，**在没有相应工具的情况下，讨论 Agent 架构的优劣是毫无意义的**。

### 4.1. 工具矩阵：能力的全方位对比

下表和图表详细列出了不同 Agent 的工具配置，揭示了能力上的鸿沟。

![图5：工具可用性矩阵](fig5_tool_matrix.png)

| 能力 | GPTSwarm 基线 | Enhanced ReAct | 备注 |
| :--- | :---: | :---: | :--- |
| **网页搜索** | ✓ | ✓ | DDG+Wikipedia vs. Google |
| **网页内容读取** | ✗ | ✓ | 基线无法读取网页正文 |
| **PDF/Excel/Word 解析** | ✗ | ✓ | **关键能力缺失** |
| **代码执行** | ✗ | ✓ | **关键能力缺失** |
| **图像分析** | ✗ | ✓ | **关键能力缺失** |

### 4.2. 逐级分析：难度越高，工具越重要

我们将难度等级（Level 1-3）与是否需要文件工具进行交叉分析，发现随着难度增加，工具的重要性愈发凸显。

![图4：按难度和文件需求划分的准确率](fig4_level_by_file.png)

> **图4解读**: 在所有难度等级中，一旦问题涉及文件处理（右侧“Has File”组），基线的表现都接近于零，而 `Enhanced ReAct` 则维持了较高的准确率。尤其是在 Level 2 和 Level 3 的复杂问题中，这种差距最为明显。

## 5. 标准与实践：GAIA 排行榜的启示

我们的调研发现，GAIA 作为一个旨在评估通用 AI 助手的基准，其设计初衷就是测试一个 **完整的、配备了全面工具的 Agent 系统** [2]。

*   **排行榜现状**: Hugging Face 上的官方排行榜 [3] 显示，所有顶尖系统（如 JD、NVIDIA、Lenovo 的方案）都明确使用了包括网页浏览、文件处理在内的丰富工具集，并且通常会融合多个强大的大模型（如 GPT-5, Gemini-3 等）。
*   **GPTSwarm 论文 vs. 代码**: 我们发现 `GPTSwarm` 的原始论文 [4] 中提到其框架支持“41种文件分析和谷歌搜索”，并报告了 18.45% 的平均准确率。然而，其开源代码中的 GAIA 实验配置远未达到此标准，这解释了为何我们复现的基线结果（14.5%）低于其论文报告值。

这进一步证实，**使用一个“残缺”的工具集去跑分，不仅违背了 GAIA 的设计原则，也使得跨研究的比较变得不可靠。**

## 6. 结论与建议：如何为 AgentGroup 建立公平的实验

当前的实验设置存在四个主要的混杂因素，使得我们无法得出关于 Agent 架构优劣的有效结论。

![图6：实验中的四个混杂因素](fig6_confounding_factors.png)

为了在 AgentGroup 论文中进行严谨、可信的实验，我们强烈建议采取以下措施：

1.  **统一工具标准 (Standardize Tools)**: 为所有参与比较的 Agent（包括 AgentGroup 和所有基线）配备 **完全相同、功能全面** 的工具套件。这应至少包括：
    *   高质量的网页搜索（如 Google SerpAPI）
    *   网页内容读取
    *   PDF、Excel/CSV、Word 文档解析
    *   Python 代码执行沙箱
    *   图像内容分析

2.  **统一模型 (Standardize Model)**: 所有 Agent 必须使用 **完全相同的大语言模型**（例如，统一使用 `gpt-4.1-mini` 或其他选定的模型）。这是隔离架构影响的先决条件。

3.  **分维度报告 (Report by Dimension)**: 在报告结果时，除了报告总体准确率，还应该 **按问题所需的核心能力进行分类报告**。例如，分别报告在“仅需搜索”、“需要文件解析”、“需要代码执行”等不同子集上的表现。这能更清晰地展示不同架构在特定能力上的优势和劣势。

4.  **明确声明实验设置**: 在论文的实验章节，必须用一个清晰的表格，详细说明每个 Agent 的 **模型、工具套件、搜索引擎以及其他关键超参数**，确保实验的可复现性和透明度。

通过遵循以上原则，AgentGroup 的实验结果将更具说服力，能够清晰地证明其架构创新所带来的真实价值，从而更有力地冲击 ICML 等顶级会议。

---

## 7. 参考文献

[1] Mialon, G., et al. (2023). *GAIA: A Benchmark for General AI Assistants*. arXiv:2311.12983. [https://arxiv.org/abs/2311.12983](https://arxiv.org/abs/2311.12983)

[2] Hugging Face. (n.d.). *What is GAIA?*. Hugging Face Agents Course. [https://huggingface.co/learn/agents-course/en/unit4/what-is-gaia](https://huggingface.co/learn/agents-course/en/unit4/what-is-gaia)

[3] GAIA Benchmark Community. (2026). *GAIA Leaderboard*. Hugging Face Spaces. [https://huggingface.co/spaces/gaia-benchmark/leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)

[4] Zhuge, M., et al. (2024). *GPTSwarm: Language Agents as Optimizable Graphs*. arXiv:2402.16823. [https://arxiv.org/abs/2402.16823](https://arxiv.org/abs/2402.16823)

## 8. 附录

*   [完整的公平性分析命令行输出 (fairness_analysis_output.txt)](fairness_analysis_output.txt)
*   [分析所用的 JSON 数据 (fairness_analysis_v2.json)](fairness_analysis_v2.json)
