## Run the following commands to reproduce our experiments in the paper

### **MMLU**
Run the baseline:
```bash
PYTHONPATH=. python experiments/run_mmlu.py --mode=DirectAnswer
```

Run fully-connected swarm ablation:
```bash
PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=FullConnectedSwarm
```

Run randomly-connected swarm ablation:
```bash
PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=RandomSwarm
```

Run the main experiment with optimization and eventual evaluation:
```bash
PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm
```

### **Mini Crosswords**
Run the REINFORCE algorithm for edge optimization with three agents as described in the paper.
```bash
PYTHONPATH=. python experiments/run_crosswords.py
```

### **HumanEval**
Run node optimization that improves the demonstration examples of each node.
```bash
PYTHONPATH=. python experiments/run_humaneval.py  --learn_demonstration True
```

### **GAIA**
Run the general assistant tasks.
```bash
PYTHONPATH=. python experiments/run_gaia.py
```

---

## GAIA Benchmark Extended Experiments

We conducted extended experiments on the full GAIA validation set (165 questions) to evaluate different agent strategies and tool configurations.

### Experiment 1: GPTSwarm Baseline Strategies (gpt-5.1-codex-mini)

Three built-in GPTSwarm strategies evaluated with basic search tools (DuckDuckGo + Wikipedia):

| Strategy | Accuracy | Avg Time/Q |
|----------|----------|------------|
| Direct (no tools) | 13.3% | 21.2s |
| WebSearch | 14.5% | 168.0s |
| ToolTOT | 12.1% | 168.8s |

```bash
# Run all three strategies
python experiments/run_gaia_direct.py
```

Results: `experiments/gaia_results/`

### Experiment 2: Enhanced ReAct Agent (gpt-4.1-mini)

Single-agent ReAct architecture with 5 enhanced tools:

| Tool | Description |
|------|-------------|
| `search_web` | Google-powered web search |
| `fetch_webpage` | Full webpage content extraction |
| `read_file` | Multi-format file reader (PDF, Excel, CSV, DOCX, ZIP, images via OCR) |
| `run_python` | Sandboxed Python code execution |
| `analyze_image` | Vision-based image analysis |

**Results: 35.2% accuracy (+20.7pp over best baseline, 2.4x improvement)**

| Level | Accuracy | Questions |
|-------|----------|-----------|
| Level 1 | 43.4% | 53 |
| Level 2 | 38.4% | 86 |
| Level 3 | 7.7% | 26 |
| **Overall** | **35.2%** | **165** |

```bash
# Run enhanced ReAct agent
python experiments/run_gaia_enhanced.py --model gpt-4.1-mini --num_questions 165

# Generate analysis report and visualizations
python experiments/generate_report.py
```

Results: `experiments/gaia_enhanced_results/`

### Key Findings

1. **Tool quality > Complex multi-agent architectures**: A single ReAct agent with rich tools (35.2%) dramatically outperforms multi-agent pipelines with basic tools (12-14%).
2. **Level 2 benefits most** (+25.6pp): Multi-step reasoning with file processing and code execution provides the greatest improvement.
3. **Level 3 remains challenging** (7.7%): Complex tasks requiring 10+ reasoning steps exceed current agent capabilities.
4. **Efficiency**: Enhanced ReAct is 3.7x faster than WebSearch/ToolTOT while being 2.4x more accurate.

Full analysis report: `experiments/gaia_enhanced_results/` and `gaia_results/GAIA_Full_Experiment_Report.md`
