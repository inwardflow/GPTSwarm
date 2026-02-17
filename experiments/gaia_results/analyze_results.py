#!/usr/bin/env python3
"""
Full GAIA Experiment Analysis and Visualization.
Analyzes results from all 3 strategies across 165 questions.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Output directory
OUT_DIR = "/home/ubuntu/gaia_full_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Result files
FILES = {
    'Direct': '/home/ubuntu/GPTSwarm/result/eval/gaia_direct_gpt-5.1-codex-mini_FULL.json',
    'WebSearch': '/home/ubuntu/GPTSwarm/result/eval/gaia_websearch_gpt-5.1-codex-mini_2026-02-16-13-34-38.json',
    'ToolTOT': '/home/ubuntu/GPTSwarm/result/eval/gaia_tooltot_gpt-5.1-codex-mini_2026-02-16-13-34-40.json',
}

# Load data
all_data = {}
for name, path in FILES.items():
    with open(path) as f:
        all_data[name] = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'Direct': '#2196F3', 'WebSearch': '#4CAF50', 'ToolTOT': '#FF9800'}

# ============================================================
# Figure 1: Overall Accuracy Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1a: Overall accuracy bar chart
strategies = list(FILES.keys())
overall_acc = []
overall_correct = []
overall_total = []
for s in strategies:
    data = all_data[s]
    correct = sum(1 for d in data if d.get('Solved', False))
    overall_acc.append(correct / len(data) * 100)
    overall_correct.append(correct)
    overall_total.append(len(data))

bars = axes[0].bar(strategies, overall_acc, color=[colors[s] for s in strategies], width=0.6, edgecolor='white', linewidth=1.5)
for bar, acc, c, t in zip(bars, overall_acc, overall_correct, overall_total):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%\n({c}/{t})', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Overall Accuracy on GAIA Validation Set (165 Questions)', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, max(overall_acc) * 1.4)

# 1b: Accuracy by Level
levels = [1, 2, 3]
level_counts = {1: 53, 2: 86, 3: 26}
x = np.arange(len(levels))
width = 0.25

for i, s in enumerate(strategies):
    data = all_data[s]
    accs = []
    for lvl in levels:
        lvl_data = [d for d in data if d.get('Level') == lvl]
        correct = sum(1 for d in lvl_data if d.get('Solved', False))
        accs.append(correct / len(lvl_data) * 100 if lvl_data else 0)
    bars = axes[1].bar(x + i * width, accs, width, label=s, color=colors[s], edgecolor='white', linewidth=1)
    for j, (bar, acc) in enumerate(zip(bars, accs)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[1].set_xlabel('GAIA Level', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Accuracy by Difficulty Level', fontsize=13, fontweight='bold')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels([f'Level {l}\n(n={level_counts[l]})' for l in levels])
axes[1].legend(fontsize=10)
axes[1].set_ylim(0, 35)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 2: Per-Question Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(20, 6))

# Build matrix: rows = strategies, cols = questions
n_questions = 165
matrix = np.zeros((3, n_questions))
question_levels = []

for i, s in enumerate(strategies):
    data = all_data[s]
    for j, d in enumerate(data):
        if j < n_questions:
            matrix[i, j] = 1 if d.get('Solved', False) else 0
            if i == 0:
                question_levels.append(d.get('Level', 0))

# Custom colormap
cmap = matplotlib.colors.ListedColormap(['#ffcdd2', '#c8e6c9'])
sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=False, linewidths=0.3, linecolor='white',
            yticklabels=strategies, xticklabels=False)

# Add level separators
level_boundaries = []
cumsum = 0
for lvl in [1, 2, 3]:
    count = sum(1 for l in question_levels if l == lvl)
    cumsum += count
    level_boundaries.append(cumsum)

for boundary in level_boundaries[:-1]:
    ax.axvline(x=boundary, color='black', linewidth=2, linestyle='--')

# Add level labels
ax.text(level_boundaries[0]/2, -0.5, f'Level 1 (n={level_counts[1]})', ha='center', fontsize=10, fontweight='bold')
if len(level_boundaries) > 1:
    ax.text((level_boundaries[0] + level_boundaries[1])/2, -0.5, f'Level 2 (n={level_counts[2]})', ha='center', fontsize=10, fontweight='bold')
if len(level_boundaries) > 2:
    ax.text((level_boundaries[1] + level_boundaries[2])/2, -0.5, f'Level 3 (n={level_counts[3]})', ha='center', fontsize=10, fontweight='bold')

ax.set_title('Per-Question Results (Green = Correct, Red = Incorrect)', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/per_question_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 3: Time Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3a: Box plot of time per question
time_data = []
time_labels = []
for s in strategies:
    times = [d.get('Time', 0) for d in all_data[s] if d.get('Time', 0) > 0]
    time_data.append(times)
    time_labels.append(s)

bp = axes[0].boxplot(time_data, labels=time_labels, patch_artist=True, showfliers=False)
for patch, s in zip(bp['boxes'], strategies):
    patch.set_facecolor(colors[s])
    patch.set_alpha(0.7)

axes[0].set_ylabel('Time per Question (seconds)', fontsize=12)
axes[0].set_title('Response Time Distribution', fontsize=13, fontweight='bold')

# Add median annotations
for i, (data_points, s) in enumerate(zip(time_data, strategies)):
    median = np.median(data_points)
    axes[0].text(i+1, median + 2, f'{median:.0f}s', ha='center', fontsize=10, fontweight='bold')

# 3b: Average time comparison
avg_times = [np.mean(t) for t in time_data]
total_times = [np.sum(t)/3600 for t in time_data]

bars = axes[1].bar(strategies, avg_times, color=[colors[s] for s in strategies], width=0.6, edgecolor='white', linewidth=1.5)
for bar, avg, total in zip(bars, avg_times, total_times):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{avg:.0f}s avg\n({total:.1f}h total)', ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[1].set_ylabel('Average Time per Question (seconds)', fontsize=12)
axes[1].set_title('Average Response Time', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, max(avg_times) * 1.4)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/time_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 4: Accuracy vs Time Scatter
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

for s in strategies:
    data = all_data[s]
    times = [d.get('Time', 0) for d in data if d.get('Time', 0) > 0]
    solved = [d.get('Solved', False) for d in data if d.get('Time', 0) > 0]
    
    correct_times = [t for t, s_flag in zip(times, solved) if s_flag]
    incorrect_times = [t for t, s_flag in zip(times, solved) if not s_flag]
    
    ax.scatter(correct_times, [s]*len(correct_times), 
              color=colors[s], marker='o', s=30, alpha=0.6, label=f'{s} (correct)')
    ax.scatter(incorrect_times, [s]*len(incorrect_times),
              color=colors[s], marker='x', s=20, alpha=0.3)

ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_title('Question Response Times by Strategy (o = correct, x = incorrect)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 200)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/time_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 5: Strategy Agreement Analysis
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

# Venn-like analysis
direct_solved = set()
ws_solved = set()
tot_solved = set()

for i, d in enumerate(all_data['Direct']):
    if d.get('Solved', False):
        direct_solved.add(i)
for i, d in enumerate(all_data['WebSearch']):
    if d.get('Solved', False):
        ws_solved.add(i)
for i, d in enumerate(all_data['ToolTOT']):
    if d.get('Solved', False):
        tot_solved.add(i)

# Calculate overlaps
all_three = direct_solved & ws_solved & tot_solved
d_ws = (direct_solved & ws_solved) - all_three
d_tot = (direct_solved & tot_solved) - all_three
ws_tot = (ws_solved & tot_solved) - all_three
d_only = direct_solved - ws_solved - tot_solved
ws_only = ws_solved - direct_solved - tot_solved
tot_only = tot_solved - direct_solved - ws_solved

# Create a summary table as text
summary = f"""Strategy Agreement Analysis
================================

All 3 strategies correct:  {len(all_three)} questions
Direct + WebSearch only:   {len(d_ws)} questions
Direct + ToolTOT only:     {len(d_tot)} questions
WebSearch + ToolTOT only:  {len(ws_tot)} questions
Direct only:               {len(d_only)} questions
WebSearch only:            {len(ws_only)} questions
ToolTOT only:              {len(tot_only)} questions

Union (any correct):       {len(direct_solved | ws_solved | tot_solved)} questions
None correct:              {165 - len(direct_solved | ws_solved | tot_solved)} questions
"""

ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=13,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Strategy Agreement Analysis', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/strategy_agreement.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Save detailed results to JSON
# ============================================================
summary_data = {
    "experiment": "GAIA Full Validation Set (165 Questions)",
    "model": "gpt-5.1-codex-mini",
    "api_endpoint": "<configured via .env>",
    "dataset": "GAIA Benchmark Validation Set",
    "strategies": {}
}

for s in strategies:
    data = all_data[s]
    correct = sum(1 for d in data if d.get('Solved', False))
    times = [d.get('Time', 0) for d in data if d.get('Time', 0) > 0]
    
    level_results = {}
    for lvl in [1, 2, 3]:
        lvl_data = [d for d in data if d.get('Level') == lvl]
        lvl_correct = sum(1 for d in lvl_data if d.get('Solved', False))
        level_results[f"Level_{lvl}"] = {
            "total": len(lvl_data),
            "correct": lvl_correct,
            "accuracy": f"{lvl_correct/len(lvl_data)*100:.1f}%" if lvl_data else "N/A"
        }
    
    summary_data["strategies"][s] = {
        "total": len(data),
        "correct": correct,
        "accuracy": f"{correct/len(data)*100:.1f}%",
        "avg_time_seconds": f"{np.mean(times):.1f}" if times else "N/A",
        "total_time_hours": f"{np.sum(times)/3600:.1f}" if times else "N/A",
        "by_level": level_results
    }

summary_data["agreement"] = {
    "all_three_correct": len(all_three),
    "any_correct": len(direct_solved | ws_solved | tot_solved),
    "none_correct": 165 - len(direct_solved | ws_solved | tot_solved),
}

with open(f'{OUT_DIR}/experiment_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("Analysis complete! Files saved to:", OUT_DIR)
print(f"\nOverall Results:")
for s in strategies:
    info = summary_data["strategies"][s]
    print(f"  {s}: {info['accuracy']} ({info['correct']}/{info['total']}), avg {info['avg_time_seconds']}s/q")
print(f"\nAgreement: {len(all_three)} all correct, {len(direct_solved | ws_solved | tot_solved)} any correct, {165 - len(direct_solved | ws_solved | tot_solved)} none correct")
