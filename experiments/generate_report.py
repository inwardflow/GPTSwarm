#!/usr/bin/env python3
"""Generate comprehensive GAIA experiment report with visualizations."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Output directory
OUT_DIR = '/home/ubuntu/gaia_enhanced_report'
os.makedirs(OUT_DIR, exist_ok=True)

# Load enhanced results
with open('/home/ubuntu/GPTSwarm/experiments/gaia_enhanced_results/gaia_enhanced_gpt-4.1-mini_full.json') as f:
    enhanced_data = json.load(f)

# Previous baseline results
baseline_results = {
    'Direct': {'overall': 13.3, 'l1': 22.6, 'l2': 10.5, 'l3': 3.8, 'correct': 22, 'total': 165, 'avg_time': 21.2},
    'WebSearch': {'overall': 14.5, 'l1': 22.6, 'l2': 12.8, 'l3': 3.8, 'correct': 24, 'total': 165, 'avg_time': 168.0},
    'ToolTOT': {'overall': 12.1, 'l1': 20.8, 'l2': 8.1, 'l3': 7.7, 'correct': 20, 'total': 165, 'avg_time': 168.8},
}

# Enhanced results
enhanced_correct = sum(1 for r in enhanced_data if r.get('is_correct'))
enhanced_total = len(enhanced_data)
enhanced_accuracy = 100 * enhanced_correct / enhanced_total

# Level breakdown for enhanced
levels_enhanced = {}
for r in enhanced_data:
    lv = str(r.get('level', '?'))
    if lv not in levels_enhanced:
        levels_enhanced[lv] = {'c': 0, 't': 0, 'time': 0}
    levels_enhanced[lv]['t'] += 1
    levels_enhanced[lv]['time'] += r.get('time', 0)
    if r['is_correct']:
        levels_enhanced[lv]['c'] += 1

total_time = sum(r.get('time', 0) for r in enhanced_data)
avg_time = total_time / enhanced_total

enhanced_results = {
    'overall': enhanced_accuracy,
    'l1': 100 * levels_enhanced['1']['c'] / levels_enhanced['1']['t'],
    'l2': 100 * levels_enhanced['2']['c'] / levels_enhanced['2']['t'],
    'l3': 100 * levels_enhanced['3']['c'] / levels_enhanced['3']['t'],
    'correct': enhanced_correct,
    'total': enhanced_total,
    'avg_time': avg_time,
}

# Tool usage
tools = {}
for r in enhanced_data:
    for t in r.get('tools_used', []):
        tools[t] = tools.get(t, 0) + 1

# Status breakdown
statuses = {}
for r in enhanced_data:
    s = r.get('status', 'unknown')
    statuses[s] = statuses.get(s, 0) + 1

# ============================================================
# FIGURE 1: Overall Accuracy Comparison
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(10, 6))
strategies = ['Direct\n(gpt-5.1-codex-mini)', 'WebSearch\n(gpt-5.1-codex-mini)', 'ToolTOT\n(gpt-5.1-codex-mini)', 'Enhanced ReAct\n(gpt-4.1-mini)']
accuracies = [13.3, 14.5, 12.1, enhanced_accuracy]
colors = ['#95a5a6', '#7f8c8d', '#bdc3c7', '#e74c3c']

bars = ax.bar(strategies, accuracies, color=colors, edgecolor='white', linewidth=2, width=0.6)
bars[-1].set_edgecolor('#c0392b')
bars[-1].set_linewidth(3)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('GAIA Benchmark: Overall Accuracy Comparison', fontsize=15, fontweight='bold')
ax.set_ylim(0, 45)
ax.axhline(y=enhanced_accuracy, color='#e74c3c', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 2: Accuracy by Level
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(3)
width = 0.2

level_labels = ['Level 1\n(n=53)', 'Level 2\n(n=86)', 'Level 3\n(n=26)']

direct_vals = [22.6, 10.5, 3.8]
websearch_vals = [22.6, 12.8, 3.8]
tooltot_vals = [20.8, 8.1, 7.7]
enhanced_vals = [enhanced_results['l1'], enhanced_results['l2'], enhanced_results['l3']]

rects1 = ax.bar(x - 1.5*width, direct_vals, width, label='Direct', color='#95a5a6', edgecolor='white')
rects2 = ax.bar(x - 0.5*width, websearch_vals, width, label='WebSearch', color='#7f8c8d', edgecolor='white')
rects3 = ax.bar(x + 0.5*width, tooltot_vals, width, label='ToolTOT', color='#bdc3c7', edgecolor='white')
rects4 = ax.bar(x + 1.5*width, enhanced_vals, width, label='Enhanced ReAct', color='#e74c3c', edgecolor='white')

def autolabel(rects, fontsize=10):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('GAIA Benchmark: Accuracy by Difficulty Level', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(level_labels, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 55)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/accuracy_by_level.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: Tool Usage
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
tool_names = list(tools.keys())
tool_counts = list(tools.values())
sorted_idx = sorted(range(len(tool_counts)), key=lambda i: tool_counts[i], reverse=True)
tool_names = [tool_names[i] for i in sorted_idx]
tool_counts = [tool_counts[i] for i in sorted_idx]

tool_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
bars = ax.barh(tool_names, tool_counts, color=tool_colors[:len(tool_names)], edgecolor='white')
for bar, count in zip(bars, tool_counts):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2.,
            str(count), ha='left', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Number of Invocations', fontsize=12)
ax.set_title('Enhanced ReAct Agent: Tool Usage Distribution', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/tool_usage.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: Time Distribution by Level
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

time_by_level = {1: [], 2: [], 3: []}
for r in enhanced_data:
    lv = r.get('level', 0)
    if lv in time_by_level:
        time_by_level[lv].append(r.get('time', 0))

bp = ax.boxplot([time_by_level[1], time_by_level[2], time_by_level[3]],
                labels=['Level 1', 'Level 2', 'Level 3'],
                patch_artist=True,
                boxprops=dict(facecolor='#3498db', alpha=0.7),
                medianprops=dict(color='#e74c3c', linewidth=2))

colors = ['#3498db', '#2ecc71', '#e67e22']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Time (seconds)', fontsize=13)
ax.set_title('Enhanced ReAct Agent: Response Time by Difficulty Level', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/time_by_level.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 5: Improvement Summary
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Overall', 'Level 1', 'Level 2', 'Level 3']
best_baseline = [14.5, 22.6, 12.8, 7.7]  # Best of Direct/WebSearch/ToolTOT
enhanced = [enhanced_accuracy, enhanced_results['l1'], enhanced_results['l2'], enhanced_results['l3']]
improvements = [e - b for e, b in zip(enhanced, best_baseline)]

x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width/2, best_baseline, width, label='Best Baseline (gpt-5.1-codex-mini)', color='#95a5a6')
rects2 = ax.bar(x + width/2, enhanced, width, label='Enhanced ReAct (gpt-4.1-mini)', color='#e74c3c')

for i, (b, e, imp) in enumerate(zip(best_baseline, enhanced, improvements)):
    ax.annotate(f'+{imp:.1f}pp',
                xy=(i + width/2, e),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#27ae60')

autolabel(rects1, fontsize=10)
autolabel(rects2, fontsize=10)

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Improvement: Enhanced ReAct vs Best Baseline', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 55)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/improvement_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print(f"Output directory: {OUT_DIR}")
for f in os.listdir(OUT_DIR):
    print(f"  {f}")
