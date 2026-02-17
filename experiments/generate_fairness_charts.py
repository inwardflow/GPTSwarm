#!/usr/bin/env python3
"""Generate comprehensive fairness analysis visualizations for GAIA experiment."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')

OUT_DIR = '/home/ubuntu/gaia_enhanced_report'
os.makedirs(OUT_DIR, exist_ok=True)

# Load analysis data
with open('/home/ubuntu/GPTSwarm/experiments/gaia_enhanced_results/fairness_analysis_v2.json') as f:
    data = json.load(f)

# Color scheme
C_DIRECT = '#95a5a6'
C_WEBSEARCH = '#7f8c8d'
C_TOOLTOT = '#bdc3c7'
C_ENHANCED = '#e74c3c'
C_FAIR = '#3498db'
C_UNFAIR = '#e67e22'

# ============================================================
# FIGURE 1: Overall Accuracy with correct baseline data
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
strategies = ['Direct\n(codex-mini)', 'WebSearch\n(codex-mini)', 'ToolTOT\n(codex-mini)', 'Enhanced ReAct\n(gpt-4.1-mini)']
n = data['overall']['total']
accuracies = [
    100 * data['overall']['direct'] / n,
    100 * data['overall']['websearch'] / n,
    100 * data['overall']['tooltot'] / n,
    100 * data['overall']['enhanced'] / n,
]
colors = [C_DIRECT, C_WEBSEARCH, C_TOOLTOT, C_ENHANCED]

bars = ax.bar(strategies, accuracies, color=colors, edgecolor='white', linewidth=2, width=0.6)
bars[-1].set_edgecolor('#c0392b')
bars[-1].set_linewidth(3)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('GAIA Benchmark: Overall Accuracy (165 Questions)', fontsize=15, fontweight='bold')
ax.set_ylim(0, 45)

# Add warning annotation
ax.annotate('⚠ Not a fair comparison\n(different models + tools)',
            xy=(3, accuracies[3]), xytext=(2.0, 42),
            fontsize=10, color='#c0392b', ha='center',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_overall_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 2: Fair vs Unfair split - stacked bar
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (subset_key, title, marker) in enumerate([
    ('no_file', 'No File Attachment (n=127)\nRelatively Fair Comparison', 'FAIR'),
    ('has_file', 'Has File Attachment (n=38)\nUnfair: Baseline Lacks File Tools', 'UNFAIR'),
]):
    ax = axes[ax_idx]
    sd = data[subset_key]
    sn = sd['total']
    
    strats = ['Direct', 'WebSearch', 'ToolTOT', 'Enhanced\nReAct']
    accs = [100*sd['direct']/sn, 100*sd['websearch']/sn, 100*sd['tooltot']/sn, 100*sd['enhanced']/sn]
    colors_local = [C_DIRECT, C_WEBSEARCH, C_TOOLTOT, C_ENHANCED]
    
    bars = ax.bar(strats, accs, color=colors_local, edgecolor='white', linewidth=2, width=0.6)
    
    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='#888')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 55)
    
    # Add colored border
    border_color = C_FAIR if marker == 'FAIR' else C_UNFAIR
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2)

plt.suptitle('GAIA Benchmark: Fair vs Unfair Comparison Split', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_fair_vs_unfair.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: Improvement decomposition waterfall
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

n_total = data['overall']['total']
websearch_acc = 100 * data['overall']['websearch'] / n_total
enhanced_acc = 100 * data['overall']['enhanced'] / n_total

# Decompose: fair subset contribution + unfair subset contribution
nf = data['no_file']
hf = data['has_file']

fair_delta = 100 * (nf['enhanced'] - nf['websearch']) / n_total
unfair_delta = 100 * (hf['enhanced'] - hf['websearch']) / n_total

# Waterfall chart
categories = ['WebSearch\nBaseline', 'Architecture +\nModel + Search\n(no-file questions)', 
              'Tool Advantage\n(file questions)', 'Enhanced\nReAct']
values = [websearch_acc, fair_delta, unfair_delta, enhanced_acc]
bottoms = [0, websearch_acc, websearch_acc + fair_delta, 0]

colors_wf = ['#95a5a6', '#3498db', '#e67e22', '#e74c3c']

for i, (cat, val, bot, col) in enumerate(zip(categories, values, bottoms, colors_wf)):
    if i == 0 or i == 3:
        ax.bar(i, val, bottom=0, color=col, edgecolor='white', linewidth=2, width=0.6)
        ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    else:
        ax.bar(i, val, bottom=bot, color=col, edgecolor='white', linewidth=2, width=0.6, alpha=0.85)
        ax.text(i, bot + val + 0.5, f'+{val:.1f}pp', ha='center', va='bottom', fontsize=12, fontweight='bold', color=col)
        # Connector line
        if i < 3:
            ax.plot([i-0.3, i+0.7], [bot+val, bot+val], color='gray', linewidth=1, linestyle='--', alpha=0.5)

# Connector from baseline to first delta
ax.plot([-0.3, 0.7], [websearch_acc, websearch_acc], color='gray', linewidth=1, linestyle='--', alpha=0.5)

ax.set_xticks(range(4))
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Improvement Decomposition: What Drives the +20.7pp Gain?', fontsize=15, fontweight='bold')
ax.set_ylim(0, 42)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label=f'Model+Arch+Search ({fair_delta:.1f}pp, {100*fair_delta/(fair_delta+unfair_delta):.0f}%)'),
    Patch(facecolor='#e67e22', label=f'+ Tool Advantage ({unfair_delta:.1f}pp, {100*unfair_delta/(fair_delta+unfair_delta):.0f}%)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_improvement_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: By Level × File Attachment
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

for lv_idx, lv in enumerate([1, 2, 3]):
    ax = axes[lv_idx]
    
    # No file
    nf_data = data['by_level_no_file'].get(str(lv), {'enhanced': 0, 'websearch': 0, 'total': 0})
    hf_data = data['by_level_has_file'].get(str(lv), {'enhanced': 0, 'websearch': 0, 'total': 0})
    
    x = np.arange(2)
    width = 0.35
    
    nf_n = nf_data['total'] if nf_data['total'] > 0 else 1
    hf_n = hf_data['total'] if hf_data['total'] > 0 else 1
    
    ws_vals = [100*nf_data.get('websearch', 0)/nf_n, 100*hf_data.get('websearch', 0)/hf_n if hf_data['total'] > 0 else 0]
    en_vals = [100*nf_data.get('enhanced', 0)/nf_n, 100*hf_data.get('enhanced', 0)/hf_n if hf_data['total'] > 0 else 0]
    
    r1 = ax.bar(x - width/2, ws_vals, width, label='WebSearch (baseline)', color=C_WEBSEARCH)
    r2 = ax.bar(x + width/2, en_vals, width, label='Enhanced ReAct', color=C_ENHANCED)
    
    for bars in [r1, r2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    labels = [f'No File\n(n={nf_data["total"]})', f'Has File\n(n={hf_data["total"]})']
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(f'Level {lv}', fontsize=14, fontweight='bold')
    
    if lv_idx == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=13)
    if lv_idx == 2:
        ax.legend(fontsize=10)

axes[0].set_ylim(0, 65)
plt.suptitle('Accuracy by Level and File Attachment', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_level_by_file.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 5: Tool Availability Matrix (visual)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

tools = ['Web Search', 'Webpage Reading', 'PDF Parsing', 'Excel/CSV', 'Word Docs', 
         'Image Analysis', 'Code Execution', 'ZIP Handling', 'Audio Processing']
baseline_has = [1, 0, 0, 0, 0, 0, 0, 0, 0]  # Only web search (DDG)
enhanced_has = [1, 1, 1, 1, 1, 1, 1, 1, 0]  # Everything except audio

y = np.arange(len(tools))
height = 0.35

# Baseline
for i, has in enumerate(baseline_has):
    color = '#27ae60' if has else '#e74c3c'
    ax.barh(i + height/2, 1, height, color=color, alpha=0.7, edgecolor='white')
    ax.text(0.5, i + height/2, '✓' if has else '✗', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')

# Enhanced
for i, has in enumerate(enhanced_has):
    color = '#27ae60' if has else '#e74c3c'
    ax.barh(i - height/2, 1, height, color=color, alpha=0.7, edgecolor='white')
    ax.text(0.5, i - height/2, '✓' if has else '✗', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')

ax.set_yticks(y)
ax.set_yticklabels(tools, fontsize=12)
ax.set_xticks([])
ax.set_xlim(0, 1)
ax.invert_yaxis()

# Legend
legend_elements = [
    Patch(facecolor=C_WEBSEARCH, label='GPTSwarm Baselines (DDG+Wiki only)'),
    Patch(facecolor=C_ENHANCED, label='Enhanced ReAct (5 tools)'),
]
ax.legend(handles=[
    mpatches.Patch(color=C_WEBSEARCH, label='GPTSwarm Baselines'),
    mpatches.Patch(color=C_ENHANCED, label='Enhanced ReAct'),
], loc='lower right', fontsize=11)

# Add column labels
ax.text(0.5, -0.8, 'Baseline | Enhanced', ha='center', va='center', fontsize=11, style='italic')

ax.set_title('Tool Availability: GPTSwarm Baselines vs Enhanced ReAct', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_tool_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 6: Confounding Factors Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Four Confounding Factors in Current Comparison', 
        ha='center', fontsize=16, fontweight='bold')

# Boxes for each factor
factors = [
    ('1. Tool Suite\nDifference', '23% of questions need\nfile processing tools\nBaseline: 0 tools\nEnhanced: 5 tools', '#e74c3c', 0.5),
    ('2. Search Quality\nDifference', 'DDG+Wikipedia\nvs\nGoogle SerpAPI', '#e67e22', 3.0),
    ('3. Model\nDifference', 'gpt-5.1-codex-mini\nvs\ngpt-4.1-mini', '#f1c40f', 5.5),
    ('4. Architecture\nDifference', 'Fixed pipeline (1-5 calls)\nvs\nReAct (12 turns)', '#3498db', 8.0),
]

for title, desc, color, x in factors:
    rect = plt.Rectangle((x, 2.5), 2.0, 5.5, linewidth=2, edgecolor=color, facecolor=color, alpha=0.15)
    ax.add_patch(rect)
    ax.text(x + 1.0, 7.5, title, ha='center', va='center', fontsize=11, fontweight='bold', color=color)
    ax.text(x + 1.0, 5.0, desc, ha='center', va='center', fontsize=9, color='#333')

# Arrow showing what we want to measure
ax.annotate('What we WANT\nto measure', xy=(9.0, 5.0), xytext=(9.0, 1.5),
            fontsize=11, fontweight='bold', color='#3498db', ha='center',
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))

# Arrow showing confounds
ax.annotate('Confounding\nfactors', xy=(2.0, 5.0), xytext=(2.0, 1.5),
            fontsize=11, fontweight='bold', color='#e74c3c', ha='center',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_confounding_factors.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 7: Venn-style overlap (regression analysis)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Data from analysis
both = 21
enhanced_only = 37
baseline_only = 16
neither = 91

labels = ['Both Correct\n(21)', 'Enhanced Only\n(37)', 'Baseline Only\n(16)', 'Neither\n(91)']
sizes = [both, enhanced_only, baseline_only, neither]
colors_pie = ['#27ae60', '#e74c3c', '#95a5a6', '#ecf0f1']
explode = (0.05, 0.05, 0.05, 0)

wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, explode=explode,
                                   autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_fontweight('bold')

ax.set_title('Question-Level Agreement: Baseline vs Enhanced ReAct', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_agreement_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith('fig'):
        print(f"  {f}")
