#!/usr/bin/env python3
"""
Analyze GAIA experiment results with focus on tool fairness.
Classify questions by required capabilities and compare accuracy across configurations.
"""

import json
import os
import re

# ============================================================
# Load all data
# ============================================================
with open('/home/ubuntu/GPTSwarm/datasets/gaia/gaia_full_validation.json') as f:
    dataset = json.load(f)

with open('/home/ubuntu/GPTSwarm/experiments/gaia_enhanced_results/gaia_enhanced_gpt-4.1-mini_full.json') as f:
    enhanced = json.load(f)

with open('/home/ubuntu/GPTSwarm/experiments/gaia_results/gaia_direct_gpt-5.1-codex-mini_FULL.json') as f:
    direct = json.load(f)
with open('/home/ubuntu/GPTSwarm/experiments/gaia_results/gaia_websearch_gpt-5.1-codex-mini_FULL.json') as f:
    websearch = json.load(f)
with open('/home/ubuntu/GPTSwarm/experiments/gaia_results/gaia_tooltot_gpt-5.1-codex-mini_FULL.json') as f:
    tooltot = json.load(f)

# Build lookups
enhanced_by_id = {}
for r in enhanced:
    tid = r.get('task_id', r.get('id', ''))
    enhanced_by_id[tid] = r

direct_by_id = {}
for r in direct:
    tid = r.get('task_id', r.get('id', ''))
    direct_by_id[tid] = r

websearch_by_id = {}
for r in websearch:
    tid = r.get('task_id', r.get('id', ''))
    websearch_by_id[tid] = r

tooltot_by_id = {}
for r in tooltot:
    tid = r.get('task_id', r.get('id', ''))
    tooltot_by_id[tid] = r

# ============================================================
# Classify questions by REQUIRED CAPABILITIES
# ============================================================
file_ext_map = {
    '.pdf': 'file_parsing',
    '.xlsx': 'file_parsing', '.xls': 'file_parsing', '.csv': 'file_parsing',
    '.png': 'image_understanding', '.jpg': 'image_understanding', '.jpeg': 'image_understanding',
    '.gif': 'image_understanding', '.bmp': 'image_understanding',
    '.mp3': 'audio_processing', '.wav': 'audio_processing', '.mp4': 'audio_processing',
    '.py': 'code_execution', '.js': 'code_execution',
    '.docx': 'file_parsing', '.doc': 'file_parsing',
    '.pptx': 'file_parsing', '.zip': 'file_parsing', '.jsonld': 'file_parsing',
    '.txt': 'file_parsing', '.pdb': 'file_parsing',
}

# Heuristic: detect if question likely needs code execution or computation
computation_keywords = [
    'calculate', 'compute', 'sum', 'total', 'average', 'mean', 'how many',
    'count', 'number of', 'difference between', 'percentage', 'ratio',
    'convert', 'formula', 'equation', 'solve', 'maximum', 'minimum',
]

questions_classified = []
for q in dataset:
    task_id = q.get('task_id', '')
    file_name = q.get('file_name', '')
    question_text = q.get('Question', q.get('question', '')).lower()
    level = q.get('Level', q.get('level', 0))
    
    capabilities_needed = set()
    
    # File-based capabilities
    if file_name and file_name.strip():
        ext = os.path.splitext(file_name)[1].lower()
        cap = file_ext_map.get(ext, 'file_parsing')
        capabilities_needed.add(cap)
    
    # Computation detection
    if any(kw in question_text for kw in computation_keywords):
        capabilities_needed.add('computation')
    
    # Web search is almost always needed for non-trivial questions
    if level >= 1:
        capabilities_needed.add('web_search')
    
    # Determine primary category
    if 'audio_processing' in capabilities_needed:
        primary_cat = 'audio_processing'
    elif 'image_understanding' in capabilities_needed:
        primary_cat = 'image_understanding'
    elif 'file_parsing' in capabilities_needed:
        primary_cat = 'file_parsing'
    elif 'computation' in capabilities_needed:
        primary_cat = 'web_search_only'  # needs search + maybe computation
    else:
        primary_cat = 'web_search_only'
    
    # Determine tool availability
    has_file = bool(file_name and file_name.strip())
    
    # Get results from each strategy
    e = enhanced_by_id.get(task_id, {})
    d = direct_by_id.get(task_id, {})
    w = websearch_by_id.get(task_id, {})
    t = tooltot_by_id.get(task_id, {})
    
    questions_classified.append({
        'task_id': task_id,
        'level': level,
        'has_file': has_file,
        'file_name': file_name,
        'primary_cat': primary_cat,
        'capabilities': capabilities_needed,
        'enhanced_correct': e.get('is_correct', False),
        'direct_correct': d.get('is_correct', False),
        'websearch_correct': w.get('is_correct', False),
        'tooltot_correct': t.get('is_correct', False),
        'enhanced_tools': e.get('tools_used', []),
    })

# ============================================================
# Analysis 1: Accuracy by file attachment presence
# ============================================================
print("=" * 90)
print("ANALYSIS 1: Accuracy by File Attachment Presence")
print("=" * 90)

for has_file_val, label in [(False, "No File Attachment"), (True, "Has File Attachment")]:
    subset = [q for q in questions_classified if q['has_file'] == has_file_val]
    n = len(subset)
    if n == 0:
        continue
    
    e_correct = sum(1 for q in subset if q['enhanced_correct'])
    d_correct = sum(1 for q in subset if q['direct_correct'])
    w_correct = sum(1 for q in subset if q['websearch_correct'])
    t_correct = sum(1 for q in subset if q['tooltot_correct'])
    
    print(f"\n{label} (n={n}):")
    print(f"  {'Strategy':<25} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Direct':<25} {d_correct:>8} {100*d_correct/n:>9.1f}%")
    print(f"  {'WebSearch':<25} {w_correct:>8} {100*w_correct/n:>9.1f}%")
    print(f"  {'ToolTOT':<25} {t_correct:>8} {100*t_correct/n:>9.1f}%")
    print(f"  {'Enhanced ReAct':<25} {e_correct:>8} {100*e_correct/n:>9.1f}%")

# ============================================================
# Analysis 2: Accuracy by primary capability category
# ============================================================
print(f"\n{'=' * 90}")
print("ANALYSIS 2: Accuracy by Required Capability")
print("=" * 90)

cat_labels = {
    'web_search_only': 'Web Search Only (no file)',
    'file_parsing': 'File Parsing (PDF/Excel/DOCX/ZIP)',
    'image_understanding': 'Image Understanding',
    'audio_processing': 'Audio Processing',
}

for cat, label in cat_labels.items():
    subset = [q for q in questions_classified if q['primary_cat'] == cat]
    n = len(subset)
    if n == 0:
        continue
    
    e_correct = sum(1 for q in subset if q['enhanced_correct'])
    d_correct = sum(1 for q in subset if q['direct_correct'])
    w_correct = sum(1 for q in subset if q['websearch_correct'])
    t_correct = sum(1 for q in subset if q['tooltot_correct'])
    
    # Baseline tools available for this category
    baseline_has_tools = cat == 'web_search_only'
    fair_marker = "FAIR" if baseline_has_tools else "UNFAIR (baseline lacks tools)"
    
    print(f"\n{label} (n={n}) [{fair_marker}]:")
    print(f"  {'Strategy':<25} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Direct':<25} {d_correct:>8} {100*d_correct/n:>9.1f}%")
    print(f"  {'WebSearch':<25} {w_correct:>8} {100*w_correct/n:>9.1f}%")
    print(f"  {'ToolTOT':<25} {t_correct:>8} {100*t_correct/n:>9.1f}%")
    print(f"  {'Enhanced ReAct':<25} {e_correct:>8} {100*e_correct/n:>9.1f}%")

# ============================================================
# Analysis 3: Fair comparison (web_search_only questions)
# ============================================================
print(f"\n{'=' * 90}")
print("ANALYSIS 3: FAIR COMPARISON (Web Search Only Questions)")
print("=" * 90)

fair_subset = [q for q in questions_classified if not q['has_file']]
n = len(fair_subset)

e_correct = sum(1 for q in fair_subset if q['enhanced_correct'])
d_correct = sum(1 for q in fair_subset if q['direct_correct'])
w_correct = sum(1 for q in fair_subset if q['websearch_correct'])
t_correct = sum(1 for q in fair_subset if q['tooltot_correct'])

print(f"\nQuestions without file attachments (n={n}):")
print(f"All strategies have access to web search capabilities.")
print(f"\n  {'Strategy':<25} {'Correct':>8} {'Accuracy':>10} {'vs Best Baseline':>18}")
print(f"  {'-'*65}")
best_baseline = max(d_correct, w_correct, t_correct)
best_baseline_acc = 100 * best_baseline / n
for name, correct in [('Direct', d_correct), ('WebSearch', w_correct), ('ToolTOT', t_correct), ('Enhanced ReAct', e_correct)]:
    acc = 100 * correct / n
    diff = acc - best_baseline_acc
    diff_str = f"+{diff:.1f}pp" if diff > 0 else f"{diff:.1f}pp" if diff < 0 else "baseline"
    print(f"  {name:<25} {correct:>8} {acc:>9.1f}% {diff_str:>18}")

# By level within fair subset
print(f"\n  By Level (fair subset only):")
for lv in [1, 2, 3]:
    lv_subset = [q for q in fair_subset if q['level'] == lv]
    if not lv_subset:
        continue
    n_lv = len(lv_subset)
    e_c = sum(1 for q in lv_subset if q['enhanced_correct'])
    d_c = sum(1 for q in lv_subset if q['direct_correct'])
    w_c = sum(1 for q in lv_subset if q['websearch_correct'])
    t_c = sum(1 for q in lv_subset if q['tooltot_correct'])
    print(f"    Level {lv} (n={n_lv}): Direct={100*d_c/n_lv:.1f}%, WebSearch={100*w_c/n_lv:.1f}%, ToolTOT={100*t_c/n_lv:.1f}%, Enhanced={100*e_c/n_lv:.1f}%")

# ============================================================
# Analysis 4: Unfair comparison (file-dependent questions)
# ============================================================
print(f"\n{'=' * 90}")
print("ANALYSIS 4: UNFAIR COMPARISON (File-Dependent Questions)")
print("=" * 90)

unfair_subset = [q for q in questions_classified if q['has_file']]
n = len(unfair_subset)

e_correct = sum(1 for q in unfair_subset if q['enhanced_correct'])
d_correct = sum(1 for q in unfair_subset if q['direct_correct'])
w_correct = sum(1 for q in unfair_subset if q['websearch_correct'])
t_correct = sum(1 for q in unfair_subset if q['tooltot_correct'])

print(f"\nQuestions with file attachments (n={n}):")
print(f"Baseline strategies CANNOT process these files (no file parsing tools).")
print(f"Enhanced ReAct HAS file parsing, image analysis, and code execution tools.")
print(f"\n  {'Strategy':<25} {'Correct':>8} {'Accuracy':>10} {'Has File Tools':>16}")
print(f"  {'-'*65}")
print(f"  {'Direct':<25} {d_correct:>8} {100*d_correct/n:>9.1f}% {'NO':>16}")
print(f"  {'WebSearch':<25} {w_correct:>8} {100*w_correct/n:>9.1f}% {'NO':>16}")
print(f"  {'ToolTOT':<25} {t_correct:>8} {100*t_correct/n:>9.1f}% {'NO':>16}")
print(f"  {'Enhanced ReAct':<25} {e_correct:>8} {100*e_correct/n:>9.1f}% {'YES':>16}")

# ============================================================
# Analysis 5: Decompose the improvement
# ============================================================
print(f"\n{'=' * 90}")
print("ANALYSIS 5: DECOMPOSING THE IMPROVEMENT")
print("=" * 90)

# Overall
total = len(questions_classified)
e_total = sum(1 for q in questions_classified if q['enhanced_correct'])
w_total = sum(1 for q in questions_classified if q['websearch_correct'])

# Fair subset
fair_n = len(fair_subset)
e_fair = sum(1 for q in fair_subset if q['enhanced_correct'])
w_fair = sum(1 for q in fair_subset if q['websearch_correct'])

# Unfair subset
unfair_n = len(unfair_subset)
e_unfair = sum(1 for q in unfair_subset if q['enhanced_correct'])
w_unfair = sum(1 for q in unfair_subset if q['websearch_correct'])

print(f"\nOverall improvement: {100*e_total/total:.1f}% - {100*w_total/total:.1f}% = +{100*(e_total-w_total)/total:.1f}pp")
print(f"\nDecomposition:")
print(f"  Fair subset (n={fair_n}, {100*fair_n/total:.0f}% of questions):")
print(f"    Enhanced: {100*e_fair/fair_n:.1f}%, WebSearch: {100*w_fair/fair_n:.1f}%, Δ = +{100*(e_fair-w_fair)/fair_n:.1f}pp")
print(f"    Contribution to overall: +{100*(e_fair-w_fair)/total:.1f}pp")
print(f"    → This is the ARCHITECTURE + MODEL improvement (same tool access)")
print(f"\n  Unfair subset (n={unfair_n}, {100*unfair_n/total:.0f}% of questions):")
print(f"    Enhanced: {100*e_unfair/unfair_n:.1f}%, WebSearch: {100*w_unfair/unfair_n:.1f}%, Δ = +{100*(e_unfair-w_unfair)/unfair_n:.1f}pp")
print(f"    Contribution to overall: +{100*(e_unfair-w_unfair)/total:.1f}pp")
print(f"    → This includes BOTH tool advantage AND architecture improvement")

# ============================================================
# Analysis 6: Enhanced ReAct tool usage patterns
# ============================================================
print(f"\n{'=' * 90}")
print("ANALYSIS 6: Enhanced ReAct Tool Usage Patterns")
print("=" * 90)

# For questions that Enhanced got right, what tools were used?
for subset_name, subset in [("Correct answers", [q for q in questions_classified if q['enhanced_correct']]),
                              ("Wrong answers", [q for q in questions_classified if not q['enhanced_correct']])]:
    tools_count = {}
    for q in subset:
        for t in q['enhanced_tools']:
            tools_count[t] = tools_count.get(t, 0) + 1
    n = len(subset)
    print(f"\n{subset_name} (n={n}):")
    for t, c in sorted(tools_count.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({c/n:.1f} per question)")

# ============================================================
# Analysis 7: Questions where baseline got right but enhanced didn't
# ============================================================
print(f"\n{'=' * 90}")
print("ANALYSIS 7: Regression Analysis")
print("=" * 90)

baseline_only = [q for q in questions_classified 
                 if (q['direct_correct'] or q['websearch_correct'] or q['tooltot_correct'])
                 and not q['enhanced_correct']]
enhanced_only = [q for q in questions_classified 
                 if q['enhanced_correct'] 
                 and not q['direct_correct'] and not q['websearch_correct'] and not q['tooltot_correct']]

print(f"\nBaseline correct, Enhanced wrong: {len(baseline_only)} questions")
print(f"Enhanced correct, All baselines wrong: {len(enhanced_only)} questions")

# Save all analysis data for visualization
analysis_data = {
    'total_questions': total,
    'fair_subset': {
        'n': fair_n,
        'enhanced_correct': e_fair,
        'direct_correct': sum(1 for q in fair_subset if q['direct_correct']),
        'websearch_correct': w_fair,
        'tooltot_correct': sum(1 for q in fair_subset if q['tooltot_correct']),
    },
    'unfair_subset': {
        'n': unfair_n,
        'enhanced_correct': e_unfair,
        'direct_correct': sum(1 for q in unfair_subset if q['direct_correct']),
        'websearch_correct': w_unfair,
        'tooltot_correct': sum(1 for q in unfair_subset if q['tooltot_correct']),
    },
    'categories': {},
    'questions': questions_classified,
}

for cat in ['web_search_only', 'file_parsing', 'image_understanding', 'audio_processing']:
    subset = [q for q in questions_classified if q['primary_cat'] == cat]
    if not subset:
        continue
    analysis_data['categories'][cat] = {
        'n': len(subset),
        'enhanced_correct': sum(1 for q in subset if q['enhanced_correct']),
        'direct_correct': sum(1 for q in subset if q['direct_correct']),
        'websearch_correct': sum(1 for q in subset if q['websearch_correct']),
        'tooltot_correct': sum(1 for q in subset if q['tooltot_correct']),
    }

with open('/home/ubuntu/GPTSwarm/experiments/gaia_enhanced_results/fairness_analysis.json', 'w') as f:
    json.dump(analysis_data, f, indent=2, default=str)

print(f"\nAnalysis data saved to fairness_analysis.json")
