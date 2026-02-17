#!/usr/bin/env python3
"""
GAIA Experiment Fairness Analysis v2
Correctly uses 'Solved' field for baseline and 'is_correct' for enhanced.
Classifies questions by required capabilities and decomposes improvement sources.
"""

import json
import os

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

# Build lookups by task_id
enhanced_by_id = {r.get('task_id', ''): r for r in enhanced}
direct_by_id = {r.get('task_id', ''): r for r in direct}
websearch_by_id = {r.get('task_id', ''): r for r in websearch}
tooltot_by_id = {r.get('task_id', ''): r for r in tooltot}

# ============================================================
# Classify each question
# ============================================================
file_ext_map = {
    '.pdf': 'file_parsing', '.xlsx': 'file_parsing', '.xls': 'file_parsing',
    '.csv': 'file_parsing', '.docx': 'file_parsing', '.doc': 'file_parsing',
    '.pptx': 'file_parsing', '.zip': 'file_parsing', '.jsonld': 'file_parsing',
    '.txt': 'file_parsing', '.pdb': 'file_parsing',
    '.png': 'image_understanding', '.jpg': 'image_understanding', '.jpeg': 'image_understanding',
    '.gif': 'image_understanding', '.bmp': 'image_understanding',
    '.mp3': 'audio_processing', '.wav': 'audio_processing', '.mp4': 'audio_processing',
    '.py': 'code_execution', '.js': 'code_execution',
}

questions = []
for q in dataset:
    task_id = q.get('task_id', '')
    file_name = q.get('file_name', '')
    level = q.get('Level', 0)
    
    # Determine file category
    has_file = bool(file_name and file_name.strip())
    if has_file:
        ext = os.path.splitext(file_name)[1].lower()
        file_cat = file_ext_map.get(ext, 'file_parsing')
    else:
        file_cat = 'none'
    
    # Get results (use correct field names!)
    e = enhanced_by_id.get(task_id, {})
    d = direct_by_id.get(task_id, {})
    w = websearch_by_id.get(task_id, {})
    t = tooltot_by_id.get(task_id, {})
    
    questions.append({
        'task_id': task_id,
        'level': level,
        'has_file': has_file,
        'file_name': file_name,
        'file_cat': file_cat,
        'enhanced_correct': e.get('is_correct', False),
        'direct_correct': d.get('Solved', False),
        'websearch_correct': w.get('Solved', False),
        'tooltot_correct': t.get('Solved', False),
        'enhanced_tools': e.get('tools_used', []),
    })

# ============================================================
# Helper
# ============================================================
def print_table(subset, label):
    n = len(subset)
    if n == 0:
        return
    e = sum(1 for q in subset if q['enhanced_correct'])
    d = sum(1 for q in subset if q['direct_correct'])
    w = sum(1 for q in subset if q['websearch_correct'])
    t = sum(1 for q in subset if q['tooltot_correct'])
    best_b = max(d, w, t)
    
    print(f"\n  {label} (n={n}):")
    print(f"  {'Strategy':<30} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Direct (no tools)':<30} {d:>8} {100*d/n:>9.1f}%")
    print(f"  {'WebSearch (DDG+Wiki)':<30} {w:>8} {100*w/n:>9.1f}%")
    print(f"  {'ToolTOT (DDG+Wiki)':<30} {t:>8} {100*t/n:>9.1f}%")
    print(f"  {'Best Baseline':<30} {best_b:>8} {100*best_b/n:>9.1f}%")
    print(f"  {'Enhanced ReAct (5 tools)':<30} {e:>8} {100*e/n:>9.1f}%")
    print(f"  {'Δ (Enhanced - Best Baseline)':<30} {'+' + str(e-best_b):>8} {'+' + f'{100*(e-best_b)/n:.1f}':>9}pp")
    return {'n': n, 'e': e, 'd': d, 'w': w, 't': t, 'best_b': best_b}

# ============================================================
# ANALYSIS 1: Overall
# ============================================================
print("=" * 80)
print("GAIA BENCHMARK FAIRNESS ANALYSIS")
print("=" * 80)

overall = print_table(questions, "OVERALL (all 165 questions)")

# ============================================================
# ANALYSIS 2: By file attachment
# ============================================================
print(f"\n{'=' * 80}")
print("SPLIT BY FILE ATTACHMENT (Fair vs Unfair Comparison)")
print("=" * 80)

no_file = [q for q in questions if not q['has_file']]
has_file = [q for q in questions if q['has_file']]

print("\n--- Questions WITHOUT file attachments ---")
print("  All strategies have web search; comparison is RELATIVELY FAIR")
print("  (Still differs in search quality: DDG+Wiki vs Google SerpAPI)")
nf_stats = print_table(no_file, "No File Attachment")

print(f"\n--- Questions WITH file attachments ---")
print("  Baseline strategies CANNOT process files; comparison is UNFAIR")
print("  Enhanced ReAct has: PDF reader, Excel parser, image OCR, code execution")
hf_stats = print_table(has_file, "Has File Attachment")

# ============================================================
# ANALYSIS 3: By file type
# ============================================================
print(f"\n{'=' * 80}")
print("SPLIT BY FILE TYPE")
print("=" * 80)

for cat, label in [
    ('none', 'No File'),
    ('file_parsing', 'File Parsing (PDF/Excel/CSV/DOCX/ZIP/TXT)'),
    ('image_understanding', 'Image Understanding (PNG/JPG)'),
    ('audio_processing', 'Audio Processing (MP3)'),
    ('code_execution', 'Code Execution (PY)'),
]:
    subset = [q for q in questions if q['file_cat'] == cat]
    if subset:
        print_table(subset, label)

# ============================================================
# ANALYSIS 4: By level, split by file
# ============================================================
print(f"\n{'=' * 80}")
print("BY LEVEL × FILE ATTACHMENT")
print("=" * 80)

for lv in [1, 2, 3]:
    for has_f, f_label in [(False, "no file"), (True, "has file")]:
        subset = [q for q in questions if q['level'] == lv and q['has_file'] == has_f]
        if subset:
            print_table(subset, f"Level {lv}, {f_label}")

# ============================================================
# ANALYSIS 5: Decompose improvement
# ============================================================
print(f"\n{'=' * 80}")
print("IMPROVEMENT DECOMPOSITION")
print("=" * 80)

total_n = len(questions)
e_total = sum(1 for q in questions if q['enhanced_correct'])
best_b_total = sum(1 for q in questions if q['direct_correct'] or q['websearch_correct'] or q['tooltot_correct'])
# Use WebSearch as best single baseline
w_total = sum(1 for q in questions if q['websearch_correct'])

print(f"\nOverall: Enhanced={100*e_total/total_n:.1f}%, Best Single Baseline (WebSearch)={100*w_total/total_n:.1f}%")
print(f"Total improvement: +{100*(e_total-w_total)/total_n:.1f}pp")

# Fair subset contribution
nf_n = len(no_file)
nf_e = sum(1 for q in no_file if q['enhanced_correct'])
nf_w = sum(1 for q in no_file if q['websearch_correct'])

hf_n = len(has_file)
hf_e = sum(1 for q in has_file if q['enhanced_correct'])
hf_w = sum(1 for q in has_file if q['websearch_correct'])

print(f"\nDecomposition:")
print(f"  A. Fair subset (no file, n={nf_n}, {100*nf_n/total_n:.0f}% of questions):")
print(f"     Enhanced: {nf_e}/{nf_n} = {100*nf_e/nf_n:.1f}%")
print(f"     WebSearch: {nf_w}/{nf_n} = {100*nf_w/nf_n:.1f}%")
print(f"     Δ = +{100*(nf_e-nf_w)/nf_n:.1f}pp within subset")
print(f"     Contribution to overall Δ: +{100*(nf_e-nf_w)/total_n:.1f}pp")
print(f"     → Attributable to: MODEL + ARCHITECTURE + SEARCH QUALITY differences")

print(f"\n  B. Unfair subset (has file, n={hf_n}, {100*hf_n/total_n:.0f}% of questions):")
print(f"     Enhanced: {hf_e}/{hf_n} = {100*hf_e/hf_n:.1f}%")
print(f"     WebSearch: {hf_w}/{hf_n} = {100*hf_w/hf_n:.1f}%")
print(f"     Δ = +{100*(hf_e-hf_w)/hf_n:.1f}pp within subset")
print(f"     Contribution to overall Δ: +{100*(hf_e-hf_w)/total_n:.1f}pp")
print(f"     → Attributable to: MODEL + ARCHITECTURE + SEARCH QUALITY + TOOL ADVANTAGE")

# ============================================================
# ANALYSIS 6: What the baselines CAN'T do
# ============================================================
print(f"\n{'=' * 80}")
print("TOOL AVAILABILITY MATRIX")
print("=" * 80)

print(f"""
                          GPTSwarm Baselines        Enhanced ReAct
                          (Direct/WebSearch/TOT)     (gpt-4.1-mini)
  ─────────────────────── ──────────────────────── ──────────────────
  Web Search              DDG + Wikipedia           Google (SerpAPI)
  Webpage Content Read    ✗                         ✓ (readability)
  PDF Parsing             ✗                         ✓ (PyPDF2)
  Excel/CSV Parsing       ✗                         ✓ (openpyxl/csv)
  Word Doc Parsing        ✗                         ✓ (python-docx)
  Image Understanding     ✗                         ✓ (GPT-4.1-mini vision)
  Audio Transcription     ✗                         ✗
  Code Execution          ✗                         ✓ (Python sandbox)
  ZIP Archive Handling    ✗                         ✓
  ─────────────────────── ──────────────────────── ──────────────────
  LLM Model               gpt-5.1-codex-mini        gpt-4.1-mini
  Architecture             3 strategies              Single ReAct agent
  Max reasoning steps      1-5 LLM calls             12 turns
""")

# ============================================================
# ANALYSIS 7: Confounding factors summary
# ============================================================
print(f"{'=' * 80}")
print("CONFOUNDING FACTORS IN CURRENT COMPARISON")
print("=" * 80)

print("""
The current experiment has FOUR confounding factors that prevent attributing
the improvement to any single cause:

  1. TOOL SUITE DIFFERENCE (most impactful for 23% of questions)
     - Enhanced has file parsing, code execution, image analysis
     - Baseline has NONE of these
     - Impact: ~{hf_contrib:.1f}pp of overall improvement comes from file-dependent questions

  2. SEARCH QUALITY DIFFERENCE (affects all questions)
     - Enhanced uses Google via SerpAPI (high quality, comprehensive)
     - Baseline uses DuckDuckGo lite + Wikipedia API (lower quality)
     - Impact: Unknown, but likely significant for factual questions

  3. MODEL DIFFERENCE
     - Enhanced: gpt-4.1-mini (OpenAI, well-tested)
     - Baseline: gpt-5.1-codex-mini (custom API endpoint, potentially unstable)
     - Impact: Unknown, models may differ in reasoning and tool-use capability

  4. ARCHITECTURE DIFFERENCE
     - Enhanced: ReAct with function calling (12 turns, dynamic tool selection)
     - Baseline: Fixed pipelines (Direct=1 call, WebSearch=2 calls, TOT=4-5 calls)
     - Impact: This is what we WANT to measure, but it's confounded by 1-3

FOR A FAIR COMPARISON IN THE AGENTGROUP PAPER:
  → All methods MUST use the SAME model
  → All methods MUST have access to the SAME tool suite
  → All methods MUST use the SAME search engine
  → Only the ARCHITECTURE should differ (single-agent vs multi-agent, graph structure, etc.)
""".format(hf_contrib=100*(hf_e-hf_w)/total_n))

# ============================================================
# ANALYSIS 8: Regression analysis
# ============================================================
print(f"{'=' * 80}")
print("REGRESSION ANALYSIS")
print("=" * 80)

baseline_only = [q for q in questions 
                 if (q['direct_correct'] or q['websearch_correct'] or q['tooltot_correct'])
                 and not q['enhanced_correct']]
enhanced_only = [q for q in questions 
                 if q['enhanced_correct'] 
                 and not q['direct_correct'] and not q['websearch_correct'] and not q['tooltot_correct']]
both_correct = [q for q in questions
                if q['enhanced_correct']
                and (q['direct_correct'] or q['websearch_correct'] or q['tooltot_correct'])]

print(f"\n  Both correct: {len(both_correct)} questions")
print(f"  Enhanced only correct: {len(enhanced_only)} questions")
print(f"  Baseline only correct: {len(baseline_only)} questions")
print(f"  Neither correct: {len(questions) - len(both_correct) - len(enhanced_only) - len(baseline_only)} questions")

if baseline_only:
    print(f"\n  Questions where baseline succeeded but Enhanced failed:")
    for q in baseline_only[:10]:
        which = []
        if q['direct_correct']: which.append('D')
        if q['websearch_correct']: which.append('W')
        if q['tooltot_correct']: which.append('T')
        print(f"    Level {q['level']}, file={q['has_file']}, solved by: {'+'.join(which)}")

# Save analysis data
analysis = {
    'overall': {'enhanced': e_total, 'direct': sum(1 for q in questions if q['direct_correct']),
                'websearch': w_total, 'tooltot': sum(1 for q in questions if q['tooltot_correct']),
                'total': total_n},
    'no_file': {'enhanced': nf_e, 'direct': sum(1 for q in no_file if q['direct_correct']),
                'websearch': nf_w, 'tooltot': sum(1 for q in no_file if q['tooltot_correct']),
                'total': nf_n},
    'has_file': {'enhanced': hf_e, 'direct': sum(1 for q in has_file if q['direct_correct']),
                 'websearch': hf_w, 'tooltot': sum(1 for q in has_file if q['tooltot_correct']),
                 'total': hf_n},
    'by_level_no_file': {},
    'by_level_has_file': {},
}

for lv in [1, 2, 3]:
    for has_f, key in [(False, 'by_level_no_file'), (True, 'by_level_has_file')]:
        subset = [q for q in questions if q['level'] == lv and q['has_file'] == has_f]
        if subset:
            analysis[key][str(lv)] = {
                'enhanced': sum(1 for q in subset if q['enhanced_correct']),
                'direct': sum(1 for q in subset if q['direct_correct']),
                'websearch': sum(1 for q in subset if q['websearch_correct']),
                'tooltot': sum(1 for q in subset if q['tooltot_correct']),
                'total': len(subset),
            }

with open('/home/ubuntu/GPTSwarm/experiments/gaia_enhanced_results/fairness_analysis_v2.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"\nAnalysis data saved to fairness_analysis_v2.json")
