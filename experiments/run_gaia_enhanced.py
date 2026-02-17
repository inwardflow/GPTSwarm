#!/usr/bin/env python3
"""
GAIA Enhanced Experiment Runner (ReAct Agent with OpenAI SDK)
=============================================================
Runs the GAIA benchmark using enhanced tools and the OpenAI-compatible API.

Architecture:
  - Tools: search_web, read_file, run_python, analyze_image, fetch_webpage
  - Agent: ReAct loop (Thought → Action → Observation → ... → FINAL ANSWER)
  - API: OpenAI SDK (works with any OpenAI-compatible endpoint)

Features:
  - Robust retry with exponential backoff
  - Checkpoint/resume: saves results after each question
  - Comprehensive file handling: PDF, Excel, CSV, DOCX, PPTX, ZIP, images, audio

Usage:
  python3 run_gaia_enhanced.py --dataset path/to/gaia.json [--start 0] [--end -1]
  python3 run_gaia_enhanced.py --dataset path/to/gaia.json --resume result.json
"""

import os
import sys
import json
import time
import argparse
import re
import warnings
import random

warnings.filterwarnings("ignore")

# Import tool manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp_client import DirectToolManager

# ============================================================
# Configuration
# ============================================================
MODEL = os.environ.get("GAIA_MODEL", "gpt-4.1-mini")
ATTACHMENT_DIR = os.environ.get("GAIA_ATTACHMENT_DIR",
    "/home/ubuntu/GPTSwarm/datasets/gaia/files")

# ============================================================
# OpenAI Client
# ============================================================
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY and OPENAI_BASE_URL from env


# ============================================================
# System Prompt (ReAct format)
# ============================================================
SYSTEM_PROMPT = """You are a precise research assistant solving questions from the GAIA benchmark.
You MUST use tools to gather information before answering. Do NOT guess or rely on memory alone.

Available tools:
- search_web(query): Search the web using DuckDuckGo. Returns titles, URLs, and snippets.
- read_file(file_path): Read files: PDF, Excel (.xlsx/.xls), CSV, DOCX, PPTX, ZIP, audio (MP3/WAV via transcription), images (OCR), text, JSON, Python, etc.
- run_python(code): Execute Python code. Available packages: numpy, pandas, openpyxl, matplotlib, scipy, sympy, etc. Always print() results.
- fetch_webpage(url): Fetch and extract text content from a URL.
- analyze_image(file_path): Perform OCR on an image file to extract text.

Response format — follow this EXACTLY:

Thought: <your step-by-step reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON object with the tool's parameters>

After receiving an Observation (tool result), continue with another Thought/Action cycle, or finish:

Thought: I now have enough information to answer.
FINAL ANSWER: <your concise answer>

IMPORTANT RULES:
1. Output EXACTLY ONE Action per turn. Wait for the Observation before continuing.
2. For questions with attached files, ALWAYS read the file first using read_file.
3. For math/computation, ALWAYS use run_python with print() to get results.
4. For factual questions, use search_web then fetch_webpage for details.
5. The final answer should be concise: a number, name, date, or short phrase.
6. For numbers, match the requested format (commas, decimals, units).
7. Do NOT include explanations in the FINAL ANSWER line — just the answer.
8. If a question asks about an attached file, you MUST read it before answering."""


# ============================================================
# API Call with Retry
# ============================================================
def call_api(messages, max_retries=8):
    """Call the Chat Completions API with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=2048,
                timeout=90,
            )
            content = resp.choices[0].message.content
            if content:
                return content
            # Empty content
            wait = min(30, 5 * (attempt + 1))
            print(f"      Empty content, waiting {wait}s... (attempt {attempt+1}/{max_retries})", flush=True)
            time.sleep(wait)
        except Exception as e:
            err_str = str(e)
            if "rate" in err_str.lower() or "429" in err_str:
                wait = min(120, 30 * (attempt + 1)) + random.uniform(0, 5)
                print(f"      Rate limited, waiting {wait:.0f}s... (attempt {attempt+1}/{max_retries})", flush=True)
            else:
                wait = min(60, 10 * (attempt + 1)) + random.uniform(0, 3)
                print(f"      API error: {err_str[:100]}, waiting {wait:.0f}s... (attempt {attempt+1}/{max_retries})", flush=True)
            time.sleep(wait)

    print(f"      API failed after {max_retries} retries", flush=True)
    return None


# ============================================================
# ReAct Agent Loop
# ============================================================
def parse_action(text):
    """Parse Action and Action Input from model output."""
    # Try standard format: Action: tool_name\nAction Input: {...}
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)

    if action_match and input_match:
        tool_name = action_match.group(1)
        try:
            args = json.loads(input_match.group(1))
            return tool_name, args
        except json.JSONDecodeError:
            raw = input_match.group(1).replace("'", '"')
            try:
                args = json.loads(raw)
                return tool_name, args
            except Exception:
                pass

    # Try alternative format: Action: tool_name({"key": "value"})
    alt_match = re.search(r"Action:\s*(\w+)\s*\(\s*(\{.*?\})\s*\)", text, re.DOTALL)
    if alt_match:
        tool_name = alt_match.group(1)
        try:
            args = json.loads(alt_match.group(2))
            return tool_name, args
        except Exception:
            pass

    # Try multiline JSON
    if action_match:
        tool_name = action_match.group(1)
        # Find JSON block after "Action Input:"
        json_match = re.search(r"Action Input:\s*(\{[^}]+\})", text, re.DOTALL)
        if json_match:
            try:
                args = json.loads(json_match.group(1))
                return tool_name, args
            except Exception:
                pass
        # Try with larger block
        json_match = re.search(r"Action Input:\s*```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                args = json.loads(json_match.group(1))
                return tool_name, args
            except Exception:
                pass

    return None, None


def extract_final_answer(text):
    """Extract FINAL ANSWER from model output."""
    match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        answer = answer.rstrip(".")
        return answer
    return None


def run_agent(question, file_path, tool_mgr, max_turns=15, timeout=300):
    """Run the ReAct agent loop for a single GAIA question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_msg = question
    if file_path and os.path.exists(file_path):
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        user_msg += f"\n\n[Attached file: {file_path}]"
        if ext in ("png", "jpg", "jpeg", "gif", "bmp"):
            user_msg += "\n[This is an image file. Use analyze_image or read_file to examine it.]"
        elif ext in ("mp3", "wav", "mp4", "webm"):
            user_msg += "\n[This is an audio file. Use read_file to transcribe it.]"
        elif ext in ("xlsx", "xls"):
            user_msg += "\n[This is an Excel file. Use read_file to examine it, or run_python with pandas for analysis.]"
        elif ext == "pdf":
            user_msg += "\n[This is a PDF file. Use read_file to extract text.]"
        elif ext == "zip":
            user_msg += "\n[This is a ZIP archive. Use read_file to list and extract contents.]"
        elif ext == "pptx":
            user_msg += "\n[This is a PowerPoint file. Use read_file to extract text.]"
        elif ext == "docx":
            user_msg += "\n[This is a Word document. Use read_file to extract text.]"
        elif ext == "csv":
            user_msg += "\n[This is a CSV file. Use read_file to examine it, or run_python with pandas.]"
        elif ext == "py":
            user_msg += "\n[This is a Python file. Use read_file to read it, or run_python to execute it.]"

    messages.append({"role": "user", "content": user_msg})

    start_time = time.time()
    tools_used = []
    api_failures = 0

    for turn in range(max_turns):
        if time.time() - start_time > timeout:
            return None, turn, "timeout", tools_used

        response = call_api(messages)

        if response is None:
            api_failures += 1
            if api_failures >= 3:
                return None, turn, "api_down", tools_used
            time.sleep(10)
            continue

        api_failures = 0

        # Check for final answer first
        final_answer = extract_final_answer(response)
        if final_answer:
            return final_answer, turn + 1, "success", tools_used

        # Parse action
        tool_name, tool_args = parse_action(response)

        if tool_name and tool_args:
            print(f"      Turn {turn}: {tool_name}({json.dumps(tool_args)[:80]})", flush=True)
            tools_used.append(tool_name)

            try:
                result = tool_mgr.call_tool(tool_name, tool_args)
            except Exception as e:
                result = f"Tool error: {e}"

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {str(result)[:6000]}"})
        else:
            # No valid action found, ask model to try again
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "Please either use a tool (Action: tool_name / Action Input: {args}) or provide your FINAL ANSWER: <answer>"
            })

    # Try to extract answer from conversation history
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            answer = extract_final_answer(msg["content"])
            if answer:
                return answer, max_turns, "extracted", tools_used

    return None, max_turns, "max_turns", tools_used


# ============================================================
# Scoring
# ============================================================
def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    for prefix in ["the ", "a ", "an "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = s.rstrip(".,;:!?")
    s = " ".join(s.split())
    return s


def check_answer(predicted, expected):
    if predicted is None:
        return False
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    if pred_norm == exp_norm:
        return True
    if exp_norm in pred_norm or pred_norm in exp_norm:
        return True

    try:
        pred_num = float(pred_norm.replace(",", "").replace("$", "").replace("%", ""))
        exp_num = float(exp_norm.replace(",", "").replace("$", "").replace("%", ""))
        if abs(pred_num - exp_num) < 0.01:
            return True
    except (ValueError, TypeError):
        pass

    return False


# ============================================================
# Checkpoint Management
# ============================================================
def save_checkpoint(results, output_path):
    """Save results to checkpoint file atomically."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def load_checkpoint(path):
    """Load results from checkpoint file."""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GAIA Enhanced Experiment Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Path to GAIA dataset JSON")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=-1, help="End question index (-1 for all)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per question (seconds)")
    parser.add_argument("--max-turns", type=int, default=15, help="Max agent turns per question")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    parser.add_argument("--resume", type=str, default="", help="Resume from existing results file")
    parser.add_argument("--model", type=str, default="", help="Override model name")
    args = parser.parse_args()

    global MODEL, client
    if args.model:
        MODEL = args.model

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)

    end = args.end if args.end > 0 else len(dataset)
    questions = dataset[args.start:end]

    # Output path
    if not args.output:
        ts = time.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs("gaia_enhanced_results", exist_ok=True)
        args.output = f"gaia_enhanced_results/gaia_enhanced_{MODEL}_{ts}.json"

    # Determine resume path
    resume_path = args.resume or args.output

    print(f"{'='*60}", flush=True)
    print(f"GAIA Enhanced Experiment (ReAct Agent)", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Questions: {len(questions)} [{args.start}:{end}] of {len(dataset)}", flush=True)
    print(f"Timeout: {args.timeout}s, Max turns: {args.max_turns}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Resume from: {resume_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # Quick API health check
    print(f"\nChecking API health...", flush=True)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            timeout=15,
        )
        print(f"  API healthy: {resp.choices[0].message.content}", flush=True)
    except Exception as e:
        print(f"  API check failed: {e}", flush=True)
        print(f"  Continuing anyway...", flush=True)

    # Initialize tool manager
    tool_mgr = DirectToolManager()
    print(f"Tools: {list(tool_mgr.tools.keys())}", flush=True)

    # Load checkpoint for resume
    existing_results = load_checkpoint(resume_path)
    completed_ids = {r["task_id"] for r in existing_results}
    if existing_results:
        print(f"Resuming: {len(existing_results)} questions already completed", flush=True)

    results = list(existing_results)
    correct = sum(1 for r in results if r.get("is_correct"))
    total = len(results)

    experiment_start = time.time()

    for i, q in enumerate(questions):
        idx = args.start + i
        question = q.get("Question", q.get("question", ""))
        expected = q.get("Final answer", q.get("final_answer", q.get("answer", q.get("GT", ""))))
        level = q.get("Level", q.get("level", "?"))
        file_name = q.get("file_name", "")
        task_id = q.get("task_id", f"q_{idx}")

        if task_id in completed_ids:
            continue

        print(f"\n[{idx+1}/{len(dataset)}] Level {level}: {question[:100]}...", flush=True)

        file_path = None
        if file_name:
            file_path = os.path.join(ATTACHMENT_DIR, file_name)
            if not os.path.exists(file_path):
                print(f"    WARNING: Attachment not found: {file_path}", flush=True)
                file_path = None
            else:
                print(f"    Attachment: {file_name}", flush=True)

        start_time = time.time()
        try:
            answer, turns, status, tools_used = run_agent(
                question, file_path=file_path, tool_mgr=tool_mgr,
                max_turns=args.max_turns, timeout=args.timeout
            )
        except Exception as e:
            print(f"    Exception: {e}", flush=True)
            answer, turns, status, tools_used = None, 0, f"error: {e}", []

        elapsed = time.time() - start_time

        is_correct = check_answer(answer, expected)

        if is_correct:
            correct += 1
        total += 1

        result = {
            "task_id": task_id,
            "question": question[:200],
            "level": level,
            "expected": expected,
            "predicted": answer,
            "is_correct": is_correct,
            "turns": turns,
            "status": status,
            "time": round(elapsed, 1),
            "has_file": bool(file_name),
            "file_type": file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "",
            "tools_used": tools_used
        }
        results.append(result)

        mark = "✓" if is_correct else "✗"
        print(f"    {mark} Expected: {expected} | Predicted: {answer}", flush=True)
        print(f"    Status: {status}, Turns: {turns}, Time: {elapsed:.1f}s", flush=True)
        print(f"    Tools: {', '.join(set(tools_used)) if tools_used else 'none'}", flush=True)
        print(f"    Running Accuracy: {correct}/{total} ({100*correct/total:.1f}%)", flush=True)

        # Save checkpoint after each question
        save_checkpoint(results, args.output)

        # Small delay between questions to avoid rate limiting
        time.sleep(1)

    total_time = time.time() - experiment_start

    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT COMPLETE", flush=True)
    print(f"Model: {MODEL}", flush=True)
    if total > 0:
        print(f"Total: {total}, Correct: {correct}, Accuracy: {100*correct/total:.1f}%", flush=True)
    else:
        print("No questions answered", flush=True)
    print(f"Total time: {total_time/3600:.1f}h", flush=True)

    # Per-level stats
    level_stats = {}
    for r in results:
        lv = str(r.get("level", "?"))
        if lv not in level_stats:
            level_stats[lv] = {"correct": 0, "total": 0}
        level_stats[lv]["total"] += 1
        if r["is_correct"]:
            level_stats[lv]["correct"] += 1

    for lv in sorted(level_stats.keys()):
        s = level_stats[lv]
        print(f"  Level {lv}: {s['correct']}/{s['total']} ({100*s['correct']/s['total']:.1f}%)", flush=True)

    # Tool usage stats
    tool_counts = {}
    for r in results:
        for t in r.get("tools_used", []):
            tool_counts[t] = tool_counts.get(t, 0) + 1
    print(f"\nTool usage:", flush=True)
    for t, c in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} calls", flush=True)

    # File-type stats
    file_stats = {}
    for r in results:
        ft = r.get("file_type", "")
        if ft:
            if ft not in file_stats:
                file_stats[ft] = {"correct": 0, "total": 0}
            file_stats[ft]["total"] += 1
            if r["is_correct"]:
                file_stats[ft]["correct"] += 1
    if file_stats:
        print(f"\nFile type accuracy:", flush=True)
        for ft, s in sorted(file_stats.items()):
            print(f"  .{ft}: {s['correct']}/{s['total']} ({100*s['correct']/s['total']:.1f}%)", flush=True)

    print(f"\nResults saved to: {args.output}", flush=True)
    tool_mgr.stop_all()


if __name__ == "__main__":
    main()
