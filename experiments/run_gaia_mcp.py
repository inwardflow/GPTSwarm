#!/usr/bin/env python3
"""
GAIA MCP Experiment Runner (ReAct Agent)
=========================================
Runs the GAIA benchmark using MCP-based tool infrastructure with a ReAct-style
prompt-based tool calling loop.

Architecture:
  - Tools: Defined in mcp.json, implemented in mcp_tools_server.py / mcp_client.py
  - Agent: ReAct loop (Thought → Action → Observation → ... → FINAL ANSWER)
  - API: Chat Completions via httpx (no native tool calling required)

Features:
  - Robust retry with exponential backoff (handles 429, 503, 521, empty responses)
  - Checkpoint/resume: saves results after each question, can resume from partial runs
  - API health check before starting

Configuration:
  export CODEX_API_KEY="your-api-key"
  export CODEX_BASE_URL="https://your-api-endpoint"
  export CODEX_MODEL="gpt-5.1-codex-mini"

Usage:
  python3 run_gaia_mcp.py --dataset path/to/gaia.json [--start 0] [--end -1]
  python3 run_gaia_mcp.py --dataset path/to/gaia.json --resume result.json  # resume
"""

import os
import sys
import json
import time
import argparse
import re
import warnings

warnings.filterwarnings("ignore")

# Import tool manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp_client import DirectToolManager

# ============================================================
# Configuration
# ============================================================
API_KEY = os.environ.get("CODEX_API_KEY", "")
BASE_URL = os.environ.get("CODEX_BASE_URL", "")
MODEL = os.environ.get("CODEX_MODEL", "gpt-5.1-codex-mini")
ATTACHMENT_DIR = os.environ.get("GAIA_ATTACHMENT_DIR", "/tmp/gaia_data/data/gaia/validation")


# ============================================================
# API Call via subprocess curl (most reliable)
# ============================================================
import subprocess

import tempfile

def curl_post(url, payload, timeout=120):
    """
    Make a POST request using curl via os.popen + temp file for payload.
    
    Key insight: subprocess.run with capture_output=True and os.system both
    cause TLS timeouts with this API proxy. Only os.popen (which reads output
    via a shell pipe) works reliably, matching direct shell curl behavior.
    
    We write the payload to a temp file to avoid shell escaping issues with
    complex JSON, and read the response via os.popen pipe.
    """
    tmp_in = None
    try:
        # Write payload to temp file to avoid shell escaping issues
        tmp_in = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        tmp_in.write(json.dumps(payload))
        tmp_in.close()

        cmd = (
            f'curl -s -m {timeout} -X POST "{url}" '
            f'-H "Content-Type: application/json" '
            f'-H "Authorization: Bearer {API_KEY}" '
            f'-d @{tmp_in.name}'
        )
        stream = os.popen(cmd)
        text = stream.read().strip()
        stream.close()

        if text:
            return json.loads(text)
        return None
    except json.JSONDecodeError:
        return None
    except Exception:
        return None
    finally:
        if tmp_in:
            try: os.unlink(tmp_in.name)
            except: pass


# ============================================================
# API Health Check
# ============================================================
def api_health_check(max_wait=600, interval=30):
    """
    Wait until the API is healthy. Returns True if healthy, False if timed out.
    Tries a simple request every `interval` seconds for up to `max_wait` seconds.
    """
    url = BASE_URL.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
    }

    start = time.time()
    attempt = 0
    while time.time() - start < max_wait:
        attempt += 1
        try:
            data = curl_post(url, payload, timeout=30)
            if data and "choices" in data:
                print(f"  API healthy (attempt {attempt})", flush=True)
                return True
            print(f"  API not ready: {data} (attempt {attempt})", flush=True)
        except Exception as e:
            print(f"  API check failed: {e} (attempt {attempt})", flush=True)
        time.sleep(interval)

    print(f"  API health check timed out after {max_wait}s", flush=True)
    return False


# ============================================================
# System Prompt (ReAct format)
# ============================================================
SYSTEM_PROMPT = """You are a precise research assistant solving questions from the GAIA benchmark.
You MUST use tools to gather information before answering. Do NOT guess or rely on memory alone.

Available tools:
- search_web(query): Search the web using DuckDuckGo. Returns titles, URLs, and snippets.
- read_file(file_path): Read files: PDF, Excel (.xlsx), CSV, DOCX, ZIP, audio (MP3/WAV), images (OCR), text, JSON, Python, etc.
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
2. For questions with attached files, ALWAYS read the file first.
3. For math/computation, ALWAYS use run_python with print() to get results.
4. For factual questions, use search_web then fetch_webpage if needed.
5. The final answer should be concise: a number, name, date, or short phrase.
6. For numbers, match the requested format (commas, decimals, units).
7. Do NOT include explanations in the FINAL ANSWER line — just the answer."""


# ============================================================
# API Call with Aggressive Retry (via curl)
# ============================================================
def call_api(messages, max_retries=12):
    """
    Call the Chat Completions API using subprocess curl with aggressive retry logic.
    Uses curl because Python HTTP clients (httpx, requests) intermittently fail
    with read timeouts on this API proxy, while curl works reliably.
    """
    import random

    url = BASE_URL.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }

    for attempt in range(max_retries):
        try:
            data = curl_post(url, payload, timeout=120)

            if data is None:
                # Empty response or timeout
                wait = min(120, 15 * (attempt + 1)) + random.uniform(0, 5)
                print(f"      Empty/timeout response, waiting {wait:.0f}s... (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait)
                continue

            if "error" in data:
                error_msg = str(data.get("error", ""))[:200]
                wait = min(120, 20 * (attempt + 1)) + random.uniform(0, 5)
                print(f"      API error: {error_msg}, waiting {wait:.0f}s... (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait)
                continue

            if "choices" in data:
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if content:
                    return content
                # Content is None/empty
                wait = min(60, 10 * (attempt + 1))
                print(f"      Empty content in response, waiting {wait}s... (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait)
                continue

            # Unknown response format
            wait = min(60, 10 * (attempt + 1))
            print(f"      Unknown response format: {str(data)[:100]}, waiting {wait}s... (attempt {attempt+1}/{max_retries})", flush=True)
            time.sleep(wait)

        except Exception as e:
            wait = min(120, 15 * (2 ** min(attempt, 4))) + random.uniform(0, 5)
            print(f"      API exception: {e}, waiting {wait:.0f}s... (attempt {attempt+1}/{max_retries})", flush=True)
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

    # Try to extract just the action name if input is on the same line
    if action_match:
        tool_name = action_match.group(1)
        json_match = re.search(r"Action Input[:\s]*(\{[^}]+\})", text, re.DOTALL)
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
            user_msg += "\n[This is an image file. Use analyze_image to examine it, or read_file for OCR.]"
        elif ext in ("mp3", "wav", "mp4", "webm"):
            user_msg += "\n[This is an audio file. Use read_file to transcribe it.]"
        elif ext in ("xlsx", "xls"):
            user_msg += "\n[This is an Excel file. Use read_file to examine it, or run_python with pandas for analysis.]"
        elif ext == "pdf":
            user_msg += "\n[This is a PDF file. Use read_file to extract text.]"
        elif ext == "zip":
            user_msg += "\n[This is a ZIP archive. Use read_file to list and extract contents.]"

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
                # API is persistently down, wait for recovery
                print(f"      API persistently failing, running health check...", flush=True)
                if api_health_check(max_wait=300, interval=30):
                    api_failures = 0
                    continue
                else:
                    return None, turn, "api_down", tools_used
            continue

        api_failures = 0  # Reset on success

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
    parser = argparse.ArgumentParser(description="GAIA MCP Experiment Runner (ReAct Agent)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to GAIA dataset JSON")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=-1, help="End question index (-1 for all)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per question (seconds)")
    parser.add_argument("--max-turns", type=int, default=15, help="Max agent turns per question")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    parser.add_argument("--resume", type=str, default="", help="Resume from existing results file")
    parser.add_argument("--health-check-wait", type=int, default=1800,
                        help="Max seconds to wait for API health before starting (default: 1800)")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)

    end = args.end if args.end > 0 else len(dataset)
    questions = dataset[args.start:end]

    # Output path
    if not args.output:
        ts = time.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs("result/eval", exist_ok=True)
        args.output = f"result/eval/gaia_mcp_{MODEL}_{ts}.json"

    # Determine resume path
    resume_path = args.resume or args.output

    print(f"{'='*60}", flush=True)
    print(f"GAIA MCP Experiment (ReAct Agent)", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Questions: {len(questions)} [{args.start}:{end}] of {len(dataset)}", flush=True)
    print(f"Timeout: {args.timeout}s, Max turns: {args.max_turns}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Resume from: {resume_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # API health check before starting
    print(f"\nChecking API health...", flush=True)
    if not api_health_check(max_wait=args.health_check_wait, interval=30):
        print(f"ERROR: API not available after {args.health_check_wait}s. Exiting.", flush=True)
        sys.exit(1)

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
    consecutive_api_failures = 0

    for i, q in enumerate(questions):
        idx = args.start + i
        question = q.get("question", q.get("Question", ""))
        expected = q.get("answer", q.get("Final answer", q.get("final_answer", "")))
        level = q.get("level", q.get("Level", "?"))
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

        # Handle persistent API failures
        if status == "api_down":
            consecutive_api_failures += 1
            print(f"    API down (consecutive: {consecutive_api_failures}). Waiting for recovery...", flush=True)
            if consecutive_api_failures >= 3:
                print(f"    API persistently down. Running extended health check (30 min)...", flush=True)
                if not api_health_check(max_wait=1800, interval=60):
                    print(f"    API not recovered. Saving checkpoint and exiting.", flush=True)
                    save_checkpoint(results, args.output)
                    sys.exit(2)
            # Retry this question after recovery
            if api_health_check(max_wait=600, interval=30):
                consecutive_api_failures = 0
                print(f"    API recovered. Retrying question...", flush=True)
                start_time = time.time()
                try:
                    answer, turns, status, tools_used = run_agent(
                        question, file_path=file_path, tool_mgr=tool_mgr,
                        max_turns=args.max_turns, timeout=args.timeout
                    )
                except Exception as e:
                    answer, turns, status, tools_used = None, 0, f"error: {e}", []
                elapsed = time.time() - start_time
        else:
            consecutive_api_failures = 0

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
        print(f"    {mark} Answer: {answer} | Expected: {expected} | {elapsed:.1f}s | {turns} turns", flush=True)
        print(f"    Tools: {', '.join(set(tools_used)) if tools_used else 'none'}", flush=True)
        print(f"    Running Accuracy: {correct}/{total} ({100*correct/total:.1f}%)", flush=True)

        # Save checkpoint after each question
        save_checkpoint(results, args.output)

    total_time = time.time() - experiment_start

    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT COMPLETE", flush=True)
    print(f"Total: {total}, Correct: {correct}, Accuracy: {100*correct/total:.1f}%" if total > 0 else "No questions answered", flush=True)
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
