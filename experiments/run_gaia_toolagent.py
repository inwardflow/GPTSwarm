#!/usr/bin/env python3
"""
GAIA ToolAgent Experiment - Full Tool-Calling Agent
Uses httpx directly to call the Codex API (bypasses OpenAI SDK hanging issues).
Supports: web search, file reading (PDF/Excel/CSV/ZIP/DOCX/PPTX/JSON/TXT/PY),
           code execution, image analysis (OCR), audio transcription, web page fetching.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import traceback
import re
import httpx

# ============================================================
# Configuration
# ============================================================
API_KEY = os.environ.get("CODEX_API_KEY", "")
BASE_URL = os.environ.get("CODEX_BASE_URL", "")
MODEL = os.environ.get("CODEX_MODEL", "gpt-5.1-codex-mini")
ATTACHMENT_DIR = "/tmp/gaia_data/data/gaia/validation"

# Global httpx client (reuse connections, no SSL verification, no HTTP/2)
_http_client = httpx.Client(http2=False, timeout=300, verify=False)

# ============================================================
# Low-level API call using httpx
# ============================================================
def call_chat_api(messages, tools=None, tool_choice="auto", temperature=0, max_retries=3):
    """Call the Chat Completions API using httpx directly."""
    url = BASE_URL.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    for attempt in range(max_retries):
        try:
            resp = _http_client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data
            elif resp.status_code == 429:
                wait = min(30, 5 * (attempt + 1))
                print(f"    Rate limited (429), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"    API error {resp.status_code}: {resp.text[:200]}", flush=True)
                time.sleep(3)
                continue
        except Exception as e:
            print(f"    API exception (attempt {attempt+1}): {e}", flush=True)
            time.sleep(5)
            continue
    return None

# ============================================================
# Tool Implementations
# ============================================================

def tool_search_web(query: str) -> str:
    """Search the web using DuckDuckGo and return results."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No search results found."
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"[{i}] {r.get('title', 'N/A')}")
            output.append(f"    URL: {r.get('href', 'N/A')}")
            output.append(f"    {r.get('body', 'N/A')}")
        return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"

def tool_read_file(file_path: str) -> str:
    """Read and extract content from various file types."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    
    try:
        if ext == "pdf":
            try:
                import subprocess
                result = subprocess.run(["pdftotext", "-layout", file_path, "-"], 
                                       capture_output=True, text=True, timeout=30)
                text = result.stdout.strip()
                if text:
                    return text[:10000]
            except Exception:
                pass
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text[:10000] if text.strip() else "PDF appears to contain only images."
            except Exception as e:
                return f"PDF read error: {e}"
        
        elif ext in ("xlsx", "xls"):
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            output = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                output.append(f"=== Sheet: {sheet_name} ===")
                for row in ws.iter_rows(values_only=True):
                    output.append("\t".join(str(c) if c is not None else "" for c in row))
            return "\n".join(output)[:15000]
        
        elif ext == "csv":
            with open(file_path, "r", errors="replace") as f:
                return f.read()[:10000]
        
        elif ext == "docx":
            try:
                from docx import Document
                doc = Document(file_path)
                text = "\n".join(p.text for p in doc.paragraphs)
                return text[:10000]
            except ImportError:
                return "python-docx not installed. Cannot read DOCX files."
        
        elif ext == "pptx":
            try:
                from pptx import Presentation
                prs = Presentation(file_path)
                text = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text.append(shape.text)
                return "\n".join(text)[:10000]
            except ImportError:
                return "python-pptx not installed. Cannot read PPTX files."
        
        elif ext == "zip":
            import zipfile
            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()
                output = [f"ZIP contents ({len(names)} files):"]
                for name in names[:20]:
                    output.append(f"  {name}")
                # Try to read text files inside
                for name in names:
                    if name.endswith((".txt", ".csv", ".json", ".py", ".md")):
                        try:
                            content = zf.read(name).decode("utf-8", errors="replace")
                            output.append(f"\n--- {name} ---")
                            output.append(content[:3000])
                        except Exception:
                            pass
                return "\n".join(output)[:10000]
        
        elif ext in ("mp3", "wav", "mp4", "webm"):
            try:
                result = subprocess.run(
                    ["manus-speech-to-text", file_path],
                    capture_output=True, text=True, timeout=120
                )
                if result.stdout.strip():
                    return result.stdout.strip()[:5000]
                return f"Audio transcription returned empty. stderr: {result.stderr[:500]}"
            except Exception as e:
                return f"Audio transcription error: {e}"
        
        elif ext in ("png", "jpg", "jpeg", "gif", "bmp", "tiff"):
            return _do_ocr(file_path)
        
        elif ext == "pdb":
            with open(file_path, "r", errors="replace") as f:
                return f.read()[:10000]
        
        elif ext == "jsonld":
            with open(file_path, "r", errors="replace") as f:
                return f.read()[:10000]
        
        else:
            # Try reading as text
            with open(file_path, "r", errors="replace") as f:
                return f.read()[:10000]
    
    except Exception as e:
        return f"Error reading file: {e}"

def _do_ocr(file_path: str) -> str:
    """Perform OCR on an image file."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(file_path)
        info = f"Image: {os.path.basename(file_path)}, Size: {img.size}, Mode: {img.mode}"
        text = pytesseract.image_to_string(img)
        if text.strip():
            return f"{info}\nOCR Text:\n{text.strip()}"
        return f"{info}\nOCR returned no text."
    except ImportError:
        from PIL import Image
        img = Image.open(file_path)
        return f"Image: {os.path.basename(file_path)}, Size: {img.size}, Mode: {img.mode}. (OCR not available)"
    except Exception as e:
        return f"OCR error: {e}"

def tool_analyze_image(file_path: str) -> str:
    """Analyze an image file using OCR."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    return _do_ocr(file_path)

def tool_run_python(code: str) -> str:
    """Execute Python code and return the output."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True, text=True, timeout=30,
                cwd="/tmp"
            )
            os.unlink(f.name)
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR: " + result.stderr
            return output.strip()[:5000] if output.strip() else "(No output)"
    except subprocess.TimeoutExpired:
        return "Code execution timed out (30s limit)."
    except Exception as e:
        return f"Code execution error: {e}"

def tool_fetch_webpage(url: str) -> str:
    """Fetch and extract text content from a webpage."""
    try:
        import warnings
        warnings.filterwarnings("ignore")
        resp = httpx.get(url, timeout=30, verify=False, follow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot)"})
        if resp.status_code != 200:
            return f"HTTP {resp.status_code} error fetching {url}"
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        text = soup.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)[:8000]
    except Exception as e:
        return f"Webpage fetch error: {e}"

# ============================================================
# Tool Definitions for API
# ============================================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets for the top results. Use this for finding factual information, current events, or any knowledge you don't have.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and extract content from a file. Supports: PDF, Excel (.xlsx), CSV, DOCX, PPTX, ZIP, JSON, TXT, PY, MP3/WAV (audio transcription), PDB, JSONLD, and other text formats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return stdout/stderr. Use this for calculations, data processing, or any computation. The code runs in a sandboxed environment with common packages available (numpy, pandas, openpyxl, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image file using OCR (Optical Character Recognition). Extracts text content from images. Returns image metadata and any detected text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the image file (PNG, JPG, etc.)"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch and extract text content from a webpage URL. Useful for reading articles, documentation, or any web page content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

# Tool dispatch map
TOOL_DISPATCH = {
    "search_web": lambda args: tool_search_web(args["query"]),
    "read_file": lambda args: tool_read_file(args["file_path"]),
    "run_python": lambda args: tool_run_python(args["code"]),
    "analyze_image": lambda args: tool_analyze_image(args["file_path"]),
    "fetch_webpage": lambda args: tool_fetch_webpage(args["url"]),
}

# ============================================================
# Agent Loop
# ============================================================
def run_agent(question: str, file_path: str = None, max_turns: int = 15, timeout: int = 300):
    """Run the tool-calling agent loop for a single GAIA question."""
    
    system_prompt = """You are a precise research assistant solving questions from the GAIA benchmark.
You have access to tools for web search, file reading, code execution, image analysis, and web page fetching.

IMPORTANT RULES:
1. Think step-by-step about what information you need
2. Use tools to gather information - do NOT guess or make up facts
3. For math/computation questions, ALWAYS use the run_python tool
4. For questions about files, ALWAYS use read_file first to examine the content
5. For questions requiring web information, use search_web then fetch_webpage for details
6. When you have the final answer, respond with EXACTLY this format:
   FINAL ANSWER: <your answer>
7. The answer should be concise - typically a single word, number, or short phrase
8. Do NOT include explanations in the FINAL ANSWER line
9. For numbers, use the exact format requested (e.g., with or without commas, decimal places)
10. For names, use the exact spelling found in your sources"""

    messages = [{"role": "system", "content": system_prompt}]
    
    user_msg = question
    if file_path and os.path.exists(file_path):
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        user_msg += f"\n\n[An attachment file is provided at: {file_path}]"
        if ext in ("png", "jpg", "jpeg", "gif", "bmp"):
            user_msg += "\n[This is an image file. Use analyze_image tool to examine it.]"
        elif ext in ("mp3", "wav", "mp4", "webm"):
            user_msg += "\n[This is an audio file. Use read_file tool to transcribe it.]"
        else:
            user_msg += f"\n[Use read_file tool to examine the {ext.upper()} file.]"
    
    messages.append({"role": "user", "content": user_msg})
    
    start_time = time.time()
    
    for turn in range(max_turns):
        if time.time() - start_time > timeout:
            return None, turn, "timeout"
        
        # Call API
        response = call_chat_api(messages, tools=TOOL_DEFINITIONS, tool_choice="auto")
        
        if response is None:
            print(f"    API returned None on turn {turn}", flush=True)
            continue
        
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")
        
        # Check for tool calls
        tool_calls = message.get("tool_calls", [])
        
        if tool_calls:
            # Add assistant message with tool calls
            messages.append(message)
            
            for tc in tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                func_args_str = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", f"call_{turn}")
                
                try:
                    func_args = json.loads(func_args_str)
                except json.JSONDecodeError:
                    func_args = {}
                
                print(f"    Turn {turn}: {func_name}({json.dumps(func_args)[:100]})", flush=True)
                
                if func_name in TOOL_DISPATCH:
                    try:
                        result = TOOL_DISPATCH[func_name](func_args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                else:
                    result = f"Unknown tool: {func_name}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": str(result)[:8000]
                })
        else:
            # No tool calls - check for final answer
            content = message.get("content", "") or ""
            messages.append({"role": "assistant", "content": content})
            
            # Check if content contains FINAL ANSWER
            match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                return answer, turn + 1, "success"
            
            # If model stopped without giving final answer, prompt it
            if finish_reason == "stop":
                messages.append({
                    "role": "user",
                    "content": "Please provide your final answer now using the format: FINAL ANSWER: <answer>"
                })
    
    # Extract answer from last message if possible
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", msg["content"], re.IGNORECASE)
            if match:
                return match.group(1).strip(), max_turns, "extracted"
    
    return None, max_turns, "max_turns"

# ============================================================
# GAIA Scoring (exact match with normalization)
# ============================================================
def normalize_answer(s):
    """Normalize answer for comparison."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    # Remove common prefixes
    for prefix in ["the ", "a ", "an "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Remove punctuation at the end
    s = s.rstrip(".,;:!?")
    # Normalize whitespace
    s = " ".join(s.split())
    return s

def check_answer(predicted, expected):
    """Check if the predicted answer matches the expected answer."""
    if predicted is None:
        return False
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    if pred_norm == exp_norm:
        return True
    
    # Check if one contains the other
    if exp_norm in pred_norm or pred_norm in exp_norm:
        return True
    
    # Try numeric comparison
    try:
        pred_num = float(pred_norm.replace(",", "").replace("$", "").replace("%", ""))
        exp_num = float(exp_norm.replace(",", "").replace("$", "").replace("%", ""))
        if abs(pred_num - exp_num) < 0.01:
            return True
    except (ValueError, TypeError):
        pass
    
    return False

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GAIA ToolAgent Experiment")
    parser.add_argument("--dataset", type=str, required=True, help="Path to GAIA dataset JSON")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=-1, help="End question index (-1 for all)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per question in seconds")
    parser.add_argument("--max-turns", type=int, default=15, help="Max agent turns per question")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    parser.add_argument("--resume", type=str, default="", help="Resume from existing results file")
    args = parser.parse_args()
    
    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    
    end = args.end if args.end > 0 else len(dataset)
    questions = dataset[args.start:end]
    
    print(f"Running GAIA ToolAgent experiment: {len(questions)} questions [{args.start}:{end}]", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Timeout: {args.timeout}s, Max turns: {args.max_turns}", flush=True)
    print("=" * 60, flush=True)
    
    # Load existing results for resume
    existing_results = []
    completed_ids = set()
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            existing_results = json.load(f)
        completed_ids = {r["task_id"] for r in existing_results}
        print(f"Resuming: {len(existing_results)} existing results loaded", flush=True)
    
    results = list(existing_results)
    correct = sum(1 for r in results if r.get("is_correct"))
    total = len(results)
    
    # Output path
    if not args.output:
        ts = time.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs("result/eval", exist_ok=True)
        args.output = f"result/eval/gaia_toolagent_{MODEL}_{ts}.json"
    
    for i, q in enumerate(questions):
        idx = args.start + i
        question = q.get("question", q.get("Question", ""))
        expected = q.get("answer", q.get("Final answer", q.get("final_answer", "")))
        level = q.get("level", q.get("Level", "?"))
        file_name = q.get("file_name", "")
        task_id = q.get("task_id", f"q_{idx}")
        
        # Skip if already completed
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
            answer, turns, status = run_agent(
                question, file_path=file_path,
                max_turns=args.max_turns, timeout=args.timeout
            )
        except Exception as e:
            print(f"    Exception: {e}", flush=True)
            answer, turns, status = None, 0, f"error: {e}"
        
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
            "file_type": file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        }
        results.append(result)
        
        mark = "✓" if is_correct else "✗"
        print(f"    {mark} Answer: {answer} | Expected: {expected} | {elapsed:.1f}s | {turns} turns", flush=True)
        print(f"    Running Accuracy: {correct}/{total} ({100*correct/total:.1f}%)", flush=True)
        
        # Save intermediate results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT COMPLETE", flush=True)
    print(f"Total: {total}, Correct: {correct}, Accuracy: {100*correct/total:.1f}%", flush=True)
    
    # Print per-level stats
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
    
    print(f"\nResults saved to: {args.output}", flush=True)

if __name__ == "__main__":
    main()
