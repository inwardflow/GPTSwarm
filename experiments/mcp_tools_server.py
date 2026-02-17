#!/usr/bin/env python3
"""
GAIA Tools MCP Server
=====================
A Model Context Protocol (MCP) server that provides all tools needed for the
GAIA benchmark: web search, file reading, code execution, image analysis,
audio transcription, and web page fetching.

Usage:
  # Run directly via FastMCP
  python3 mcp_tools_server.py

  # Or via fastmcp CLI
  fastmcp run mcp_tools_server.py

  # Or via manus-mcp-cli (after configuring in mcp.json)
  manus-mcp-cli tool list -s gaia-tools
"""

import os
import json
import subprocess
import tempfile
import re
from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    "gaia-tools",
    description="Tools for the GAIA benchmark: web search, file reading, code execution, image analysis, audio transcription, and web page fetching."
)


# ============================================================
# Tool 1: Web Search (DuckDuckGo)
# ============================================================
@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return titles, URLs, and snippets.
    
    Use this for finding factual information, current events, or any knowledge
    you don't already have.
    
    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
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


# ============================================================
# Tool 2: File Reading (PDF, Excel, CSV, DOCX, PPTX, ZIP, etc.)
# ============================================================
@mcp.tool()
def read_file(file_path: str) -> str:
    """Read and extract content from various file types.
    
    Supports: PDF, Excel (.xlsx/.xls), CSV, DOCX, PPTX, ZIP, JSON, TXT, PY,
    MP3/WAV (audio transcription), PDB, JSONLD, and other text formats.
    
    Args:
        file_path: The absolute path to the file to read.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    
    try:
        if ext == "pdf":
            return _read_pdf(file_path)
        elif ext in ("xlsx", "xls"):
            return _read_excel(file_path)
        elif ext == "csv":
            with open(file_path, "r", errors="replace") as f:
                return f.read()[:10000]
        elif ext == "docx":
            return _read_docx(file_path)
        elif ext == "pptx":
            return _read_pptx(file_path)
        elif ext == "zip":
            return _read_zip(file_path)
        elif ext in ("mp3", "wav", "mp4", "webm"):
            return _transcribe_audio(file_path)
        elif ext in ("png", "jpg", "jpeg", "gif", "bmp", "tiff"):
            return _do_ocr(file_path)
        elif ext in ("pdb", "jsonld"):
            with open(file_path, "r", errors="replace") as f:
                return f.read()[:10000]
        else:
            with open(file_path, "r", errors="replace") as f:
                return f.read()[:10000]
    except Exception as e:
        return f"Error reading file: {e}"


def _read_pdf(file_path: str) -> str:
    """Extract text from PDF using pdftotext or PyPDF2."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", file_path, "-"],
            capture_output=True, text=True, timeout=30
        )
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


def _read_excel(file_path: str) -> str:
    """Read Excel file and return content as tab-separated text."""
    import openpyxl
    wb = openpyxl.load_workbook(file_path, data_only=True)
    output = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        output.append(f"=== Sheet: {sheet_name} ===")
        for row in ws.iter_rows(values_only=True):
            output.append("\t".join(str(c) if c is not None else "" for c in row))
    return "\n".join(output)[:15000]


def _read_docx(file_path: str) -> str:
    """Read DOCX file."""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text[:10000]
    except ImportError:
        return "python-docx not installed. Cannot read DOCX files."


def _read_pptx(file_path: str) -> str:
    """Read PPTX file."""
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


def _read_zip(file_path: str) -> str:
    """Read ZIP file contents."""
    import zipfile
    with zipfile.ZipFile(file_path, "r") as zf:
        names = zf.namelist()
        output = [f"ZIP contents ({len(names)} files):"]
        for name in names[:20]:
            output.append(f"  {name}")
        for name in names:
            if name.endswith((".txt", ".csv", ".json", ".py", ".md")):
                try:
                    content = zf.read(name).decode("utf-8", errors="replace")
                    output.append(f"\n--- {name} ---")
                    output.append(content[:3000])
                except Exception:
                    pass
        return "\n".join(output)[:10000]


def _transcribe_audio(file_path: str) -> str:
    """Transcribe audio using manus-speech-to-text."""
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
        return f"Image: {os.path.basename(file_path)}, Size: {img.size}, Mode: {img.mode}. (OCR not available - install pytesseract)"
    except Exception as e:
        return f"OCR error: {e}"


# ============================================================
# Tool 3: Python Code Execution
# ============================================================
@mcp.tool()
def run_python(code: str) -> str:
    """Execute Python code and return stdout/stderr.
    
    Use this for calculations, data processing, or any computation.
    Common packages available: numpy, pandas, openpyxl, matplotlib, scipy, etc.
    
    Args:
        code: Python code to execute.
    """
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


# ============================================================
# Tool 4: Image Analysis (OCR)
# ============================================================
@mcp.tool()
def analyze_image(file_path: str) -> str:
    """Analyze an image file using OCR (Optical Character Recognition).
    
    Extracts text content from images. Returns image metadata and any detected text.
    
    Args:
        file_path: The absolute path to the image file (PNG, JPG, etc.).
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    return _do_ocr(file_path)


# ============================================================
# Tool 5: Web Page Fetching
# ============================================================
@mcp.tool()
def fetch_webpage(url: str) -> str:
    """Fetch and extract text content from a webpage URL.
    
    Useful for reading articles, documentation, or any web page content.
    
    Args:
        url: The URL of the webpage to fetch.
    """
    try:
        import httpx
        resp = httpx.get(
            url, timeout=30, verify=False, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot)"}
        )
        if resp.status_code != 200:
            return f"HTTP {resp.status_code} error fetching {url}"
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)[:8000]
    except Exception as e:
        return f"Webpage fetch error: {e}"


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    mcp.run()
