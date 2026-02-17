#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAIA Experiment Runner - Direct LLM approach.
Bypasses the complex agent graph to avoid retry/SSL issues.
Supports multiple configurations: DirectAnswer, WebSearch+Answer, ToolTOT-style.
"""

import os
import sys
import argparse
import json
import time
import asyncio
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

# Load environment configuration from .env file or system environment variables.
# Set OPENAI_API_KEY and OPENAI_BASE_URL in your .env file before running.
from dotenv import load_dotenv
load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
from swarm.environment.tools.reader.readers import JSONReader
from swarm.environment.tools.search.mcp_search import MCPWebSearchEngine
from swarm.environment.domain.gaia import question_scorer
from swarm.utils.globals import PromptTokens, CompletionTokens

MODEL = "gpt-5.1-codex-mini"
search_engine = MCPWebSearchEngine()
aclient = None  # Will be initialized in run_experiment

GAIA_SYSTEM_PROMPT = (
    "You are a general AI assistant. "
    "I will ask you a question. Report your thoughts, and finish your answer with the following template: "
    "FINAL ANSWER: [YOUR FINAL ANSWER]. "
    "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. "
    "If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. "
    "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. "
    "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
)


async def call_llm(messages, max_retries=3):
    """Call the LLM with retry logic."""
    global aclient
    for attempt in range(max_retries):
        try:
            response = await aclient.responses.create(
                model=MODEL,
                input=messages,
            )
            # Use output_text property (simpler and more reliable)
            text = getattr(response, 'output_text', None) or ""
            if not text and response.output:
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for piece in item.content:
                            if hasattr(piece, 'text'):
                                text = piece.text
                                break
            # Count tokens
            if hasattr(response, 'usage') and response.usage:
                PromptTokens.instance().value += getattr(response.usage, 'input_tokens', 0)
                CompletionTokens.instance().value += getattr(response.usage, 'output_tokens', 0)
            return text
        except Exception as e:
            print(f"  LLM call attempt {attempt+1} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                # Recreate client on SSL errors
                if 'SSL' in str(e) or 'Connection' in str(e):
                    aclient = AsyncOpenAI(
                        api_key=os.environ['OPENAI_API_KEY'],
                        base_url=os.environ['OPENAI_BASE_URL'],
                    )
            else:
                raise


async def direct_answer(question):
    """Strategy 1: Direct answer without tools."""
    messages = [
        {"role": "system", "content": GAIA_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return await call_llm(messages)


async def web_search_answer(question):
    """Strategy 2: Web search + answer."""
    # Step 1: Generate search queries
    query_prompt = (
        f"Generate exactly 3 short web search queries (each under 10 words) to answer this question. "
        f"Output ONLY the queries, one per line, no numbering, no quotes, no markdown.\n\n"
        f"Question: {question}"
    )
    query_messages = [
        {"role": "system", "content": "Output only plain text search queries, one per line. No formatting, no markdown, no numbering."},
        {"role": "user", "content": query_prompt},
    ]
    queries_text = await call_llm(query_messages)
    # Parse queries: split by newline first, then by comma
    raw_queries = []
    for line in queries_text.strip().split('\n'):
        line = line.strip()
        if line and not line.lower().startswith(('search', 'quer')):
            # Remove numbering like "1." or "1)" or "- "
            import re as _re
            line = _re.sub(r'^[\d]+[\.)\-]\s*', '', line).strip()
            line = line.strip('"').strip("'").strip()
            if line and len(line) > 3:
                raw_queries.append(line)
    if not raw_queries:
        raw_queries = [q.strip() for q in queries_text.split(',') if q.strip()]
    queries = raw_queries[:3]
    
    # Step 2: Search
    search_results = []
    for query in queries:
        try:
            result = search_engine.search(query, num=3)
            if result:
                search_results.append(f"Query: {query}\nResults: {result}")
        except Exception as e:
            print(f"  Search failed for '{query}': {e}")
    
    search_context = "\n\n".join(search_results) if search_results else "No search results available."
    
    # Step 3: Answer with context
    answer_messages = [
        {"role": "system", "content": GAIA_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Use the following search results to answer the question.\n\n"
            f"## Search Results:\n{search_context}\n\n"
            f"## Question:\n{question}"
        )},
    ]
    return await call_llm(answer_messages)


async def tool_tot_answer(question):
    """Strategy 3: Tool-augmented Tree of Thought (like ToolTOT)."""
    # Step 1: Analyze question and generate clues
    clue_messages = [
        {"role": "system", "content": "You are an analytical assistant. Identify key clues and concepts needed to answer the question. Be concise."},
        {"role": "user", "content": (
            f"Evaluate if additional information is needed to answer this question. "
            f"Identify critical clues and concepts.\n\nQuestion: {question}"
        )},
    ]
    clues = await call_llm(clue_messages)
    
    # Step 2: Web search based on clues
    search_query_messages = [
        {"role": "system", "content": "Output only plain text search queries, one per line. No formatting, no markdown, no numbering. Each query under 10 words."},
        {"role": "user", "content": (
            f"Generate 3 short web search queries to answer this question.\n\n"
            f"Question: {question}\n\n"
            f"Clues: {clues[:500]}\n\n"
            f"Output ONLY the queries, one per line."
        )},
    ]
    queries_text = await call_llm(search_query_messages)
    import re as _re
    raw_queries = []
    for line in queries_text.strip().split('\n'):
        line = line.strip()
        if line and not line.lower().startswith(('search', 'quer')):
            line = _re.sub(r'^[\d]+[\.)\-]\s*', '', line).strip()
            line = line.strip('"').strip("'").strip()
            if line and len(line) > 3:
                raw_queries.append(line)
    if not raw_queries:
        raw_queries = [q.strip().strip('"').strip("'") for q in queries_text.split(',') if q.strip()]
    queries = raw_queries[:3]
    
    search_results = []
    for query in queries:
        try:
            result = search_engine.search(query, num=3)
            if result:
                search_results.append(f"Query: {query}\nResults: {result}")
        except Exception as e:
            print(f"  Search failed for '{query}': {e}")
    
    search_context = "\n\n".join(search_results) if search_results else "No search results available."
    
    # Step 3: Synthesize answer
    answer_messages = [
        {"role": "system", "content": GAIA_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"## Analysis:\n{clues}\n\n"
            f"## Search Results:\n{search_context}\n\n"
            f"## Question:\n{question}\n\n"
            f"Based on the analysis and search results, provide your answer."
        )},
    ]
    return await call_llm(answer_messages)


def extract_final_answer(response):
    """Extract the FINAL ANSWER from the response."""
    # Try various FINAL ANSWER patterns
    for marker in ["FINAL ANSWER:", "FINAL ANSWER :", "Final Answer:", "Final answer:"]:
        if marker in response:
            answer = response.split(marker)[-1].strip()
            # Take only the first line of the answer (avoid multi-line)
            first_line = answer.split("\n")[0].strip()
            # Remove leading dashes/bullets
            first_line = first_line.lstrip("- ").strip()
            return first_line
    
    # Fallback: try to extract the last meaningful line
    lines = [l.strip().lstrip("- ") for l in response.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return response.strip()


async def run_experiment(args):
    """Run GAIA experiment."""
    
    global aclient
    aclient = AsyncOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_BASE_URL'],
    )
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    dataset = JSONReader.parse_file(args.dataset_json)
    if args.max_questions > 0:
        dataset = dataset[:args.max_questions]
    
    strategy_fn = {
        "direct": direct_answer,
        "websearch": web_search_answer,
        "tooltot": tool_tot_answer,
    }[args.strategy]
    
    # Resume support: load existing results if resuming
    result_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "result" / "eval"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    start_index = 0
    if args.resume:
        # Find the latest result file for this strategy
        existing = sorted(result_dir.glob(f"gaia_{args.strategy}_{MODEL.replace('/', '_')}_*.json"), reverse=True)
        for ef in existing:
            if 'summary' not in ef.name:
                try:
                    with open(ef) as f:
                        results = json.load(f)
                    start_index = len(results)
                    print(f"Resuming from question {start_index + 1} (loaded {len(results)} results from {ef.name})")
                    break
                except Exception:
                    continue
    
    result_file = result_dir / f"gaia_{args.strategy}_{MODEL.replace('/', '_')}_{current_time}.json"
    
    print(f"\n{'='*60}")
    print(f"GAIA Experiment Configuration")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Strategy: {args.strategy}")
    print(f"Dataset: {args.dataset_json}")
    print(f"Questions: {len(dataset)} (starting from {start_index + 1})")
    print(f"Timeout per question: {args.timeout}s")
    print(f"{'='*60}\n")
    
    total_solved = sum(1 for r in results if r.get('Solved', False))
    total_executed = len(results)
    total_time = sum(r.get('Time', 0) for r in results)
    
    for i, item in enumerate(dataset):
        if i < start_index:
            continue
        
        start_time = time.time()
        
        task = item["Question"]
        ground_truth = item["Final answer"]
        
        print(f"\n--- Question {i+1}/{len(dataset)} ---")
        print(f"Q: {task[:120]}...")
        print(f"GT: {ground_truth}")
        
        try:
            raw_answer = await asyncio.wait_for(strategy_fn(task), timeout=args.timeout)
            answer = extract_final_answer(raw_answer)
        except asyncio.TimeoutError:
            print(f"  TIMEOUT after {args.timeout}s")
            raw_answer = f"TIMEOUT after {args.timeout}s"
            answer = "TIMEOUT"
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            raw_answer = f"ERROR: {e}"
            answer = "ERROR"
        
        end_time = time.time()
        exe_time = end_time - start_time
        total_time += exe_time
        
        is_solved = question_scorer(answer, ground_truth)
        total_solved += is_solved
        total_executed += 1
        accuracy = total_solved / total_executed
        
        print(f"Raw: {raw_answer[:200]}...")
        print(f"A: {answer}")
        print(f"Correct: {'✅' if is_solved else '❌'}")
        print(f"Running Accuracy: {accuracy:.2%} ({total_solved}/{total_executed})")
        print(f"Time: {exe_time:.1f}s")
        
        result_item = {
            "question_id": i + 1,
            "task_id": item.get("task_id", ""),
            "Level": item.get("Level", 0),
            "Question": task,
            "GT": ground_truth,
            "Raw Answer": raw_answer,
            "Extracted Answer": answer,
            "Solved": bool(is_solved),
            "Total solved": total_solved,
            "Total executed": total_executed,
            "Accuracy": accuracy,
            "Time": exe_time,
            "Prompt Tokens": PromptTokens.instance().value,
            "Completion Tokens": CompletionTokens.instance().value,
        }
        results.append(result_item)
        
        # Save intermediate results
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Strategy: {args.strategy}")
    print(f"Model: {MODEL}")
    print(f"Total Questions: {total_executed}")
    print(f"Correct: {total_solved}")
    print(f"Accuracy: {total_solved/max(total_executed,1):.2%}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Avg Time per Question: {total_time/max(total_executed,1):.1f}s")
    print(f"Total Prompt Tokens: {PromptTokens.instance().value}")
    print(f"Total Completion Tokens: {CompletionTokens.instance().value}")
    print(f"Results saved to: {result_file}")
    print(f"{'='*60}")
    
    # Save summary
    summary = {
        "strategy": args.strategy,
        "model": MODEL,
        "total_questions": total_executed,
        "correct": total_solved,
        "accuracy": total_solved / max(total_executed, 1),
        "total_time": total_time,
        "avg_time_per_question": total_time / max(total_executed, 1),
        "prompt_tokens": PromptTokens.instance().value,
        "completion_tokens": CompletionTokens.instance().value,
    }
    summary_file = result_dir / f"gaia_summary_{args.strategy}_{current_time}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GPTSwarm GAIA Direct Experiment")
    parser.add_argument("--dataset_json", type=str,
                       default="datasets/gaia/gaia_full_validation.json")
    parser.add_argument("--strategy", type=str, default="tooltot",
                       choices=["direct", "websearch", "tooltot"])
    parser.add_argument("--max_questions", type=int, default=0,
                       help="Max questions to run (0 = all)")
    parser.add_argument("--timeout", type=int, default=120,
                       help="Timeout per question in seconds")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    
    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == '__main__':
    main()
