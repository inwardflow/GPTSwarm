#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAIA Experiment Runner for GPTSwarm with gpt-5.1-codex-mini.
Runs GAIA Level 1 validation questions using different agent configurations.
"""

import os
import sys
import argparse
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime

# Load environment configuration from .env file.
# Set OPENAI_API_KEY and OPENAI_BASE_URL in your .env file before running.
from dotenv import load_dotenv
load_dotenv()

from swarm.graph.swarm import Swarm
from swarm.environment.tools.reader.readers import JSONReader
from swarm.environment.agents.io import IO
from swarm.environment.agents.gaia.normal_io import NormalIO
from swarm.environment.agents.gaia.tool_io import ToolIO
from swarm.environment.agents.gaia.web_io import WebIO
from swarm.environment.agents.gaia.tool_tot import ToolTOT
from swarm.environment.operations import DirectAnswer
from swarm.memory.memory import GlobalMemory
from swarm.utils.globals import Time, Cost, CompletionTokens, PromptTokens
from swarm.utils.const import GPTSWARM_ROOT
from swarm.utils.log import initialize_log_file, logger, swarmlog
from swarm.environment.domain.gaia import question_scorer
from swarm.environment.operations.final_decision import MergingStrategy


def dataloader(data_list):
    for data in data_list:
        yield data


async def run_experiment(args):
    """Run GAIA experiment with specified configuration."""
    
    model_name = args.llm
    agent_type = args.agent_type
    
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    
    log_file_path = initialize_log_file("GAIA", Time.instance().value)
    
    dataset = JSONReader.parse_file(args.dataset_json)
    
    # Limit dataset if specified
    if args.max_questions > 0:
        dataset = dataset[:args.max_questions]
    
    print(f"\n{'='*60}")
    print(f"GAIA Experiment Configuration")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Agent Type: {agent_type}")
    print(f"Dataset: {args.dataset_json}")
    print(f"Questions: {len(dataset)}")
    print(f"{'='*60}\n")
    
    # Setup result file
    result_dir = Path(f"{GPTSWARM_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"gaia_{agent_type}_{model_name.replace('/', '_')}_{current_time}.json"
    
    results = []
    total_solved = 0
    total_executed = 0
    
    for i, item in enumerate(dataloader(dataset)):
        start_time = time.time()
        
        # Clear memory and create fresh agent for each question
        GlobalMemory.instance().memory = {}
        if agent_type == "NormalIO":
            agent = NormalIO(domain="gaia", model_name=model_name)
        elif agent_type == "ToolIO":
            agent = ToolIO(domain="gaia", model_name=model_name)
        elif agent_type == "WebIO":
            agent = WebIO(domain="gaia", model_name=model_name)
        elif agent_type == "ToolTOT":
            agent = ToolTOT(domain="gaia", model_name=model_name)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        task = item["Question"]
        files = []  # No file attachments for our Level 1 questions
        ground_truth = item["Final answer"]
        inputs = {"task": task, "files": files, "GT": ground_truth}
        
        print(f"\n--- Question {i+1}/{len(dataset)} ---")
        print(f"Q: {task[:100]}...")
        print(f"GT: {ground_truth}")
        
        try:
            answer = await agent.run(inputs=inputs)
            answer = answer[-1].split("FINAL ANSWER: ")[-1].strip()
        except Exception as e:
            logger.error(f"Error on question {i+1}: {e}")
            answer = "ERROR"
        
        end_time = time.time()
        exe_time = end_time - start_time
        
        is_solved = question_scorer(answer, ground_truth)
        total_solved += is_solved
        total_executed += 1
        accuracy = total_solved / total_executed
        
        print(f"A: {answer}")
        print(f"Correct: {'✅' if is_solved else '❌'}")
        print(f"Running Accuracy: {accuracy:.2%} ({total_solved}/{total_executed})")
        print(f"Time: {exe_time:.1f}s | Cost: ${Cost.instance().value:.4f}")
        
        result_item = {
            "Question": task,
            "GT": ground_truth,
            "Attempt answer": answer,
            "Solved": is_solved,
            "Total solved": total_solved,
            "Total executed": total_executed,
            "Accuracy": accuracy,
            "Time": exe_time,
            "Total Cost": Cost.instance().value,
            "Prompt Tokens": PromptTokens.instance().value,
            "Completion Tokens": CompletionTokens.instance().value,
        }
        results.append(result_item)
        
        # Save intermediate results
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Agent: {agent_type}")
    print(f"Model: {model_name}")
    print(f"Total Questions: {total_executed}")
    print(f"Correct: {total_solved}")
    print(f"Accuracy: {total_solved/total_executed:.2%}")
    print(f"Total Cost: ${Cost.instance().value:.4f}")
    print(f"Total Prompt Tokens: {PromptTokens.instance().value}")
    print(f"Total Completion Tokens: {CompletionTokens.instance().value}")
    print(f"Results saved to: {result_file}")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GPTSwarm GAIA Experiment")
    parser.add_argument("--dataset_json", type=str, 
                       default="datasets/gaia/level_1_val.json")
    parser.add_argument("--llm", type=str, default="gpt-5.1-codex-mini")
    parser.add_argument("--agent_type", type=str, default="ToolTOT",
                       choices=["NormalIO", "ToolIO", "WebIO", "ToolTOT"])
    parser.add_argument("--max_questions", type=int, default=0,
                       help="Max questions to run (0 = all)")
    
    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == '__main__':
    main()
