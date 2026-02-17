#!/usr/bin/env python
"""Debug GAIA experiment - minimal version to find the issue."""

import os
import sys
import asyncio
import json

from dotenv import load_dotenv
load_dotenv()
# Requires OPENAI_API_KEY and OPENAI_BASE_URL in .env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI

async def main():
    aclient = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    
    messages = [
        {"role": "system", "content": "You are a general AI assistant."},
        {"role": "user", "content": "What is the capital city of Australia?"},
    ]
    
    print(f"Client base_url: {aclient.base_url}")
    print(f"Client api_key: {aclient.api_key[:10]}...")
    
    try:
        response = await aclient.responses.create(
            model="gpt-5.1-codex-mini",
            input=messages,
        )
        
        print(f"Response type: {type(response)}")
        print(f"Response id: {response.id}")
        print(f"Response status: {response.status}")
        print(f"Response output is None: {response.output is None}")
        
        if response.output is not None:
            print(f"Output length: {len(response.output)}")
            for i, item in enumerate(response.output):
                print(f"  Output[{i}]: type={getattr(item, 'type', 'unknown')}")
            print(f"Output text: {response.output_text}")
        else:
            # Print raw dump
            raw = response.model_dump()
            print(f"RAW DUMP: {json.dumps(raw, indent=2)[:1000]}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
