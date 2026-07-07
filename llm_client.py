# llm_client.py
import os
import torch
from transformers import pipeline
from openai import OpenAI
from dotenv import load_dotenv

# Global pipeline instance to ensure the local model is only loaded into memory once
_local_pipeline = None

def get_local_pipeline():
    """Lazily loads the Hugging Face pipeline."""
    global _local_pipeline
    if _local_pipeline is None:
        print("Loading local Hugging Face model... (this may take a minute)")
        
        # Specify your desired local instruction-tuned model here
        # Example: meta-llama/Meta-Llama-3-8B-Instruct or HuggingFaceTB/SmolLM3-3B
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
        
        _local_pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"dtype": torch.bfloat16}, # Load in 16-bit to save VRAM
            device_map="auto" # Automatically maps to GPU if available
        )
    return _local_pipeline

def call_llm(system_prompt, user_text, model_type="local"):
    """
    Reusable LLM client supporting local Hugging Face models and 
    proprietary models via OpenRouter.
    """
    load_dotenv()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]

    if model_type == "local":
        try:
            pipe = get_local_pipeline()
            
            # The pipeline automatically applies the correct chat template for the model
            outputs = pipe(
                messages,
                #max_new_tokens=1024, #erst nach letztem run geändert 
                temperature=0.3, # Low temperature for consistent qualitative analysis
                do_sample=True
            )
            
            # The pipeline returns the full conversation history; extract the last assistant response
            return outputs[0]["generated_text"][-1]["content"]
            
        except Exception as e:
            print(f"Error calling local Hugging Face model: {e}")
            

    elif model_type == "proprietary":
        try:
            # OpenRouter setup using the standard OpenAI Python SDK
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_KEY"),
            )
            
            # Select the proprietary model hosted on OpenRouter (e.g., GPT-4o or Claude 3.5 Sonnet)
            model_name = "openai/gpt-oss-120b"
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenRouter LLM: {e}")
            return ""