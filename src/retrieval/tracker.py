import time
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler

# 2026 Pricing (per 1M tokens) - Updated for Feb 2026
PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.1-pro-preview": {"input": 1.00, "output": 6.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50}
}

class PerformanceTracker(BaseCallbackHandler):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.start_time = 0
        # Initialize with 'latency' and 'tokens' so ui.py doesn't throw a KeyError
        self.metrics = {
            "latency": 0.0, 
            "tokens": 0, 
            "cost_usd": "$0.000000", 
            "model": model_name
        }

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any):
        self.start_time = time.perf_counter()

    def on_llm_end(self, response: Any, **kwargs: Any):
        duration = time.perf_counter() - self.start_time
        
        # 2026 Gemini response extraction logic
        # LangChain usually puts usage in response_metadata or usage_metadata
        usage = getattr(response, "usage_metadata", {}) or {}
        
        # Mapping for 2026 Gemini API keys
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_token_count") or 0
        completion_tokens = usage.get("output_tokens") or usage.get("candidates_token_count") or 0
        thinking_tokens = usage.get("thought_tokens") or usage.get("thoughts_token_count") or 0
        
        total_tokens = prompt_tokens + completion_tokens + thinking_tokens
        
        # Calculate Cost based on the Pricing Table
        rates = PRICING.get(self.model_name, {"input": 0.50, "output": 3.00})
        # Thinking tokens are billed at the output rate in 2026
        cost = (prompt_tokens / 1_000_000 * rates["input"]) + \
               ((completion_tokens + thinking_tokens) / 1_000_000 * rates["output"])
        
        self.metrics = {
            "model": self.model_name,
            "latency": round(duration, 2), 
            "tokens": total_tokens,
            "cost_usd": f"${cost:.6f}",
            "thinking_tokens": thinking_tokens
        }