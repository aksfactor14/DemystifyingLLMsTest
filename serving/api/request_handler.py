"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


class RequestHandler:
    """
    Handles endpoint request validation, inference invocation, and response formatting.
    """

    def __init__(self, engine: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.engine = engine
        self.tokenizer = tokenizer

    def handle_completion(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OpenAI-compatible /v1/completions request.

        Args:
            request_body: Dictionary containing 'prompt', optional 'max_tokens', 'temperature', etc.

        Returns:
            Formatted response dictionary.
        """
        prompt = request_body.get("prompt", "")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)

        # Tokenize prompt
        prompt_tokens = self._encode(prompt)

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            completion_tokens = generated_tokens[len(prompt_tokens):]
            response_text = self._decode(generated_tokens)
        elif self.engine is not None and hasattr(self.engine, "generate"):
            response_text = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            continuation = response_text[len(prompt):] if response_text.startswith(prompt) else response_text
            completion_tokens = self._encode(continuation, add_bos=False)
        else:
            completion_tokens = []
            response_text = self._decode(prompt_tokens)

        eos_token_id = getattr(getattr(self.tokenizer, "vocab", None), "eos_token_id", None)
        if eos_token_id is not None and completion_tokens and completion_tokens[-1] == eos_token_id:
            finish_reason = "stop"
        elif len(completion_tokens) >= max_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": response_text,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens)
            }
        }

    def _encode(self, text: str, add_bos: bool = True) -> List[int]:
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                return self.tokenizer.encode(text, add_bos=add_bos)
            except TypeError:
                return self.tokenizer.encode(text)
        return [ord(c) for c in text]

    def _decode(self, token_ids: List[int]) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids)
        return "".join(chr(t) for t in token_ids)

    def handle_health(self) -> Dict[str, Any]:
        """Process /health endpoint check."""
        return {
            "status": "ok",
            "service": "broken-transformer-server",
            "timestamp": time.time()
        }

    def handle_models(self) -> Dict[str, Any]:
        """Process /v1/models endpoint check."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "broken-transformer",
                    "object": "model",
                    "owned_by": "openai-dsg-iitr",
                    "ready": True
                }
            ]
        }
