"""Generic LLM wrapper for quick-testing any HuggingFace causal LM.

This is a convenience tool for exploratory benchmarking only. Results from
this wrapper are NOT eligible for official ranking because the fixed prompt
and output parser are chosen by the organizers, not the contestant. For
official submissions, contestants must implement
``ModularMultiplicationModel`` directly.

The wrapper emits answers as base-10 digits (one digit per list entry,
MSB-first), since that is the natural format for LLMs that output decimal
text. The pipeline decoder is invoked exactly the same way as for any other
submission.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modchallenge.interface.base_model import ModularMultiplicationModel

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Compute {a} * {b} mod {p}. "
    "Only output the final numerical result, nothing else."
)

# This wrapper always emits base-10 digits; the manifest synthesised for
# `modchallenge evaluate-llm` should declare ``output_base = 10`` to match.
WRAPPER_OUTPUT_BASE = 10


class GenericLLMWrapper(ModularMultiplicationModel):
    """Wraps any HuggingFace causal LM for modular multiplication evaluation.

    Works with both base models and chat/instruct models (auto-detects
    chat template support).
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        dtype: str = "bfloat16",
        prompt_template: str = DEFAULT_PROMPT,
    ):
        self.model_id = model_id
        self.revision = revision
        self.dtype = getattr(torch, dtype, torch.bfloat16)
        self.prompt_template = prompt_template
        self.model = None
        self.tokenizer = None
        self.device = None
        self._has_chat_template = False

    def load(self, model_dir: str) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(
            "Loading %s (revision=%s, dtype=%s, device=%s)",
            self.model_id, self.revision, self.dtype, self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()

        self._has_chat_template = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )
        logger.info(
            "Model loaded. chat_template=%s, vocab_size=%d",
            self._has_chat_template, len(self.tokenizer),
        )

    # The LLM wrapper consumes raw decimal strings directly; the default
    # identity preprocess hooks suit it. (predict_a, predict_b, predict_p
    # all return their input unchanged.)

    def _build_input(self, a: str, b: str, p: str) -> str:
        """Build the model input text for a single problem."""
        prompt = self.prompt_template.format(a=a, b=b, p=p)

        if self._has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return prompt

    def _generate_decimal(self, a: str, b: str, p: str) -> str:
        """Generate the LLM response and extract the last decimal number."""
        input_text = self._build_input(a, b, p)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract the last number from the response.
        matches = re.findall(r"\d+", response)
        if matches:
            return matches[-1]
        return ""

    def predict_digits(self, a_enc: Any, b_enc: Any, p_enc: Any) -> list[int]:
        """Generate the LLM answer and return it as base-10 digits, MSB-first.

        Empty / unparseable responses return ``[]``, which the pipeline
        treats as ``0`` after decoding (scored as wrong unless the true
        answer is 0).
        """
        decimal = self._generate_decimal(a_enc, b_enc, p_enc)
        if not decimal:
            return []
        # Strip leading zeros so the canonical decoded value is correct.
        stripped = decimal.lstrip("0") or "0"
        return [int(c) for c in stripped]

    def predict_digits_batch(
        self, inputs: list[tuple[Any, Any, Any]]
    ) -> list[list[int]]:
        return [self.predict_digits(a, b, p) for a, b, p in inputs]

    def max_batch_size(self) -> int:
        return 1
