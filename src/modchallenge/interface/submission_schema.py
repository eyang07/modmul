"""Pydantic schema for manifest.json validation."""

from __future__ import annotations

import re
from typing import Union

from pydantic import BaseModel, Field, field_validator

# Largest base contestants can declare. Bounded so the decoder can validate
# digit ranges without unbounded resource usage.
MAX_OUTPUT_BASE = 2**32

# Sentinel string meaning "use the current problem's prime as the base".
OUTPUT_BASE_PRIME_SENTINEL = "p"


class SubmissionManifest(BaseModel):
    """Schema for the manifest.json file in a contestant's HF repo."""

    entry_class: str
    output_base: Union[int, str]
    framework: str = "pytorch"
    # Required, free-text. What the model is: architecture (Transformer / RNN /
    # CNN / hybrid / novel), approximate parameter count, input and output
    # representations, and any key design choices a reviewer would need to
    # understand the submission at a glance. Must be non-empty.
    model_description: str = Field(min_length=1)
    # Required, free-text. How the weights were obtained (training / fine-tuning
    # procedure, data, starting point). Must be non-empty; content quality is
    # judged by manual review, not mechanically. Submissions with no trained
    # parameters at all should say so explicitly — see Prohibited Practices.
    training_description: str = Field(min_length=1)

    @field_validator("entry_class")
    @classmethod
    def validate_entry_class(cls, v: str) -> str:
        """entry_class must be a valid dotted Python path like 'model.MyModel'."""
        if not re.match(r"^[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)+$", v):
            raise ValueError(
                f"entry_class must be a dotted Python path (e.g. 'model.MyModel'), got: {v!r}"
            )
        return v

    @field_validator("output_base")
    @classmethod
    def validate_output_base(cls, v: object) -> Union[int, str]:
        """Either an integer in [2, 2^32], or the string 'p' (use current prime)."""
        if isinstance(v, bool):
            # `bool` is a subclass of `int` in Python; reject explicitly.
            raise ValueError(
                f"output_base must be int in [2, {MAX_OUTPUT_BASE}] or string "
                f"{OUTPUT_BASE_PRIME_SENTINEL!r}; got bool {v!r}"
            )
        if isinstance(v, str):
            if v == OUTPUT_BASE_PRIME_SENTINEL:
                return v
            raise ValueError(
                f"output_base string must be {OUTPUT_BASE_PRIME_SENTINEL!r} "
                f"(meaning 'use the current prime as the base'); got {v!r}"
            )
        if isinstance(v, int):
            if not (2 <= v <= MAX_OUTPUT_BASE):
                raise ValueError(
                    f"output_base int must be in [2, {MAX_OUTPUT_BASE}]; got {v}"
                )
            return v
        raise ValueError(
            f"output_base must be int in [2, {MAX_OUTPUT_BASE}] or string "
            f"{OUTPUT_BASE_PRIME_SENTINEL!r}; got type {type(v).__name__}"
        )


class SubmissionRef(BaseModel):
    """A reference to an immutable submission on HuggingFace."""

    repo_id: str
    revision: str

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, v: str) -> str:
        """revision must be a full 40-character commit SHA."""
        if not re.match(r"^[0-9a-f]{40}$", v):
            raise ValueError(
                f"revision must be a 40-character hex commit SHA, got: {v!r}"
            )
        return v
