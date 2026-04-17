"""Pydantic schema for manifest.json validation."""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator


class SubmissionManifest(BaseModel):
    """Schema for the manifest.json file in a contestant's HF repo."""

    entry_class: str
    framework: str = "pytorch"
    model_description: str = ""

    @field_validator("entry_class")
    @classmethod
    def validate_entry_class(cls, v: str) -> str:
        """entry_class must be a valid dotted Python path like 'model.MyModel'."""
        if not re.match(r"^[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)+$", v):
            raise ValueError(
                f"entry_class must be a dotted Python path (e.g. 'model.MyModel'), got: {v!r}"
            )
        return v


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
