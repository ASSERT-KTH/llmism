"""llmism — detector and remediator for LLM-sounding idioms in prose."""

from __future__ import annotations

from .detector import Detector, Finding
from .remediator import RemediationResult, Remediator

__all__ = ["Detector", "Finding", "RemediationResult", "Remediator"]
__version__ = "0.1.0"
