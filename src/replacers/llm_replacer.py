"""
LLM-powered content replacement module.

Uses local MLX inference pipeline or cloud APIs to generate
style-matched replacements for flagged content.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .local_inference import LocalInferenceClient, LocalModelConfig, RECOMMENDED_MODELS

logger = logging.getLogger(__name__)


@dataclass
class StyleAnalysis:
    """Analysis of an author's writing style."""

    sentence_structure: str
    vocabulary_level: str
    tone: str
    pov_tense: str
    distinctive_elements: list[str]
    raw_analysis: str


@dataclass
class ReplacementResult:
    """Result of replacing content."""

    original: str
    replacement: str
    style_matched: bool
    continuity_valid: bool
    notes: str = ""


class LLMReplacer:
    """
    Generates age-appropriate content replacements using LLMs.

    Defaults to local MLX inference pipeline. Analyzes author style
    and generates replacements that maintain narrative voice while
    removing graphic content.
    """

    def __init__(
        self,
        use_local: bool = True,
        local_url: str = "http://localhost:8000",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        templates_path: Optional[str] = None
    ):
        """
        Initialize the LLM replacer.

        Args:
            use_local: Use local inference (default True).
            local_url: URL of local inference server.
            model: Model to use (auto-selected if None).
            api_key: Anthropic API key (only for cloud fallback).
            templates_path: Path to prompt templates file.
        """
        self.use_local = use_local

        if use_local:
            config = LocalModelConfig(
                base_url=local_url,
                default_model=model or RECOMMENDED_MODELS["content_replacement"]
            )
            self.client = LocalInferenceClient(config)

            # Check server health
            if not self.client.health_check():
                logger.warning(
                    f"Local server at {local_url} not responding. "
                    "Start server with: python -m uvicorn src.api.main:app --port 8000"
                )
        else:
            # Cloud fallback (Anthropic)
            try:
                from anthropic import Anthropic
                self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set for cloud mode")
                self.client = Anthropic(api_key=self.api_key)
                self.model = model or "claude-sonnet-4-5-20250929"
            except ImportError:
                raise ImportError(
                    "For cloud mode, install anthropic: pip install anthropic"
                )

        # Load prompt templates
        if templates_path is None:
            templates_path = (
                Path(__file__).parent.parent.parent
                / "data" / "templates" / "replacement_prompts.txt"
            )
        self.templates = self._load_templates(templates_path)

        # Rate limiting for cloud
        self.last_request_time = 0
        self.min_request_interval = 0.1 if use_local else 0.5

    def _load_templates(self, path: str | Path) -> dict:
        """Load prompt templates from file."""
        templates = {}
        current_name = None
        current_content = []

        try:
            with open(path, "r") as f:
                for line in f:
                    if line.startswith("## "):
                        if current_name:
                            templates[current_name] = "\n".join(current_content).strip()
                        current_name = line[3:].strip()
                        current_content = []
                    elif current_name:
                        current_content.append(line.rstrip())

                if current_name:
                    templates[current_name] = "\n".join(current_content).strip()

        except FileNotFoundError:
            templates = self._get_default_templates()

        return templates

    def _get_default_templates(self) -> dict:
        """Return default prompt templates."""
        return {
            "STYLE_ANALYSIS_PROMPT": """Analyze the following text excerpt and describe the author's writing style:

TEXT:
{text_sample}

Provide analysis of:
1. Sentence structure (simple/complex, length patterns)
2. Vocabulary level (basic/intermediate/advanced)
3. Tone (formal/casual, serious/light)
4. Point of view and tense
5. Any distinctive stylistic elements

Be concise but specific.""",

            "CONTENT_REPLACEMENT_PROMPT": """You are helping create an age-appropriate version of a book for readers age {target_age}+.

CONTEXT BEFORE:
{context_before}

CONTENT TO REPLACE (flagged as {content_type}):
{flagged_content}

CONTEXT AFTER:
{context_after}

AUTHOR STYLE:
{style_analysis}

INSTRUCTIONS:
1. The event/scene MUST still occur - do not remove plot points
2. Replace explicit/graphic details with tasteful, age-appropriate descriptions
3. Maintain the emotional impact and character development
4. Match the author's writing style exactly
5. Ensure smooth transitions with surrounding text
6. Keep approximately the same length (within 20%)

Write ONLY the replacement text, no explanations."""
        }

    def _generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.3) -> str:
        """Generate text using configured backend."""
        self._rate_limit()

        if self.use_local:
            return self.client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            # Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

    def analyze_style(self, text_sample: str) -> StyleAnalysis:
        """
        Analyze the author's writing style from a text sample.

        Args:
            text_sample: Sample text (ideally 2000+ chars).

        Returns:
            StyleAnalysis with style characteristics.
        """
        prompt = self.templates.get(
            "STYLE_ANALYSIS_PROMPT",
            self._get_default_templates()["STYLE_ANALYSIS_PROMPT"]
        ).format(text_sample=text_sample[:3000])

        analysis_text = self._generate(prompt, max_tokens=1024)

        # Parse basic style indicators from response
        analysis_lower = analysis_text.lower()

        sentence_structure = "varied"
        if "simple" in analysis_lower:
            sentence_structure = "simple"
        elif "complex" in analysis_lower:
            sentence_structure = "complex"

        vocabulary_level = "intermediate"
        if "basic" in analysis_lower or "simple vocabulary" in analysis_lower:
            vocabulary_level = "basic"
        elif "advanced" in analysis_lower or "sophisticated" in analysis_lower:
            vocabulary_level = "advanced"

        tone = "narrative"
        if "formal" in analysis_lower:
            tone = "formal"
        elif "casual" in analysis_lower or "informal" in analysis_lower:
            tone = "casual"

        pov_tense = "third person past"
        if "first person" in analysis_lower:
            pov_tense = "first person"
        if "present tense" in analysis_lower:
            pov_tense = pov_tense.replace("past", "present")

        return StyleAnalysis(
            sentence_structure=sentence_structure,
            vocabulary_level=vocabulary_level,
            tone=tone,
            pov_tense=pov_tense,
            distinctive_elements=[],
            raw_analysis=analysis_text
        )

    def replace_content(
        self,
        flagged_content: str,
        context_before: str,
        context_after: str,
        content_type: str,
        target_age: int,
        style_analysis: StyleAnalysis
    ) -> ReplacementResult:
        """
        Generate a replacement for flagged content.

        Args:
            flagged_content: The content to replace.
            context_before: Text before the flagged section.
            context_after: Text after the flagged section.
            content_type: Type of content (sexual, violence, etc.).
            target_age: Target age tier.
            style_analysis: Author style analysis.

        Returns:
            ReplacementResult with generated replacement.
        """
        prompt = self.templates.get(
            "CONTENT_REPLACEMENT_PROMPT",
            self._get_default_templates()["CONTENT_REPLACEMENT_PROMPT"]
        ).format(
            target_age=target_age,
            context_before=context_before[-1500:],
            flagged_content=flagged_content,
            context_after=context_after[:1500],
            content_type=content_type,
            style_analysis=style_analysis.raw_analysis
        )

        replacement = self._generate(prompt, max_tokens=4096, temperature=0.3)
        replacement = replacement.strip()

        return ReplacementResult(
            original=flagged_content,
            replacement=replacement,
            style_matched=True,
            continuity_valid=True,
            notes=""
        )

    def batch_replace(
        self,
        scenes: list[dict],
        style_analysis: StyleAnalysis,
        target_age: int
    ) -> list[ReplacementResult]:
        """
        Replace multiple scenes in batch.

        Args:
            scenes: List of scene dicts with content, context, type.
            style_analysis: Author style analysis.
            target_age: Target age tier.

        Returns:
            List of ReplacementResults.
        """
        results = []

        for scene in scenes:
            result = self.replace_content(
                flagged_content=scene["content"],
                context_before=scene.get("context_before", ""),
                context_after=scene.get("context_after", ""),
                content_type=scene.get("type", "adult_content"),
                target_age=target_age,
                style_analysis=style_analysis
            )
            results.append(result)

        return results

    def verify_continuity(
        self,
        text_before: str,
        replacement: str,
        text_after: str
    ) -> dict:
        """
        Verify that a replacement maintains continuity.

        Args:
            text_before: Text before replacement.
            replacement: The replacement text.
            text_after: Text after replacement.

        Returns:
            Dict with validation results.
        """
        prompt = f"""Review this text transition for continuity:

BEFORE REPLACEMENT:
{text_before[-500:]}

REPLACEMENT:
{replacement}

AFTER REPLACEMENT:
{text_after[:500]}

Check for:
1. Logical flow between sections
2. Consistent character names and pronouns
3. Maintained timeline
4. No contradictions
5. Natural transitions

Is this transition smooth and logical? Identify any issues."""

        analysis = self._generate(prompt, max_tokens=1024)

        is_valid = "issue" not in analysis.lower() and "problem" not in analysis.lower()

        return {
            "continuity_valid": is_valid,
            "issues": [] if is_valid else [analysis],
            "suggested_fixes": []
        }

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def estimate_cost(self, text_length: int, num_scenes: int) -> dict:
        """
        Estimate processing requirements.

        Args:
            text_length: Total text length in characters.
            num_scenes: Number of scenes to replace.

        Returns:
            Estimate dict.
        """
        style_tokens = 5000
        scene_tokens = num_scenes * 8000
        total_tokens = style_tokens + scene_tokens

        if self.use_local:
            # Local inference - no cost, estimate time
            # ~30 tokens/sec for Qwen3-32B
            estimated_time = total_tokens / 30 / 60  # minutes

            return {
                "estimated_tokens": total_tokens,
                "estimated_cost_usd": 0.0,
                "estimated_time_minutes": round(estimated_time, 1),
                "num_api_calls": num_scenes + 1,
                "backend": "local"
            }
        else:
            # Cloud pricing
            input_cost_per_1k = 0.003
            output_cost_per_1k = 0.015
            input_tokens = total_tokens * 0.7
            output_tokens = total_tokens * 0.3

            estimated_cost = (
                (input_tokens / 1000) * input_cost_per_1k +
                (output_tokens / 1000) * output_cost_per_1k
            )

            return {
                "estimated_tokens": total_tokens,
                "estimated_cost_usd": round(estimated_cost, 2),
                "num_api_calls": num_scenes + 1,
                "backend": "cloud"
            }

    def get_status(self) -> dict:
        """Get status of the inference backend."""
        if self.use_local:
            healthy = self.client.health_check()
            return {
                "backend": "local",
                "healthy": healthy,
                "url": self.client.config.base_url,
                "model": self.client.config.default_model,
                "loaded_models": list(self.client._loaded_models)
            }
        else:
            return {
                "backend": "cloud",
                "healthy": True,
                "model": self.model
            }
