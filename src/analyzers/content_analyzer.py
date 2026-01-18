"""
Content analysis module for detecting adult scenes.

Analyzes text to identify sections containing graphic sexual content,
extreme violence, or detailed intoxication scenes.
"""

import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ContentScene:
    """Represents a detected content scene requiring review."""

    start_pos: int
    end_pos: int
    content: str
    category: str
    intensity_score: float
    context_before: str = ""
    context_after: str = ""
    keywords_found: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of analyzing text for adult content."""

    text: str
    scenes: list[ContentScene]
    total_scenes: int
    categories: dict[str, int]
    needs_llm_review: bool


class ContentAnalyzer:
    """
    Analyzes text to detect adult content scenes.

    Uses keyword matching and pattern recognition to identify
    sections that may need content filtering or replacement.
    """

    def __init__(self, keywords_path: Optional[str] = None):
        """
        Initialize the content analyzer.

        Args:
            keywords_path: Path to content keywords JSON file.
        """
        if keywords_path is None:
            keywords_path = (
                Path(__file__).parent.parent.parent
                / "data" / "word_lists" / "content_keywords.json"
            )

        self.keywords = self._load_keywords(keywords_path)
        self._build_patterns()

    def _load_keywords(self, path: str | Path) -> dict:
        """Load content keywords from JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def _build_patterns(self) -> None:
        """Build regex patterns for content detection."""
        self.category_patterns = {}

        categories = self.keywords.get("categories", {})

        for category_name, category_data in categories.items():
            patterns = []

            for term_type, terms in category_data.items():
                if term_type == "intensity_multipliers":
                    continue

                if isinstance(terms, list):
                    for term in terms:
                        # Create pattern for multi-word phrases and single words
                        if " " in term:
                            pattern = re.escape(term)
                        else:
                            pattern = rf"\b{re.escape(term)}\w*\b"
                        patterns.append(pattern)

            if patterns:
                combined = "|".join(patterns)
                self.category_patterns[category_name] = re.compile(
                    combined, re.IGNORECASE
                )

    def analyze(
        self,
        text: str,
        age: int,
        context_size: int = 1500
    ) -> AnalysisResult:
        """
        Analyze text for adult content.

        Args:
            text: Text to analyze.
            age: Target age tier for threshold determination.
            context_size: Characters of context to include.

        Returns:
            AnalysisResult with detected scenes.
        """
        thresholds = self._get_thresholds(age)
        all_scenes = []
        category_counts = {cat: 0 for cat in self.category_patterns.keys()}

        # Find all keyword matches
        for category, pattern in self.category_patterns.items():
            matches = list(pattern.finditer(text))

            if not matches:
                continue

            # Group nearby matches into scenes
            scenes = self._group_matches_into_scenes(
                text, matches, category, context_size
            )

            # Calculate intensity and filter by threshold
            threshold = thresholds.get(category, 0.5)

            for scene in scenes:
                if scene.intensity_score >= threshold:
                    all_scenes.append(scene)
                    category_counts[category] += 1

        # Merge overlapping scenes
        merged_scenes = self._merge_overlapping_scenes(all_scenes)

        # Add context to scenes
        for scene in merged_scenes:
            scene.context_before = text[
                max(0, scene.start_pos - context_size):scene.start_pos
            ]
            scene.context_after = text[
                scene.end_pos:scene.end_pos + context_size
            ]

        return AnalysisResult(
            text=text,
            scenes=merged_scenes,
            total_scenes=len(merged_scenes),
            categories=category_counts,
            needs_llm_review=len(merged_scenes) > 0
        )

    def _get_thresholds(self, age: int) -> dict:
        """Get content thresholds for age tier."""
        age_thresholds = self.keywords.get("age_thresholds", {})

        # Find closest age tier
        if age <= 10:
            tier_key = "10"
        elif age <= 13:
            tier_key = "13"
        elif age <= 15:
            tier_key = "15"
        else:
            tier_key = "17"

        return age_thresholds.get(tier_key, {})

    def _group_matches_into_scenes(
        self,
        text: str,
        matches: list,
        category: str,
        context_size: int
    ) -> list[ContentScene]:
        """Group nearby matches into scene segments."""
        if not matches:
            return []

        scenes = []
        min_scene_length = 100
        max_scene_length = 5000
        merge_distance = 200

        # Sort matches by position
        matches = sorted(matches, key=lambda m: m.start())

        current_start = matches[0].start()
        current_end = matches[0].end()
        current_keywords = [matches[0].group()]

        for match in matches[1:]:
            # Check if this match should be merged with current scene
            if match.start() - current_end <= merge_distance:
                current_end = match.end()
                current_keywords.append(match.group())
            else:
                # Save current scene and start new one
                scene = self._create_scene(
                    text, current_start, current_end,
                    category, current_keywords
                )
                if scene:
                    scenes.append(scene)

                current_start = match.start()
                current_end = match.end()
                current_keywords = [match.group()]

        # Don't forget the last scene
        scene = self._create_scene(
            text, current_start, current_end,
            category, current_keywords
        )
        if scene:
            scenes.append(scene)

        return scenes

    def _create_scene(
        self,
        text: str,
        start: int,
        end: int,
        category: str,
        keywords: list[str]
    ) -> Optional[ContentScene]:
        """Create a ContentScene from match data."""
        # Expand to sentence/paragraph boundaries
        expanded_start = self._find_boundary_before(text, start)
        expanded_end = self._find_boundary_after(text, end)

        content = text[expanded_start:expanded_end]

        # Calculate intensity score
        intensity = self._calculate_intensity(
            content, category, keywords
        )

        return ContentScene(
            start_pos=expanded_start,
            end_pos=expanded_end,
            content=content,
            category=category,
            intensity_score=intensity,
            keywords_found=keywords
        )

    def _find_boundary_before(self, text: str, pos: int) -> int:
        """Find paragraph/sentence boundary before position."""
        # Look for paragraph break
        para_break = text.rfind("\n\n", 0, pos)
        if para_break != -1 and pos - para_break < 500:
            return para_break + 2

        # Look for sentence end
        sentence_end = max(
            text.rfind(". ", 0, pos),
            text.rfind("! ", 0, pos),
            text.rfind("? ", 0, pos)
        )
        if sentence_end != -1 and pos - sentence_end < 300:
            return sentence_end + 2

        # Just go back a bit
        return max(0, pos - 100)

    def _find_boundary_after(self, text: str, pos: int) -> int:
        """Find paragraph/sentence boundary after position."""
        text_len = len(text)

        # Look for paragraph break
        para_break = text.find("\n\n", pos)
        if para_break != -1 and para_break - pos < 500:
            return para_break

        # Look for sentence end
        for punct in [". ", "! ", "? "]:
            end = text.find(punct, pos)
            if end != -1 and end - pos < 300:
                return end + 1

        # Just go forward a bit
        return min(text_len, pos + 100)

    def _calculate_intensity(
        self,
        content: str,
        category: str,
        keywords: list[str]
    ) -> float:
        """
        Calculate intensity score for a scene.

        Score is based on:
        - Number of keywords
        - Types of keywords (explicit terms score higher)
        - Density of keywords in content
        """
        if not content:
            return 0.0

        base_score = 0.0
        multipliers = self.keywords.get(
            "categories", {}
        ).get(category, {}).get("intensity_multipliers", {})

        # Get all terms by type for scoring
        category_data = self.keywords.get("categories", {}).get(category, {})

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Find which term type this keyword belongs to
            for term_type, terms in category_data.items():
                if term_type == "intensity_multipliers":
                    continue

                if isinstance(terms, list):
                    # Check if keyword matches any term
                    for term in terms:
                        if term.lower() in keyword_lower or keyword_lower in term.lower():
                            multiplier = multipliers.get(term_type, 1.0)
                            base_score += multiplier
                            break

        # Factor in density
        content_length = len(content)
        density_factor = min(2.0, len(keywords) / (content_length / 100))

        # Normalize to 0-1 scale
        raw_score = base_score * density_factor
        normalized_score = min(1.0, raw_score / 10)

        return normalized_score

    def _merge_overlapping_scenes(
        self,
        scenes: list[ContentScene]
    ) -> list[ContentScene]:
        """Merge scenes that overlap."""
        if not scenes:
            return []

        # Sort by start position
        scenes = sorted(scenes, key=lambda s: s.start_pos)
        merged = [scenes[0]]

        for scene in scenes[1:]:
            last = merged[-1]

            if scene.start_pos <= last.end_pos:
                # Merge scenes
                merged_scene = ContentScene(
                    start_pos=last.start_pos,
                    end_pos=max(last.end_pos, scene.end_pos),
                    content=last.content,  # Will be updated later
                    category=f"{last.category},{scene.category}",
                    intensity_score=max(
                        last.intensity_score, scene.intensity_score
                    ),
                    keywords_found=list(set(
                        last.keywords_found + scene.keywords_found
                    ))
                )
                merged[-1] = merged_scene
            else:
                merged.append(scene)

        return merged

    def is_false_positive(
        self,
        scene: ContentScene,
        full_text: str
    ) -> bool:
        """
        Check if a detected scene is a false positive.

        Args:
            scene: The detected scene.
            full_text: Full text for context.

        Returns:
            True if this is likely a false positive.
        """
        filters = self.keywords.get("false_positive_filters", {})

        context = full_text[
            max(0, scene.start_pos - 500):
            min(len(full_text), scene.end_pos + 500)
        ].lower()

        # Check medical context
        medical_terms = filters.get("medical_context", [])
        medical_count = sum(1 for term in medical_terms if term in context)
        if medical_count >= 2:
            return True

        # Check clinical terms
        clinical_terms = filters.get("clinical_terms", [])
        if any(term in context for term in clinical_terms):
            return True

        # Check non-sexual intimacy
        if scene.category == "sexual_content":
            non_sexual = filters.get("non_sexual_intimacy", [])
            if any(phrase in context for phrase in non_sexual):
                if scene.intensity_score < 0.5:
                    return True

        return False

    def get_summary(self, result: AnalysisResult) -> str:
        """Generate a human-readable summary of analysis."""
        if not result.scenes:
            return "No adult content detected."

        lines = [
            f"Found {result.total_scenes} scene(s) requiring review:",
            ""
        ]

        for cat, count in result.categories.items():
            if count > 0:
                lines.append(f"  - {cat}: {count} scene(s)")

        lines.append("")
        lines.append("Scene details:")

        for i, scene in enumerate(result.scenes, 1):
            lines.append(
                f"  {i}. {scene.category} (intensity: {scene.intensity_score:.2f})"
            )
            lines.append(f"     Position: {scene.start_pos}-{scene.end_pos}")
            lines.append(f"     Keywords: {', '.join(scene.keywords_found[:5])}")

        return "\n".join(lines)
