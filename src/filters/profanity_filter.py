"""
Age-adaptive profanity filtering module.

Filters and replaces profanity based on age tier settings,
with context-aware replacement selection.
"""

import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class FilterResult:
    """Result of filtering a text segment."""

    original_text: str
    filtered_text: str
    replacements_made: list[dict]
    word_count: int


class ProfanityFilter:
    """
    Age-adaptive profanity filter.

    Filters profanity based on age tier, with higher tiers allowing
    more words through. Supports context-aware replacement selection.
    """

    def __init__(self, word_list_path: Optional[str] = None):
        """
        Initialize the profanity filter.

        Args:
            word_list_path: Path to profanity tiers JSON file.
        """
        if word_list_path is None:
            # Default path relative to this file
            word_list_path = (
                Path(__file__).parent.parent.parent
                / "data" / "word_lists" / "profanity_tiers.json"
            )

        self.word_lists = self._load_word_lists(word_list_path)
        self._build_patterns()

    def _load_word_lists(self, path: str | Path) -> dict:
        """Load profanity word lists from JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def _build_patterns(self) -> None:
        """Build regex patterns for each tier."""
        self.tier_patterns = {}
        self.tier_replacements = {}

        tiers = self.word_lists.get("tiers", {})

        for tier_name, tier_data in tiers.items():
            words = tier_data.get("words", {})

            # Build pattern and replacement dict for this tier
            patterns = []
            replacements = {}

            for word, replacement_list in words.items():
                if isinstance(replacement_list, list):
                    # Create pattern that matches word with word boundaries
                    # Case insensitive matching
                    pattern = rf"\b{re.escape(word)}\b"
                    patterns.append(pattern)
                    replacements[word.lower()] = replacement_list

            if patterns:
                # Combine all patterns with OR
                combined_pattern = "|".join(patterns)
                self.tier_patterns[tier_name] = re.compile(
                    combined_pattern, re.IGNORECASE
                )
                self.tier_replacements[tier_name] = replacements

    def filter_text(self, text: str, age: int) -> FilterResult:
        """
        Filter profanity from text based on age tier.

        Args:
            text: The text to filter.
            age: Target age (10, 13, 15, or 17).

        Returns:
            FilterResult with filtered text and replacement details.
        """
        # Determine which tiers to apply
        tiers_to_apply = self._get_tiers_for_age(age)

        filtered_text = text
        all_replacements = []

        for tier_name in tiers_to_apply:
            if tier_name not in self.tier_patterns:
                continue

            pattern = self.tier_patterns[tier_name]
            replacements = self.tier_replacements[tier_name]

            def replace_match(match):
                word = match.group(0)
                word_lower = word.lower()

                # Find replacement
                replacement_list = replacements.get(word_lower, [word])
                replacement = replacement_list[0] if replacement_list else word

                # Preserve case
                replacement = self._match_case(word, replacement)

                all_replacements.append({
                    "original": word,
                    "replacement": replacement,
                    "position": match.start(),
                    "tier": tier_name
                })

                return replacement

            filtered_text = pattern.sub(replace_match, filtered_text)

        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            replacements_made=all_replacements,
            word_count=len(all_replacements)
        )

    def _get_tiers_for_age(self, age: int) -> list[str]:
        """
        Get list of tiers to apply for a given age.

        Lower ages apply more tiers (stricter filtering).
        """
        if age <= 10:
            return [
                "tier_10_mild",
                "tier_13_moderate",
                "tier_15_strong",
                "tier_17_extreme"
            ]
        elif age <= 13:
            return [
                "tier_13_moderate",
                "tier_15_strong",
                "tier_17_extreme"
            ]
        elif age <= 15:
            return [
                "tier_15_strong",
                "tier_17_extreme"
            ]
        else:  # 17+
            return ["tier_17_extreme"]

    def _match_case(self, original: str, replacement: str) -> str:
        """Match the case pattern of the original word."""
        if original.isupper():
            return replacement.upper()
        elif original.istitle():
            return replacement.title()
        elif original.islower():
            return replacement.lower()
        else:
            # Mixed case - just return as is
            return replacement

    def get_word_count(self, text: str, age: int) -> dict:
        """
        Count profanity words in text by tier.

        Args:
            text: Text to analyze.
            age: Target age tier.

        Returns:
            Dictionary with counts by tier.
        """
        tiers_to_check = self._get_tiers_for_age(age)
        counts = {}

        for tier_name in tiers_to_check:
            if tier_name in self.tier_patterns:
                pattern = self.tier_patterns[tier_name]
                matches = pattern.findall(text)
                counts[tier_name] = len(matches)

        counts["total"] = sum(counts.values())
        return counts

    def is_context_appropriate(
        self,
        word: str,
        context_before: str,
        context_after: str
    ) -> bool:
        """
        Check if a word is appropriate in context.

        Some words have non-profane meanings (e.g., "ass" = donkey).

        Args:
            word: The word to check.
            context_before: Text before the word.
            context_after: Text after the word.

        Returns:
            True if the word should be filtered in this context.
        """
        context_rules = self.word_lists.get("context_sensitive", {}).get("words", {})

        if word.lower() not in context_rules:
            return True  # No special rules, filter it

        rules = context_rules[word.lower()]
        full_context = f"{context_before} {word} {context_after}".lower()

        # Check for non-profane contexts
        if "animal" in rules and not rules["animal"]:
            animal_words = ["donkey", "mule", "horse", "farm", "ride"]
            if any(w in full_context for w in animal_words):
                return False  # Don't filter

        if "rooster" in rules and not rules["rooster"]:
            rooster_words = ["rooster", "chicken", "hen", "crow", "dawn"]
            if any(w in full_context for w in rooster_words):
                return False  # Don't filter

        if "fastener" in rules and not rules["fastener"]:
            fastener_words = ["screw", "bolt", "nail", "tool", "driver"]
            if any(w in full_context for w in fastener_words):
                return False  # Don't filter

        return True  # Filter the word

    def add_custom_words(
        self,
        tier: str,
        words: dict[str, list[str]]
    ) -> None:
        """
        Add custom words to a tier.

        Args:
            tier: Tier name (e.g., "tier_13_moderate").
            words: Dict of word -> replacement list.
        """
        if tier not in self.tier_replacements:
            self.tier_replacements[tier] = {}

        for word, replacements in words.items():
            self.tier_replacements[tier][word.lower()] = replacements

        # Rebuild pattern for this tier
        patterns = []
        for word in self.tier_replacements[tier].keys():
            pattern = rf"\b{re.escape(word)}\b"
            patterns.append(pattern)

        if patterns:
            combined_pattern = "|".join(patterns)
            self.tier_patterns[tier] = re.compile(
                combined_pattern, re.IGNORECASE
            )

    def get_tier_words(self, tier: str) -> list[str]:
        """Get list of words in a tier."""
        return list(self.tier_replacements.get(tier, {}).keys())
