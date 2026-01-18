"""
Main pipeline orchestrator for ebook adjustment.

Coordinates the conversion, filtering, analysis, and replacement
pipeline to produce age-adjusted ebook versions.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from ..converters.ebook_converter import EbookConverter, ParsedEbook, Chapter
from ..filters.profanity_filter import ProfanityFilter
from ..analyzers.content_analyzer import ContentAnalyzer, ContentScene
from ..replacers.llm_replacer import LLMReplacer, StyleAnalysis


@dataclass
class ProcessingStats:
    """Statistics from processing an ebook."""

    chapters_processed: int = 0
    profanity_replaced: int = 0
    scenes_replaced: int = 0
    api_calls_made: int = 0
    estimated_cost: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class AdjustedEbook:
    """An ebook adjusted for a specific age tier."""

    age_tier: int
    ebook: ParsedEbook
    stats: ProcessingStats
    output_path: str = ""


class EbookAdjuster:
    """
    Main orchestrator for ebook age adjustment.

    Coordinates the full pipeline:
    1. Load and parse ebook
    2. Analyze author style
    3. For each age tier:
       a. Filter profanity
       b. Detect adult content
       c. Generate replacements
       d. Apply changes
    4. Save adjusted versions
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ebook adjuster.

        Args:
            config_path: Path to configuration file.
        """
        self.config = self._load_config(config_path)

        # Initialize components
        self.converter = EbookConverter(
            calibre_path=self.config.get("conversion", {}).get("calibre_path")
        )
        self.profanity_filter = ProfanityFilter()
        self.content_analyzer = ContentAnalyzer()

        # LLM replacer initialized lazily (needs API key)
        self._llm_replacer = None

        # Setup logging
        self._setup_logging()

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file."""
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent
                / "config" / "default.yaml"
            )

        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Return default configuration."""
        return {
            "age_tiers": {
                10: {"profanity_level": "strict", "content_filter": "all"},
                13: {"profanity_level": "moderate", "content_filter": "explicit"},
                15: {"profanity_level": "mild", "content_filter": "graphic"},
                17: {"profanity_level": "minimal", "content_filter": "extreme"}
            },
            "llm": {
                "model": "claude-sonnet-4-5-20250929",
                "context_before": 1500,
                "context_after": 1500
            },
            "processing": {
                "output_format": "epub"
            }
        }

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("EbookAdjuster")

    @property
    def llm_replacer(self) -> LLMReplacer:
        """Lazy initialization of LLM replacer."""
        if self._llm_replacer is None:
            llm_config = self.config.get("llm", {})

            # Default to local inference
            use_local = llm_config.get("use_local", True)
            local_url = llm_config.get("local_url", "http://localhost:8000")
            model = llm_config.get("model")

            self._llm_replacer = LLMReplacer(
                use_local=use_local,
                local_url=local_url,
                model=model
            )
        return self._llm_replacer

    def process(
        self,
        input_file: str,
        output_dir: str,
        age_tiers: Optional[list[int]] = None
    ) -> list[AdjustedEbook]:
        """
        Process an ebook and create age-adjusted versions.

        Args:
            input_file: Path to input ebook.
            output_dir: Directory for output files.
            age_tiers: List of ages to generate (default: [10, 13, 15, 17]).

        Returns:
            List of AdjustedEbook results.
        """
        if age_tiers is None:
            age_tiers = [10, 13, 15, 17]

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing: {input_file}")
        self.logger.info(f"Age tiers: {age_tiers}")

        # Step 1: Load and parse ebook
        self.logger.info("Loading ebook...")
        parsed_ebook = self.converter.load(input_file)
        self.logger.info(
            f"Loaded: {parsed_ebook.title} by {parsed_ebook.author} "
            f"({len(parsed_ebook.chapters)} chapters)"
        )

        # Step 2: Analyze author style
        self.logger.info("Analyzing author style...")
        style_sample = self._get_style_sample(parsed_ebook)
        style_analysis = self.llm_replacer.analyze_style(style_sample)
        self.logger.info("Style analysis complete")

        # Step 3: Process each age tier
        results = []
        for age in sorted(age_tiers):
            self.logger.info(f"Processing age tier: {age}+")
            adjusted = self._process_age_tier(
                parsed_ebook, age, style_analysis, output_path
            )
            results.append(adjusted)
            self.logger.info(
                f"Age {age}+ complete: {adjusted.stats.profanity_replaced} profanity, "
                f"{adjusted.stats.scenes_replaced} scenes replaced"
            )

        # Cleanup
        self.converter.cleanup()

        return results

    def _get_style_sample(self, ebook: ParsedEbook, sample_size: int = 5000) -> str:
        """Extract a representative style sample from the ebook."""
        # Get text from middle chapters (avoid intro/outro)
        num_chapters = len(ebook.chapters)

        if num_chapters <= 2:
            chapters_to_sample = ebook.chapters
        else:
            # Sample from middle third
            start = num_chapters // 3
            end = 2 * num_chapters // 3
            chapters_to_sample = ebook.chapters[start:end]

        # Combine text
        sample_text = " ".join(ch.content for ch in chapters_to_sample)

        # Return sample from middle
        if len(sample_text) > sample_size:
            mid = len(sample_text) // 2
            half_sample = sample_size // 2
            sample_text = sample_text[mid - half_sample:mid + half_sample]

        return sample_text

    def _process_age_tier(
        self,
        original_ebook: ParsedEbook,
        age: int,
        style_analysis: StyleAnalysis,
        output_path: Path
    ) -> AdjustedEbook:
        """Process an ebook for a specific age tier."""
        stats = ProcessingStats()

        # Create copies of chapters for this age tier
        adjusted_chapters = []

        for chapter in original_ebook.chapters:
            # Create a copy of the chapter
            adjusted_chapter = Chapter(
                title=chapter.title,
                content=chapter.content,
                index=chapter.index,
                html_content=chapter.html_content,
                metadata=chapter.metadata.copy()
            )

            # Step 1: Filter profanity using in-place replacement
            filter_result = self.profanity_filter.filter_text(
                adjusted_chapter.content, age
            )

            # Apply profanity replacements in-place
            for replacement in filter_result.replacements_made:
                adjusted_chapter.replace_text(
                    replacement["original"],
                    replacement["replacement"]
                )

            stats.profanity_replaced += filter_result.word_count

            # Step 2: Analyze for adult content
            analysis = self.content_analyzer.analyze(adjusted_chapter.content, age)

            # Step 3: Replace flagged scenes
            if analysis.needs_llm_review:
                for scene in analysis.scenes:
                    # Skip false positives
                    if self.content_analyzer.is_false_positive(
                        scene, adjusted_chapter.content
                    ):
                        continue

                    # Generate replacement
                    try:
                        result = self.llm_replacer.replace_content(
                            flagged_content=scene.content,
                            context_before=scene.context_before,
                            context_after=scene.context_after,
                            content_type=scene.category,
                            target_age=age,
                            style_analysis=style_analysis
                        )

                        # Apply replacement using in-place method
                        adjusted_chapter.replace_text(
                            scene.content,
                            result.replacement
                        )

                        stats.scenes_replaced += 1
                        stats.api_calls_made += 1

                    except Exception as e:
                        error_msg = f"Failed to replace scene in chapter {chapter.index}: {e}"
                        self.logger.error(error_msg)
                        stats.errors.append(error_msg)

            adjusted_chapters.append(adjusted_chapter)
            stats.chapters_processed += 1

        # Create adjusted ebook preserving all resources
        adjusted_ebook = ParsedEbook(
            title=f"{original_ebook.title} (Age {age}+)",
            author=original_ebook.author,
            chapters=adjusted_chapters,
            metadata=original_ebook.metadata.copy(),
            original_format=original_ebook.original_format,
            cover_image=original_ebook.cover_image,
            resources=original_ebook.resources,  # Preserve CSS, images, fonts
            spine_order=original_ebook.spine_order
        )

        # Generate output filename
        safe_title = "".join(
            c for c in original_ebook.title if c.isalnum() or c in " -_"
        ).strip()
        output_format = self.config.get("processing", {}).get("output_format", "epub")
        output_file = output_path / f"{safe_title}_age{age}.{output_format}"

        # Save
        saved_path = self.converter.save(
            adjusted_ebook, str(output_file), output_format
        )

        return AdjustedEbook(
            age_tier=age,
            ebook=adjusted_ebook,
            stats=stats,
            output_path=saved_path
        )

    def estimate_processing(
        self,
        input_file: str,
        age_tiers: Optional[list[int]] = None
    ) -> dict:
        """
        Estimate processing time and cost without actually processing.

        Args:
            input_file: Path to input ebook.
            age_tiers: List of ages to estimate for.

        Returns:
            Estimation dict with costs and time.
        """
        if age_tiers is None:
            age_tiers = [10, 13, 15, 17]

        # Load ebook
        parsed_ebook = self.converter.load(input_file)

        # Estimate scenes per age tier
        total_text = " ".join(ch.content for ch in parsed_ebook.chapters)
        text_length = len(total_text)

        estimates = {}
        total_scenes = 0

        for age in age_tiers:
            analysis = self.content_analyzer.analyze(total_text, age)
            estimates[age] = {
                "scenes": analysis.total_scenes,
                "categories": analysis.categories
            }
            total_scenes += analysis.total_scenes

        # Get cost estimate
        cost_estimate = self.llm_replacer.estimate_cost(
            text_length, total_scenes
        )

        return {
            "book_info": {
                "title": parsed_ebook.title,
                "author": parsed_ebook.author,
                "chapters": len(parsed_ebook.chapters),
                "characters": text_length
            },
            "per_tier_estimates": estimates,
            "total_estimates": {
                "total_scenes": total_scenes,
                "api_calls": cost_estimate["num_api_calls"],
                "estimated_cost_usd": cost_estimate["estimated_cost_usd"],
                "estimated_time_minutes": total_scenes * 0.5  # ~30 sec per scene
            }
        }

    def generate_summary(self, results: list[AdjustedEbook]) -> str:
        """Generate a summary report of processing results."""
        lines = ["=" * 50, "EBOOK ADJUSTMENT SUMMARY", "=" * 50, ""]

        total_profanity = 0
        total_scenes = 0

        for result in results:
            lines.append(f"Age {result.age_tier}+:")
            lines.append(f"  Output: {result.output_path}")
            lines.append(f"  Profanity replaced: {result.stats.profanity_replaced}")
            lines.append(f"  Scenes replaced: {result.stats.scenes_replaced}")
            lines.append(f"  API calls: {result.stats.api_calls_made}")

            if result.stats.errors:
                lines.append(f"  Errors: {len(result.stats.errors)}")

            lines.append("")

            total_profanity += result.stats.profanity_replaced
            total_scenes += result.stats.scenes_replaced

        lines.append("=" * 50)
        lines.append(f"TOTAL: {total_profanity} profanity, {total_scenes} scenes")
        lines.append("=" * 50)

        return "\n".join(lines)
