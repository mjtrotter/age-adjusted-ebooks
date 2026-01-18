#!/usr/bin/env python3
"""
Age-Adjusted Ebooks CLI

Command-line interface for converting ebooks to age-appropriate versions.
"""

import sys
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .pipeline.orchestrator import EbookAdjuster

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Age-Adjusted Ebooks - Create age-appropriate book versions."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./output",
    help="Output directory for adjusted ebooks"
)
@click.option(
    "--ages", "-a",
    default="10,13,15,17",
    help="Comma-separated age tiers to generate (default: 10,13,15,17)"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Custom configuration file"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Estimate processing without making changes"
)
def process(input_file, output, ages, config, dry_run):
    """
    Process an ebook and create age-adjusted versions.

    INPUT_FILE: Path to the ebook file (epub, mobi, or pdf)
    """
    # Parse age tiers
    try:
        age_tiers = [int(a.strip()) for a in ages.split(",")]
    except ValueError:
        console.print("[red]Error: Ages must be comma-separated numbers[/red]")
        sys.exit(1)

    # Initialize adjuster
    try:
        adjuster = EbookAdjuster(config_path=config)
    except Exception as e:
        console.print(f"[red]Error initializing: {e}[/red]")
        sys.exit(1)

    if dry_run:
        # Just estimate
        console.print("\n[yellow]DRY RUN - Estimating processing...[/yellow]\n")

        try:
            estimates = adjuster.estimate_processing(input_file, age_tiers)
            _display_estimates(estimates)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

        return

    # Process the ebook
    console.print(f"\n[bold]Processing:[/bold] {input_file}")
    console.print(f"[bold]Age tiers:[/bold] {age_tiers}")
    console.print(f"[bold]Output:[/bold] {output}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing ebook...", total=None)

        try:
            results = adjuster.process(
                input_file=input_file,
                output_dir=output,
                age_tiers=age_tiers
            )
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            sys.exit(1)

        progress.update(task, completed=True)

    # Display results
    console.print("\n[green]Processing complete![/green]\n")
    _display_results(results)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--age", "-a",
    type=int,
    default=15,
    help="Age tier to analyze for (default: 15)"
)
def analyze(input_file, age):
    """
    Analyze an ebook for content without processing.

    Shows what content would be flagged for the given age tier.
    """
    from .converters.ebook_converter import EbookConverter
    from .analyzers.content_analyzer import ContentAnalyzer
    from .filters.profanity_filter import ProfanityFilter

    console.print(f"\n[bold]Analyzing:[/bold] {input_file}")
    console.print(f"[bold]Age tier:[/bold] {age}+\n")

    # Load ebook
    converter = EbookConverter()
    ebook = converter.load(input_file)

    # Analyze
    analyzer = ContentAnalyzer()
    profanity_filter = ProfanityFilter()

    total_text = " ".join(ch.content for ch in ebook.chapters)

    # Get profanity count
    profanity_counts = profanity_filter.get_word_count(total_text, age)

    # Get content analysis
    content_result = analyzer.analyze(total_text, age)

    # Display results
    console.print(f"[bold]Book:[/bold] {ebook.title} by {ebook.author}")
    console.print(f"[bold]Chapters:[/bold] {len(ebook.chapters)}")
    console.print(f"[bold]Length:[/bold] {len(total_text):,} characters\n")

    # Profanity table
    table = Table(title="Profanity Analysis")
    table.add_column("Tier", style="cyan")
    table.add_column("Count", justify="right")

    for tier, count in profanity_counts.items():
        if tier != "total":
            table.add_row(tier, str(count))

    table.add_row("[bold]Total[/bold]", f"[bold]{profanity_counts['total']}[/bold]")
    console.print(table)
    console.print()

    # Content analysis
    console.print(f"[bold]Adult Content Scenes:[/bold] {content_result.total_scenes}\n")

    if content_result.scenes:
        table = Table(title="Detected Scenes")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Category")
        table.add_column("Intensity", justify="right")
        table.add_column("Length", justify="right")

        for i, scene in enumerate(content_result.scenes[:10], 1):
            table.add_row(
                str(i),
                scene.category,
                f"{scene.intensity_score:.2f}",
                f"{len(scene.content):,}"
            )

        console.print(table)

        if len(content_result.scenes) > 10:
            console.print(
                f"\n[dim]... and {len(content_result.scenes) - 10} more scenes[/dim]"
            )

    converter.cleanup()


@cli.command()
@click.argument("word")
@click.option(
    "--tier", "-t",
    default="tier_13_moderate",
    help="Tier to add word to"
)
@click.option(
    "--replacements", "-r",
    default="",
    help="Comma-separated replacement words"
)
def add_word(word, tier, replacements):
    """Add a custom word to the profanity filter."""
    from .filters.profanity_filter import ProfanityFilter

    filter = ProfanityFilter()

    replacement_list = [r.strip() for r in replacements.split(",") if r.strip()]
    if not replacement_list:
        replacement_list = [f"[{word}]"]

    filter.add_custom_words(tier, {word: replacement_list})

    console.print(f"[green]Added '{word}' to {tier}[/green]")
    console.print(f"Replacements: {replacement_list}")


def _display_estimates(estimates: dict) -> None:
    """Display processing estimates."""
    book = estimates["book_info"]

    console.print(f"[bold]Book:[/bold] {book['title']} by {book['author']}")
    console.print(f"[bold]Chapters:[/bold] {book['chapters']}")
    console.print(f"[bold]Characters:[/bold] {book['characters']:,}\n")

    # Per-tier estimates
    table = Table(title="Estimated Scenes by Age Tier")
    table.add_column("Age", justify="right", style="cyan")
    table.add_column("Scenes", justify="right")
    table.add_column("Sexual", justify="right")
    table.add_column("Violence", justify="right")
    table.add_column("Intoxication", justify="right")

    for age, data in estimates["per_tier_estimates"].items():
        cats = data.get("categories", {})
        table.add_row(
            f"{age}+",
            str(data["scenes"]),
            str(cats.get("sexual_content", 0)),
            str(cats.get("violence_extreme", 0)),
            str(cats.get("intoxication", 0))
        )

    console.print(table)
    console.print()

    # Totals
    totals = estimates["total_estimates"]
    console.print(f"[bold]Total scenes to process:[/bold] {totals['total_scenes']}")
    console.print(f"[bold]API calls:[/bold] {totals['api_calls']}")
    console.print(f"[bold]Estimated cost:[/bold] ${totals['estimated_cost_usd']:.2f}")
    console.print(f"[bold]Estimated time:[/bold] {totals['estimated_time_minutes']:.0f} minutes")


def _display_results(results: list) -> None:
    """Display processing results."""
    table = Table(title="Processing Results")
    table.add_column("Age", justify="right", style="cyan")
    table.add_column("Profanity", justify="right")
    table.add_column("Scenes", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Output File")

    for result in results:
        table.add_row(
            f"{result.age_tier}+",
            str(result.stats.profanity_replaced),
            str(result.stats.scenes_replaced),
            str(len(result.stats.errors)) if result.stats.errors else "-",
            Path(result.output_path).name
        )

    console.print(table)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
