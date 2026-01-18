"""
Ebook format conversion and parsing module.

Handles conversion between ebook formats (EPUB, MOBI, PDF) and
extraction of text content while preserving structure and formatting.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from copy import deepcopy

from bs4 import BeautifulSoup, NavigableString
import ebooklib
from ebooklib import epub


@dataclass
class Chapter:
    """Represents a single chapter from an ebook."""

    title: str
    content: str  # Plain text for analysis
    index: int
    html_content: str = ""  # Original HTML
    metadata: dict = field(default_factory=dict)

    def replace_text(self, old_text: str, new_text: str) -> bool:
        """
        Replace text in both content and html_content.

        Does in-place replacement preserving HTML structure.

        Args:
            old_text: Text to find and replace.
            new_text: Replacement text.

        Returns:
            True if replacement was made.
        """
        replaced = False

        # Replace in plain text
        if old_text in self.content:
            self.content = self.content.replace(old_text, new_text)
            replaced = True

        # Replace in HTML while preserving tags
        if old_text in self.html_content:
            self.html_content = self._replace_in_html(
                self.html_content, old_text, new_text
            )
            replaced = True

        return replaced

    def _replace_in_html(self, html: str, old_text: str, new_text: str) -> str:
        """
        Replace text within HTML while preserving tags.

        This method carefully replaces text content without affecting
        HTML structure, attributes, or formatting tags.
        """
        # Parse HTML
        soup = BeautifulSoup(html, "lxml")

        # Find all text nodes
        def replace_text_in_element(element):
            if isinstance(element, NavigableString):
                if old_text in element:
                    new_string = element.replace(old_text, new_text)
                    element.replace_with(new_string)
            else:
                for child in list(element.children):
                    replace_text_in_element(child)

        replace_text_in_element(soup)

        # Return modified HTML, preserving original structure
        return str(soup)


@dataclass
class EbookResource:
    """Represents a non-chapter resource (CSS, images, fonts)."""

    name: str
    content: bytes
    media_type: str


@dataclass
class ParsedEbook:
    """Represents a fully parsed ebook with all content and metadata."""

    title: str
    author: str
    chapters: list[Chapter]
    metadata: dict
    original_format: str
    cover_image: Optional[bytes] = None
    resources: list[EbookResource] = field(default_factory=list)
    spine_order: list[str] = field(default_factory=list)


class EbookConverter:
    """
    Converts ebooks to editable format and back.

    Preserves original formatting by doing in-place text replacement
    within the existing HTML structure.
    """

    def __init__(self, calibre_path: Optional[str] = None):
        """
        Initialize the ebook converter.

        Args:
            calibre_path: Path to Calibre's ebook-convert. Auto-detects if None.
        """
        self.calibre_path = calibre_path or self._find_calibre()
        self.temp_dir = None

    def _find_calibre(self) -> str:
        """Find the Calibre ebook-convert executable."""
        paths = [
            "/usr/bin/ebook-convert",
            "/usr/local/bin/ebook-convert",
            "/Applications/calibre.app/Contents/MacOS/ebook-convert",
            "C:\\Program Files\\Calibre2\\ebook-convert.exe",
            "C:\\Program Files (x86)\\Calibre2\\ebook-convert.exe",
        ]

        for path in paths:
            if os.path.exists(path):
                return path

        try:
            result = subprocess.run(
                ["which", "ebook-convert"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return "ebook-convert"

    def load(self, file_path: str) -> ParsedEbook:
        """
        Load and parse an ebook file.

        Args:
            file_path: Path to the ebook file.

        Returns:
            ParsedEbook with all content extracted.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Ebook not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".epub":
            return self._load_epub(file_path)
        elif suffix in [".mobi", ".azw", ".azw3"]:
            return self._load_via_conversion(file_path)
        elif suffix == ".pdf":
            return self._load_via_conversion(file_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _load_epub(self, file_path: Path) -> ParsedEbook:
        """Load and parse an EPUB file directly."""
        book = epub.read_epub(str(file_path))

        # Extract metadata
        title = self._get_metadata(book, "title") or file_path.stem
        author = self._get_metadata(book, "creator") or "Unknown"

        metadata = {
            "language": self._get_metadata(book, "language"),
            "publisher": self._get_metadata(book, "publisher"),
            "identifier": self._get_metadata(book, "identifier"),
        }

        # Extract resources (CSS, images, fonts)
        resources = []
        for item in book.get_items():
            if item.get_type() in [ebooklib.ITEM_STYLE, ebooklib.ITEM_IMAGE, ebooklib.ITEM_FONT]:
                resources.append(EbookResource(
                    name=item.get_name(),
                    content=item.get_content(),
                    media_type=item.media_type
                ))

        # Extract chapters
        chapters = []
        spine_order = []

        for idx, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
            html_content = item.get_content().decode("utf-8")
            soup = BeautifulSoup(html_content, "lxml")

            # Extract title from HTML or use filename
            chapter_title = ""
            title_tag = soup.find(["h1", "h2", "h3", "title"])
            if title_tag:
                chapter_title = title_tag.get_text(strip=True)
            if not chapter_title:
                chapter_title = f"Chapter {idx + 1}"

            # Extract text content for analysis
            text_content = soup.get_text(separator="\n", strip=True)

            if text_content.strip():
                chapters.append(Chapter(
                    title=chapter_title,
                    content=text_content,
                    index=idx,
                    html_content=html_content,
                    metadata={
                        "filename": item.get_name(),
                        "media_type": item.media_type
                    }
                ))
                spine_order.append(item.get_name())

        # Extract cover image
        cover_image = None
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            if "cover" in item.get_name().lower():
                cover_image = item.get_content()
                break

        return ParsedEbook(
            title=title,
            author=author,
            chapters=chapters,
            metadata=metadata,
            original_format="epub",
            cover_image=cover_image,
            resources=resources,
            spine_order=spine_order
        )

    def _load_via_conversion(self, file_path: Path) -> ParsedEbook:
        """Load a non-EPUB file by converting to EPUB first."""
        self.temp_dir = tempfile.mkdtemp(prefix="ebook_adjust_")
        temp_epub = Path(self.temp_dir) / "converted.epub"

        try:
            result = subprocess.run(
                [self.calibre_path, str(file_path), str(temp_epub)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Conversion failed: {result.stderr}")

            parsed = self._load_epub(temp_epub)
            parsed.original_format = file_path.suffix.lower().strip(".")

            return parsed

        except subprocess.TimeoutExpired:
            raise RuntimeError("Conversion timed out")

    def _get_metadata(self, book: epub.EpubBook, key: str) -> Optional[str]:
        """Extract metadata from EPUB book."""
        try:
            metadata = book.get_metadata("DC", key)
            if metadata:
                return metadata[0][0]
        except Exception:
            pass
        return None

    def save(
        self,
        parsed_ebook: ParsedEbook,
        output_path: str,
        output_format: str = "epub"
    ) -> str:
        """
        Save a parsed ebook to file.

        Args:
            parsed_ebook: The ParsedEbook to save.
            output_path: Path for the output file.
            output_format: Output format (epub, mobi, pdf).

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)

        # First save as EPUB
        epub_path = output_path.with_suffix(".epub")
        self._save_epub(parsed_ebook, epub_path)

        # Convert if needed
        if output_format.lower() != "epub":
            final_path = output_path.with_suffix(f".{output_format}")
            self._convert_format(epub_path, final_path)

            if epub_path != final_path:
                epub_path.unlink()

            return str(final_path)

        return str(epub_path)

    def _save_epub(self, parsed_ebook: ParsedEbook, output_path: Path) -> None:
        """
        Save ParsedEbook as EPUB file.

        Preserves original HTML structure and resources.
        """
        book = epub.EpubBook()

        # Set metadata
        book.set_title(parsed_ebook.title)
        book.add_author(parsed_ebook.author)
        book.set_language(parsed_ebook.metadata.get("language") or "en")

        if parsed_ebook.metadata.get("identifier"):
            book.set_identifier(parsed_ebook.metadata["identifier"])

        # Add resources (CSS, images, fonts) first
        for resource in parsed_ebook.resources:
            if resource.media_type.startswith("text/css"):
                item = epub.EpubItem(
                    file_name=resource.name,
                    media_type=resource.media_type,
                    content=resource.content
                )
                book.add_item(item)
            elif resource.media_type.startswith("image/"):
                item = epub.EpubItem(
                    file_name=resource.name,
                    media_type=resource.media_type,
                    content=resource.content
                )
                book.add_item(item)
            elif resource.media_type.startswith("font/") or resource.media_type.startswith("application/font"):
                item = epub.EpubItem(
                    file_name=resource.name,
                    media_type=resource.media_type,
                    content=resource.content
                )
                book.add_item(item)

        # Add chapters with original filenames and HTML
        epub_chapters = []
        for chapter in parsed_ebook.chapters:
            # Use original filename if available
            filename = chapter.metadata.get("filename", f"chapter_{chapter.index}.xhtml")

            epub_chapter = epub.EpubHtml(
                title=chapter.title,
                file_name=filename,
                lang=parsed_ebook.metadata.get("language") or "en"
            )

            # Use the HTML content directly - it's already been modified in place
            if chapter.html_content:
                # Ensure it's proper XHTML
                epub_chapter.content = chapter.html_content
            else:
                # Fallback: create HTML from text (shouldn't happen normally)
                paragraphs = chapter.content.split("\n\n")
                html_paragraphs = [
                    f"<p>{self._escape_html(p)}</p>"
                    for p in paragraphs if p.strip()
                ]
                epub_chapter.content = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{self._escape_html(chapter.title)}</title>
</head>
<body>
    <h1>{self._escape_html(chapter.title)}</h1>
    {"".join(html_paragraphs)}
</body>
</html>"""

            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)

        # Add navigation
        book.toc = [
            (epub.Section(ch.title), [epub_ch])
            for ch, epub_ch in zip(parsed_ebook.chapters, epub_chapters)
        ]

        # Add required items
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Set spine (reading order)
        book.spine = ["nav"] + epub_chapters

        # Add cover if exists
        if parsed_ebook.cover_image:
            book.set_cover("cover.jpg", parsed_ebook.cover_image)

        # Write file
        epub.write_epub(str(output_path), book)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _convert_format(self, input_path: Path, output_path: Path) -> None:
        """Convert between ebook formats using Calibre."""
        result = subprocess.run(
            [self.calibre_path, str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            raise RuntimeError(f"Format conversion failed: {result.stderr}")

    def cleanup(self) -> None:
        """Clean up any temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def replace_text_preserving_format(
    chapters: list[Chapter],
    old_text: str,
    new_text: str
) -> int:
    """
    Replace text across all chapters while preserving formatting.

    This is the primary function for making content replacements.

    Args:
        chapters: List of chapters to modify in place.
        old_text: Text to find and replace.
        new_text: Replacement text.

    Returns:
        Number of replacements made.
    """
    count = 0
    for chapter in chapters:
        if chapter.replace_text(old_text, new_text):
            count += 1
    return count
