"""
Age-Adjusted Ebooks

A pipeline system for converting ebooks into age-appropriate versions
by filtering profanity and replacing graphic content while maintaining
the original style and narrative continuity.
"""

__version__ = "0.1.0"
__author__ = "Age-Adjusted Ebooks Project"

from .pipeline.orchestrator import EbookAdjuster

__all__ = ["EbookAdjuster"]
