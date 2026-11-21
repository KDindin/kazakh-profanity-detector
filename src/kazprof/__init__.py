"""Kazakh profanity detection: data, training and inference utilities."""

import sys

__version__ = "0.1.0"

# Windows consoles default to cp1252 and crash on Kazakh Cyrillic in print().
# Reconfigure once here so every script and the CLI can print UTF-8 text.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # not a real terminal (pytest, redirect to file, etc.)
        pass
