"""Пакет ``mkbase`` — сборка базы сечений (console-script ``mkbase``).

Точка входа — :func:`mkbase.mkbase.main`; модуль также запускается напрямую::

    python src/mkbase/mkbase.py results_aggl -o base
"""

from .mkbase import main

__all__ = ["main"]
