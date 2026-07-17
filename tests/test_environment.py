"""Tests de entorno — verifica que el proyecto está bien configurado."""

import importlib
from pathlib import Path


def test_dependencias_principales():
    """pandas y pytest deben estar instalados y ser importables."""
    import pandas  # noqa: F401
    import pytest  # noqa: F401


def test_estructura_del_proyecto():
    """El proyecto debe tener la estructura src-layout esperada."""
    root = Path(__file__).parent.parent
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "analytics").is_dir()
    assert (root / "tests").is_dir()


def test_paquete_importable():
    """El paquete analytics debe poder importarse."""
    modulo = importlib.import_module("analytics")
    assert modulo is not None
