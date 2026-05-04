#!/usr/bin/env python3
"""Genera el Word de la revisión final como documento separado."""

from pathlib import Path

# Reutilizamos todo el motor del build principal
import importlib.util
spec = importlib.util.spec_from_file_location("bw", Path(__file__).parent / "build-word.py")
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)

SRC = Path(__file__).resolve().parent.parent / "capitulos" / "06-revision-final.md"
DST = Path(__file__).resolve().parent.parent / "output" / "revision-final.docx"

DST.parent.mkdir(parents=True, exist_ok=True)

print(f"Convirtiendo {SRC.name} -> {DST.name}")
bw.markdown_to_docx(SRC, DST)
print("Listo.")
