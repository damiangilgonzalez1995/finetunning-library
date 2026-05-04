"""
Genera una nota Markdown por cada concepto del índice en `conceptos/`.
Cada nota enlaza al capítulo central y a los capítulos donde el concepto también aparece,
de modo que Obsidian los muestre como nodos-hub en el graph view.

Uso:
    python scripts/generate-concept-notes.py

Idempotente: reescribe las notas cada vez. Las notas editadas a mano se perderán,
así que si quieres extender una nota, añade una sección `## Notas personales`
marcada con `<!-- no-regen -->` y adapta el script para preservar ese bloque.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = ROOT / "assets" / "indice-conceptos.yml"
CONCEPTS_DIR = ROOT / "conceptos"

TIPO_DESCRIPCION = {
    "técnica": "Técnica de entrenamiento o fine-tuning.",
    "concepto": "Concepto teórico / definición fundamental.",
    "modelo": "Arquitectura o modelo concreto.",
    "herramienta": "Librería, framework o utilidad.",
}


def cap_slug(cap_num: str, capitulos: dict) -> str | None:
    entry = capitulos.get(cap_num)
    if not entry:
        return None
    return entry.get("slug")


def render_concept(nombre: str, data: dict, capitulos: dict) -> str:
    tipo = data.get("tipo", "concepto")
    capitulo_central = data.get("capitulo_central")
    tambien_en = data.get("tambien_en") or []

    central_slug = cap_slug(capitulo_central, capitulos) if capitulo_central else None
    tambien_slugs = [cap_slug(c, capitulos) for c in tambien_en]
    tambien_slugs = [s for s in tambien_slugs if s]

    # Frontmatter
    fm_lines = ["---", f"concepto: {nombre}", f"tipo: {tipo}"]
    if central_slug:
        fm_lines.append(f'capitulo_central: "[[{central_slug}]]"')
    else:
        fm_lines.append("capitulo_central: null")

    if tambien_slugs:
        fm_lines.append("aparece_en:")
        for s in tambien_slugs:
            fm_lines.append(f'  - "[[{s}]]"')
    else:
        fm_lines.append("aparece_en: []")

    fm_lines.append(f"tags:")
    fm_lines.append(f"  - tipo/{tipo}")
    fm_lines.append(f"  - concepto/{nombre}")
    fm_lines.append(f'generado: "{date.today().isoformat()}"')
    fm_lines.append("---")

    # Cuerpo
    body = [
        "",
        f"# {nombre}",
        "",
        f"> {TIPO_DESCRIPCION.get(tipo, 'Concepto.')}",
        "",
    ]

    body.append("## Capítulo central")
    body.append("")
    if central_slug:
        body.append(f"[[{central_slug}]]")
    else:
        body.append("*Este concepto aún no tiene un capítulo dedicado en el libro; aparece solo mencionado.*")
    body.append("")

    if tambien_slugs:
        body.append("## Aparece también en")
        body.append("")
        for s in tambien_slugs:
            body.append(f"- [[{s}]]")
        body.append("")

    body.append("## Tipo")
    body.append("")
    body.append(f"`{tipo}`")
    body.append("")

    body.append("---")
    body.append("")
    body.append(f"#tipo/{tipo} #concepto/{nombre}")
    body.append("")

    return "\n".join(fm_lines) + "\n" + "\n".join(body)


def main() -> int:
    if not INDEX_FILE.exists():
        print(f"[ERROR] No existe el índice: {INDEX_FILE}", file=sys.stderr)
        return 1

    with INDEX_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    capitulos = data.get("capitulos", {}) or {}
    conceptos = data.get("conceptos", {}) or {}

    if not conceptos:
        print("[WARN] No hay conceptos en el índice todavía.")
        return 0

    CONCEPTS_DIR.mkdir(exist_ok=True)

    created = 0
    for nombre, concept_data in conceptos.items():
        content = render_concept(nombre, concept_data, capitulos)
        out_path = CONCEPTS_DIR / f"{nombre}.md"
        out_path.write_text(content, encoding="utf-8")
        created += 1

    print(f"[OK] Generadas {created} notas en {CONCEPTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
