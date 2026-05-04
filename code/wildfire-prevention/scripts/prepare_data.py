"""Convierte el dataset clonado en HuggingFace al formato JSONL que espera leap-finetune.

Lee `tu-usuario/wildfire-prevention` desde HF (o re-utiliza un cache local) y
genera dos archivos JSONL (`train.jsonl` y `test.jsonl`) en formato VLM SFT
"messages", con paths a imágenes que viven en `images/`.

Uso:
    uv run scripts/prepare_data.py --dataset damianGil/wildfire-prevention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

# Estos prompts vienen del cookbook original de Liquid (annotator.py).
# Son LOS MISMOS que se usaron para etiquetar el dataset con Opus, así que
# durante el fine-tuning queremos que el modelo aprenda a responder ANTE
# este mismo input.
SYSTEM_PROMPT = """\
You are a remote sensing analyst specialising in wildfire risk assessment.
You will be given two Sentinel-2 satellite images of the same land tile:
  1. RGB composite (bands B4-B3-B2): natural colour, useful for terrain, \
infrastructure, and land cover.
  2. SWIR composite (bands B12-B8-B4): highlights vegetation moisture stress \
and dryness. Healthy vegetation appears green/cyan, stressed or dry vegetation \
appears orange/red, bare soil appears magenta/pink, and burned areas appear \
dark red or black.

Assess the wildfire risk of the tile and return ONLY a valid JSON object — \
no markdown, no explanation outside the JSON — with exactly these fields:

{
  "risk_level": "low | medium | high",
  "dry_vegetation_present": true | false,
  "urban_interface": true | false,
  "steep_terrain": true | false,
  "water_body_present": true | false,
  "image_quality_limited": true | false
}

Field definitions:
- risk_level: overall wildfire risk for the tile, using these criteria:
    - low: no dry vegetation, or landscape is predominantly wet/green/bare rock
    - medium: some dry vegetation present but fuel continuity is broken by \
bare soil, water bodies, or green vegetation
    - high: extensive dry vegetation with continuous fuel load and at least \
one aggravating factor (steep terrain or urban interface)
- dry_vegetation_present: dry grass, shrubland, cropland stubble, or any \
vegetation showing low moisture (orange/red in SWIR).
- urban_interface: buildings, roads, or infrastructure adjacent to or within \
dry vegetation.
- steep_terrain: visible ridges, slopes, or canyons that would accelerate \
fire spread.
- water_body_present: river, reservoir, or lake that acts as a natural \
firebreak.
- image_quality_limited: cloud, snow, or no-data obscures a significant \
portion of the tile.
"""

USER_TEXT = (
    "Image 1 is the RGB composite. Image 2 is the SWIR composite. "
    "Return the wildfire risk JSON for this tile."
)


def make_vlm_row(rgb_name: str, swir_name: str, output: str) -> dict:
    """Devuelve UNA fila en formato VLM SFT messages.

    El campo `image` es solo el filename (sin path) — el config YAML define
    `image_root` que se concatena al cargar.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_name},
                    {"type": "image", "image": swir_name},
                    {"type": "text", "text": f"{SYSTEM_PROMPT.strip()}\n\n{USER_TEXT}"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output}],
            },
        ]
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"  Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte el dataset HF a formato leap-finetune VLM SFT."
    )
    parser.add_argument(
        "--dataset",
        default="damianGil/wildfire-prevention",
        help="Dataset HF (default: damianGil/wildfire-prevention).",
    )
    parser.add_argument(
        "--output",
        default="./data/wildfire",
        help="Carpeta de salida para los JSONL (default: ./data/wildfire).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Descargando snapshot de {args.dataset} a {output_dir} ...")
    snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta images/ en {output_dir}")
    n_images = sum(1 for _ in images_dir.iterdir())
    print(f"      OK ({n_images} imagenes en {images_dir})")

    print(f"[2/2] Cargando dataset y convirtiendo a JSONL ...")
    ds = load_dataset(str(output_dir))

    for split_name in ("train", "test"):
        if split_name not in ds:
            print(f"      Split '{split_name}' no encontrado, saltando.")
            continue

        rows = []
        for row in ds[split_name]:
            rgb_name = Path(str(row["rgb_path"])).name    # solo filename, no path
            swir_name = Path(str(row["swir_path"])).name
            output_str = str(row["output"])
            rows.append(make_vlm_row(rgb_name, swir_name, output_str))

        write_jsonl(rows, output_dir / f"wildfire_{split_name}.jsonl")

    print(f"\nListo. Set image_root: {images_dir}")
    print(f"JSONLs en: {output_dir}/wildfire_train.jsonl y wildfire_test.jsonl")


if __name__ == "__main__":
    main()
