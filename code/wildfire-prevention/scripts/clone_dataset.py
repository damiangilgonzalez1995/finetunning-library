"""Clona un dataset público de HuggingFace a tu propia cuenta.

Por defecto clona Paulescu/wildfire-prevention al destino que indiques.

Uso:
    uv run scripts/clone_dataset.py --target TU_USUARIO/wildfire-prevention
    uv run scripts/clone_dataset.py --target TU_USUARIO/wildfire-prevention --private
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

# Carga variables desde .env (si existe) en el directorio actual o ancestros.
# huggingface_hub lee HF_TOKEN del entorno automáticamente.
load_dotenv()

DEFAULT_SOURCE = "Paulescu/wildfire-prevention"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clona un dataset HF público a tu cuenta."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Dataset origen (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Dataset destino en tu cuenta, ej. 'damiangil/wildfire-prevention'.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./_dataset_cache",
        help="Carpeta local donde se descarga la snapshot (default: ./_dataset_cache).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crea el dataset destino como privado.",
    )
    args = parser.parse_args()

    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Descargando {args.source} -> {cache} ...")
    local_path = snapshot_download(
        repo_id=args.source,
        repo_type="dataset",
        local_dir=str(cache),
    )
    print(f"      OK: {local_path}")

    api = HfApi()

    print(f"[2/3] Creando repo destino {args.target} (private={args.private}) ...")
    api.create_repo(
        repo_id=args.target,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )
    print(f"      OK")

    print(f"[3/3] Subiendo contenidos a {args.target} ...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=args.target,
        repo_type="dataset",
        commit_message=f"Clone from {args.source}",
    )

    print(f"\nListo. Dataset disponible en:")
    print(f"  https://huggingface.co/datasets/{args.target}")


if __name__ == "__main__":
    main()
