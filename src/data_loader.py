"""Cargador de datos e ingesta adaptativa para viajes de micromovilidad Divvy Chicago.

Implementa el patrón de adaptador profesional: intenta cargar el dataset orgánico
completo desde data/raw/202401-divvy-tripdata.csv; si no existe o se fuerza el modo muestra,
utiliza la muestra representativa de data/sample/sample_data.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "202401-divvy-tripdata.csv"
SAMPLE_PATH = BASE_DIR / "data" / "sample" / "sample_data.csv"


def load_dataset(force_sample: bool = False, nrows: Optional[int] = None) -> Any:
    """Carga el dataset orgánico de Divvy Chicago o la muestra representativa de CI/CD.

    Args:
        force_sample: Si es True, fuerza la carga de data/sample/sample_data.csv.
        nrows: Cantidad opcional de filas a leer.

    Returns:
        pd.DataFrame | List[dict]: Dataset cargado con tipos de datos correctos.

    Raises:
        FileNotFoundError: Si no se encuentra ni la data cruda ni la muestra.
    """
    target_path = None
    if not force_sample and RAW_PATH.exists():
        target_path = RAW_PATH
        print(f"[INFO] Cargando dataset orgánico completo desde: {RAW_PATH}")
    elif SAMPLE_PATH.exists():
        target_path = SAMPLE_PATH
        print(f"[INFO] Cargando muestra representativa reproducible desde: {SAMPLE_PATH}")
    else:
        raise FileNotFoundError(
            f"No se encontró dataset crudo en '{RAW_PATH}' ni muestra en '{SAMPLE_PATH}'."
        )

    if HAS_PANDAS:
        df = pd.read_csv(target_path, nrows=nrows)
        # Parseo de fechas oficiales Divvy
        if "started_at" in df.columns:
            df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        if "ended_at" in df.columns:
            df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
        return df

    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)[:nrows] if nrows else list(reader)


if __name__ == "__main__":
    data = load_dataset()
    if HAS_PANDAS:
        print(f"Dimensiones cargadas: {data.shape}")
        print(f"Columnas detectadas: {list(data.columns)}")
    else:
        print(f"Total registros cargados: {len(data)}")
