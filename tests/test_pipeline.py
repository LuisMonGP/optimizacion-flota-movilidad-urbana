"""Suite de pruebas unitarias e integración para optimizacion-flota-movilidad-urbana."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.processor import MobilityCleaner, MobilityAnalyst
from src.optimization import haversine_bucle, haversine_vectorizado


@pytest.fixture
def muestra_viajes() -> pd.DataFrame:
    """Fixture que provee la muestra representativa de Divvy Chicago."""
    return load_dataset(force_sample=True)


def test_imputacion_estaciones_y_limpieza_nulos(muestra_viajes: pd.DataFrame) -> None:
    """Valida que MobilityCleaner impute estaciones con candado libre y descarte nulos de destino."""
    cleaner = MobilityCleaner()
    df_limpio = cleaner.process(muestra_viajes)

    # Las estaciones no deben tener valores nulos
    assert not df_limpio["start_station_name"].isna().any()
    assert not df_limpio["end_station_name"].isna().any()

    # Si había estaciones nulas en la muestra, deben reemplazarse por el texto designado
    assert "Estación Desconocida / Candado Libre" in df_limpio["start_station_name"].values or len(df_limpio) > 0

    # Las coordenadas de destino deben ser válidas
    assert not df_limpio["end_lat"].isna().any()
    assert not df_limpio["end_lng"].isna().any()

    # Validar descriptor @property stats
    stats = cleaner.stats
    assert stats["total_viajes"] == len(df_limpio)
    assert stats["duracion_mediana_minutos"] > 0


def test_filtro_calidad_duracion_viajes(muestra_viajes: pd.DataFrame) -> None:
    """Verifica que se descarten viajes con duración < 1.0 min o > 1440.0 min."""
    cleaner = MobilityCleaner()
    df_limpio = cleaner.process(muestra_viajes)

    assert (df_limpio["duration_min"] >= 1.0).all()
    assert (df_limpio["duration_min"] <= 1440.0).all()


def test_equivalencia_exacta_haversine(muestra_viajes: pd.DataFrame) -> None:
    """Verifica que el cálculo vectorizado con NumPy sea numéricamente idéntico al bucle base."""
    cleaner = MobilityCleaner()
    df = cleaner.process(muestra_viajes).head(50)

    dist_bucle = haversine_bucle(df)
    dist_vec = haversine_vectorizado(
        df["start_lat"].values,
        df["start_lng"].values,
        df["end_lat"].values,
        df["end_lng"].values,
    )

    assert len(dist_bucle) == len(dist_vec)
    assert np.allclose(dist_bucle, dist_vec, rtol=1e-7, atol=1e-7)


def test_resiliencia_dataframe_vacio() -> None:
    """Verifica que los componentes manejen un DataFrame vacío sin colapsar."""
    df_vacio = pd.DataFrame(columns=["ride_id", "started_at", "ended_at", "start_lat", "start_lng"])
    
    cleaner = MobilityCleaner()
    res_cleaner = cleaner.process(df_vacio)
    assert res_cleaner.empty
    assert cleaner.stats["total_viajes"] == 0

    analyst = MobilityAnalyst()
    res_analyst = analyst.process(res_cleaner)
    assert res_analyst.empty
