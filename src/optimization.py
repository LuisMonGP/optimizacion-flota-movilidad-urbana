"""Optimización Científica, Profiling y Pruebas de Hipótesis para Movilidad.

Implementa la refactorización algorítmica desde iteraciones de Pandas hacia
vectorización por broadcasting con NumPy, cuantificando la aceleración con el decorador
@monitor_rendimiento y evaluando distribuciones con SciPy.
"""

from __future__ import annotations

import functools
import time
import tracemalloc
from typing import Any, Callable, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def monitor_rendimiento(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorador que mide el tiempo de CPU y el consumo pico de memoria RAM.

    Args:
        func: Función a perfilar.

    Returns:
        Callable: Función decorada que retorna una tupla (resultado, metricas_dict).
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, float]]:
        tracemalloc.start()
        inicio = time.perf_counter()

        resultado = func(*args, **kwargs)

        duracion = time.perf_counter() - inicio
        mem_actual, mem_pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metricas = {
            "tiempo_segundos": duracion,
            "memoria_pico_kb": mem_pico / 1024.0,
            "memoria_actual_kb": mem_actual / 1024.0,
        }
        return resultado, metricas

    return wrapper


def haversine_bucle(df: pd.DataFrame) -> np.ndarray:
    """Calcula distancias Haversine iterando fila por fila con un bucle ineficiente.

    Args:
        df: DataFrame con start_lat, start_lng, end_lat, end_lng.

    Returns:
        np.ndarray: Arreglo de distancias en kilómetros.
    """
    radio_tierra = 6371.0
    distancias = []

    for _, fila in df.iterrows():
        lat1 = np.radians(fila["start_lat"])
        lon1 = np.radians(fila["start_lng"])
        lat2 = np.radians(fila["end_lat"])
        lon2 = np.radians(fila["end_lng"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        distancias.append(radio_tierra * c)

    return np.array(distancias, dtype=np.float64)


def haversine_vectorizado(
    lat1: np.ndarray | pd.Series,
    lon1: np.ndarray | pd.Series,
    lat2: np.ndarray | pd.Series,
    lon2: np.ndarray | pd.Series,
) -> np.ndarray:
    """Calcula distancias Haversine utilizando vectorización C pura con NumPy broadcasting.

    Args:
        lat1: Latitud de origen.
        lon1: Longitud de origen.
        lat2: Latitud de destino.
        lon2: Longitud de destino.

    Returns:
        np.ndarray: Distancias geodésicas en kilómetros.
    """
    radio_tierra = 6371.0

    r_lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    r_lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    r_lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    r_lon2 = np.radians(np.asarray(lon2, dtype=np.float64))

    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radio_tierra * c


@monitor_rendimiento
def benchmark_ejecucion_bucle(df: pd.DataFrame) -> np.ndarray:
    """Wrapper de benchmarking para la versión iterativa en bucle."""
    return haversine_bucle(df)


@monitor_rendimiento
def benchmark_ejecucion_vectorizada(df: pd.DataFrame) -> np.ndarray:
    """Wrapper de benchmarking para la versión vectorizada con NumPy."""
    return haversine_vectorizado(
        df["start_lat"].values,
        df["start_lng"].values,
        df["end_lat"].values,
        df["end_lng"].values,
    )


def prueba_hipotesis_duracion_usuarios(df: pd.DataFrame) -> Dict[str, Any]:
    """Evalúa estadísticamente si existen diferencias significativas de duración entre member y casual.

    Args:
        df: DataFrame limpio con 'member_casual' y 'duration_min'.

    Returns:
        Dict[str, Any]: Resultados estadísticos y valor p de rechazo.
    """
    miembros = df.loc[df["member_casual"] == "member", "duration_min"].dropna()
    ocasionales = df.loc[df["member_casual"] == "casual", "duration_min"].dropna()

    if len(miembros) < 2 or len(ocasionales) < 2:
        return {"error": "Muestra insuficiente para realizar pruebas estadísticas"}

    # Prueba t de Welch (varianzas desiguales)
    t_stat, t_pval = stats.ttest_ind(miembros, ocasionales, equal_var=False)
    # Prueba no paramétrica Mann-Whitney U
    u_stat, u_pval = stats.mannwhitneyu(miembros, ocasionales, alternative="two-sided")

    return {
        "n_miembros": len(miembros),
        "n_ocasionales": len(ocasionales),
        "media_miembros_min": float(np.mean(miembros)),
        "media_ocasionales_min": float(np.mean(ocasionales)),
        "welch_t_stat": float(t_stat),
        "welch_p_valor": float(t_pval),
        "mann_whitney_u_stat": float(u_stat),
        "mann_whitney_p_valor": float(u_pval),
        "diferencia_significativa": bool(u_pval < 0.05),
    }
