"""Ingeniería de Variables Geoespaciales y Temporales con GeoPandas.

Convierte coordenadas latitud/longitud en geometrías espaciales Point WGS84
y genera métricas de demanda en ventanas móviles cronológicas.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


def crear_geodataframe_viajes(
    df: pd.DataFrame,
    lat_col: str = "start_lat",
    lng_col: str = "start_lng",
) -> gpd.GeoDataFrame:
    """Convierte un DataFrame con coordenadas en un GeoDataFrame con geometrías Point.

    Args:
        df: DataFrame de viajes de Divvy.
        lat_col: Nombre de la columna de latitud de inicio.
        lng_col: Nombre de la columna de longitud de inicio.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame espacial con proyección EPSG:4326.
    """
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    valido = df[lat_col].notna() & df[lng_col].notna()
    df_valido = df[valido].copy()

    gdf = gpd.GeoDataFrame(
        df_valido,
        geometry=gpd.points_from_xy(df_valido[lng_col], df_valido[lat_col]),
        crs="EPSG:4326",
    )
    return gdf


def calcular_ventanas_moviles_demanda(
    df: pd.DataFrame,
    ventana: int = 5,
) -> pd.DataFrame:
    """Calcula la media móvil y desviación estándar de duración de viajes.

    Args:
        df: DataFrame de viajes ordenado en el tiempo.
        ventana: Tamaño de la ventana móvil de observaciones.

    Returns:
        pd.DataFrame: DataFrame con las columnas rolling añadidas.
    """
    if df.empty or "duration_min" not in df.columns:
        return df.copy()

    df_out = df.sort_values("started_at").copy() if "started_at" in df.columns else df.copy()
    df_out["rolling_duracion_media"] = (
        df_out["duration_min"].rolling(window=ventana, min_periods=1).mean()
    )
    df_out["rolling_duracion_std"] = (
        df_out["duration_min"].rolling(window=ventana, min_periods=1).std().fillna(0.0)
    )
    return df_out
