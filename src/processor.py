"""Arquitectura de Procesamiento Orientada a Objetos para Datos de Micromovilidad.

Implementa un pipeline modular bajo los principios de POO avanzada (ABC, Mixins
y descriptores de propiedades) para procesar datos de viajes de Divvy Chicago.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import datetime
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class LoggerMixin:
    """Mixin que provee capacidades estándar de logging a los componentes de datos."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def log_info(self, mensaje: str) -> None:
        """Registra un evento informativo."""
        self._logger.info(mensaje)

    def log_warning(self, mensaje: str) -> None:
        """Registra una advertencia de datos."""
        self._logger.warning(mensaje)


class DataComponent(ABC):
    """Clase Base Abstracta que define el contrato de transformación de datos."""

    def __init__(self, nombre: str = "ComponenteBase") -> None:
        self.nombre = nombre
        self._timestamp_ejecucion: Optional[datetime.datetime] = None

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformaciones de datos o lógica de negocio.

        Args:
            data: DataFrame de entrada.

        Returns:
            pd.DataFrame: DataFrame transformado.
        """
        pass

    @property
    def timestamp_ejecucion(self) -> Optional[datetime.datetime]:
        """Retorna la fecha y hora de la última ejecución."""
        return self._timestamp_ejecucion


class MobilityCleaner(DataComponent, LoggerMixin):
    """Limpieza, imputación e ingeniería de variables para viajes de Divvy Chicago."""

    def __init__(self) -> None:
        DataComponent.__init__(self, nombre="MobilityCleaner")
        LoggerMixin.__init__(self)
        self._ultimo_df: Optional[pd.DataFrame] = None
        self._registros_filtrados: int = 0

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ejecuta la limpieza e imputación según las reglas de negocio de Divvy.

        Args:
            data: DataFrame de viajes de Divvy con columnas oficiales.

        Returns:
            pd.DataFrame: DataFrame filtrado e imputado con variables enriquecidas.
        """
        self.log_info(f"Iniciando limpieza sobre {len(data)} registros de viajes.")
        self._timestamp_ejecucion = datetime.datetime.now(datetime.timezone.utc)

        if data.empty:
            self.log_warning("DataFrame vacío proporcionado. Retornando copia intacta.")
            self._ultimo_df = data.copy()
            return self._ultimo_df

        df = data.copy()
        filas_iniciales = len(df)

        # 1. Imputación de estaciones desconocidas (bicicletas con candado libre)
        if "start_station_name" in df.columns:
            df["start_station_name"] = df["start_station_name"].fillna("Estación Desconocida / Candado Libre")
        if "end_station_name" in df.columns:
            df["end_station_name"] = df["end_station_name"].fillna("Estación Desconocida / Candado Libre")

        # 2. Descarte de viajes sin coordenadas de destino
        cols_coord = [c for c in ["end_lat", "end_lng"] if c in df.columns]
        if cols_coord:
            df = df.dropna(subset=cols_coord)

        # 3. Vectorización de duración en minutos
        if "started_at" in df.columns and "ended_at" in df.columns:
            st = pd.to_datetime(df["started_at"], errors="coerce")
            et = pd.to_datetime(df["ended_at"], errors="coerce")
            df["duration_min"] = ((et - st).dt.total_seconds() / 60.0).round(2)

            # Filtro de calidad de negocio: 1.0 min <= duration_min <= 1440.0 min (24 hrs)
            df = df[(df["duration_min"] >= 1.0) & (df["duration_min"] <= 1440.0)]

            # Variables derivadas temporales
            df["hour"] = st.dt.hour
            df["day_of_week"] = st.dt.day_name()
            df["is_weekend"] = st.dt.dayofweek.isin([5, 6]).astype(int)
            df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

        self._registros_filtrados = filas_iniciales - len(df)
        self._ultimo_df = df
        self.log_info(f"Limpieza completada. {len(df)} registros válidos ({self._registros_filtrados} filtrados).")
        return df

    @property
    def stats(self) -> Dict[str, Any]:
        """Métricas operativas del último lote de viajes procesado.

        Returns:
            Dict[str, Any]: Resumen con total de viajes, duración mediana y tasa de suscriptores.
        """
        if self._ultimo_df is None or self._ultimo_df.empty:
            return {"status": "sin_datos", "total_viajes": 0}

        df = self._ultimo_df
        total = len(df)
        mediana_dur = float(df["duration_min"].median()) if "duration_min" in df else 0.0
        prop_member = (
            float((df["member_casual"] == "member").mean())
            if "member_casual" in df
            else 0.0
        )
        pct_rush = (
            float((df["is_rush_hour"] == 1).mean() * 100.0)
            if "is_rush_hour" in df
            else 0.0
        )

        return {
            "total_viajes": total,
            "duracion_mediana_minutos": round(mediana_dur, 2),
            "proporcion_miembros": round(prop_member, 3),
            "porcentaje_hora_pico": round(pct_rush, 2),
            "registros_filtrados": self._registros_filtrados,
        }


class MobilityAnalyst(DataComponent, LoggerMixin):
    """Componente para reportería de estaciones y agregaciones por tipo de unidad."""

    def __init__(self) -> None:
        DataComponent.__init__(self, nombre="MobilityAnalyst")
        LoggerMixin.__init__(self)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula agregaciones por tipo de bicicleta y membresía de usuario.

        Args:
            data: DataFrame limpio de viajes.

        Returns:
            pd.DataFrame: Resumen agregado.
        """
        if data.empty or "rideable_type" not in data.columns:
            return pd.DataFrame()

        resumen = (
            data.groupby(["rideable_type", "member_casual"])
            .agg(
                total_viajes=("ride_id", "count"),
                duracion_promedio_min=("duration_min", "mean"),
                duracion_mediana_min=("duration_min", "median"),
            )
            .reset_index()
        )
        resumen["duracion_promedio_min"] = resumen["duracion_promedio_min"].round(2)
        resumen["duracion_mediana_min"] = resumen["duracion_mediana_min"].round(2)
        return resumen
