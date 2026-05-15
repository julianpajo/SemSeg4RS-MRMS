"""
Sensor configuration from YAML/JSON files.

New strategy
------------
Sensor information is NOT inferred from GDAL, because raster metadata is often
incomplete or inconsistent.

Each sensor has a manual configuration file, for example:

    configs/sensors/spot.yaml
    configs/sensors/planetscope.yaml
    configs/sensors/sentinel2.yaml

The YAML/JSON file is the source of truth for:

    - name
    - wavelengths
    - bit_depth
    - band_names
    - gsd_m
    - rgbnir_idx
    - label_mapping

Rasterio is used only to read physical raster properties:

    - n_bands
    - dtype
    - nodata
    - crs
    - transform, optionally used to validate the GSD

Public API
----------
    read_sensor(image_path, sensor_config_path)        -> SensorConfig
    read_sensor_cached(image_path, sensor_config_path) -> SensorConfig
    read_sensor_config_file(sensor_config_path)        -> dict

Example
-------
    cfg = read_sensor(
        image_path="scene.tif",
        sensor_config_path="configs/sensors/spot.yaml",
    )

Recommended YAML format
-----------------------
    name: SPOT
    band_names: [blue, green, red, nir]
    wavelengths: [0.490, 0.560, 0.660, 0.830]
    bit_depth: 12
    gsd_m: 1.5
    rgbnir_idx: [2, 1, 0, 3]   # R, G, B, NIR — 0-based
    label_mapping:
      invalid_pixel: 0
      sealed_soil: 1
      non_sealed_soil: 2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class SensorConfigError(ValueError):
    """Generic error raised when the sensor configuration is invalid."""
    pass


class MissingSensorConfigError(SensorConfigError):
    """Raised when the sensor YAML/JSON configuration file does not exist."""
    pass


class InvalidSensorConfigError(SensorConfigError):
    """Raised when the sensor YAML/JSON configuration file is malformed."""
    pass


# Backward compatibility with old code that caught MissingWavelengthError.
class MissingWavelengthError(InvalidSensorConfigError):
    """Backward compatibility: wavelengths must now be defined in YAML/JSON."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# SensorConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SensorConfig:
    """
    Complete sensor configuration associated with a specific raster.

    Attributes
    ----------
    source_path :
        Path to the source raster.
    config_path :
        Path to the YAML/JSON file used as sensor configuration.
    sensor_name :
        Sensor name, for example "SPOT", "PlanetScope", or "Sentinel-2".
    n_bands :
        Number of raster bands.
    band_names :
        Manual band names loaded from the YAML/JSON file.
    wavelengths :
        Central wavelengths in micrometers, one for each band.
    bit_depth :
        Manual bit depth of the sensor/product.
    gsd_m :
        Manual Ground Sampling Distance, in meters.
    nodata :
        Nodata value read from the raster, or a default value.
    rgbnir_idx :
        0-based band indices in the format [R, G, B, NIR].
    crs :
        Raster CRS, if available.
    dtype :
        dtype of the first raster band, for example "uint16".
    raster_gsd_m :
        GSD estimated from the raster transform. Used only as a diagnostic check.
    label_mapping :
        Mapping between raw label values and semantic datasets classes.
    """

    source_path: str
    config_path: str
    sensor_name: str
    n_bands: int
    band_names: List[str]
    wavelengths: List[float]
    bit_depth: int
    gsd_m: float
    nodata: int | float
    rgbnir_idx: Optional[List[int]]
    crs: Optional[str]
    dtype: str
    raster_gsd_m: Optional[float] = None
    label_mapping: dict | None = None

    @property
    def max_val(self) -> int:
        """Return the maximum representable pixel value for the configured bit depth."""
        return (2 ** self.bit_depth) - 1

    @property
    def valid_range(self) -> Tuple[int, int]:
        """
        Return the default valid reflectance/intensity range.

        The value 0 is typically reserved for nodata/background, therefore the
        valid range starts from 1.
        """
        return (1, self.max_val)

    @property
    def rgbnir_wavelengths(self) -> Optional[List[float]]:
        """
        Return wavelengths ordered as [R, G, B, NIR], if rgbnir_idx is available.

        Returns
        -------
        list of float or None
            Wavelengths corresponding to RGB+NIR order, or None if the sensor
            does not define rgbnir_idx.
        """
        if self.rgbnir_idx is None:
            return None
        return [self.wavelengths[i] for i in self.rgbnir_idx]

    @property
    def has_rgbnir(self) -> bool:
        """Return True if the sensor defines RGB+NIR band indices."""
        return self.rgbnir_idx is not None

    def summary(self) -> str:
        """
        Return a human-readable multi-line summary of the sensor configuration.

        Returns
        -------
        str
            Formatted summary containing raster metadata, configuration metadata,
            wavelengths, GSD, nodata value, RGB+NIR indices and label mapping.
        """
        src = Path(self.source_path).name
        cfg = Path(self.config_path).name
        wl = [round(w, 4) for w in self.wavelengths]

        lines = [
            f"File       : {src}",
            f"Config     : {cfg}",
            f"Sensor     : {self.sensor_name}",
            f"Bands      : {self.n_bands}  ({', '.join(self.band_names)})",
            f"Wavelengths: {wl} µm",
            f"Bit depth  : {self.bit_depth}  (max={self.max_val}, dtype={self.dtype})",
            f"GSD config : {self.gsd_m} m",
            f"GSD raster : {self.raster_gsd_m} m",
            f"Nodata     : {self.nodata}",
            f"RGBNIR idx : {self.rgbnir_idx}",
            f"RGBNIR wl  : {self.rgbnir_wavelengths}",
            f"Label map  : {self.label_mapping}",
            f"CRS        : {self.crs}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# YAML/JSON loading
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = {
    "name",
    "wavelengths",
    "bit_depth",
    "band_names",
    "gsd_m",
    "rgbnir_idx",
    "label_mapping",
}


def read_sensor_config_file(config_path: str | Path) -> Dict[str, Any]:
    """
    Read a sensor configuration from YAML or JSON.

    Parameters
    ----------
    config_path :
        Path to a .yaml, .yml, or .json file.

    Returns
    -------
    dict
        Shallow-validated dictionary. Full validation is performed by
        validate_sensor_config().
    """
    path = Path(config_path)

    if not path.exists():
        raise MissingSensorConfigError(f"Sensor configuration file not found: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "To read YAML files, install PyYAML: pip install pyyaml"
                ) from exc

            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

        else:
            raise InvalidSensorConfigError(
                f"Unsupported configuration format: {path.suffix}. "
                "Use .yaml, .yml, or .json."
            )

    except SensorConfigError:
        raise
    except Exception as exc:
        raise InvalidSensorConfigError(
            f"Error while reading configuration file {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise InvalidSensorConfigError(
            f"The configuration file must contain an object/dictionary: {path}"
        )

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Sensor configuration validation
# ─────────────────────────────────────────────────────────────────────────────

def _as_float_list(value: Any, field_name: str) -> List[float]:
    """
    Convert a generic value to a list of floats.

    Parameters
    ----------
    value :
        Input value expected to be a list.
    field_name :
        Name of the configuration field, used in error messages.

    Returns
    -------
    list of float
        Numeric list rounded to 4 decimal places.
    """
    if not isinstance(value, list):
        raise InvalidSensorConfigError(f"'{field_name}' must be a list.")

    try:
        result = [round(float(x), 4) for x in value]
    except (TypeError, ValueError) as exc:
        raise InvalidSensorConfigError(
            f"'{field_name}' must contain numeric values only."
        ) from exc

    return result


def _as_str_list(value: Any, field_name: str) -> List[str]:
    """
    Convert a generic value to a list of non-empty strings.

    Parameters
    ----------
    value :
        Input value expected to be a list.
    field_name :
        Name of the configuration field, used in error messages.

    Returns
    -------
    list of str
        Cleaned list of stripped strings.
    """
    if not isinstance(value, list):
        raise InvalidSensorConfigError(f"'{field_name}' must be a list.")

    result = []
    for x in value:
        if not isinstance(x, str) or not x.strip():
            raise InvalidSensorConfigError(
                f"'{field_name}' must contain non-empty strings only."
            )
        result.append(x.strip())

    return result


def _as_optional_int_list(value: Any, field_name: str) -> Optional[List[int]]:
    """
    Convert a generic value to an optional list of integers.

    Parameters
    ----------
    value :
        Input value expected to be either None or a list.
    field_name :
        Name of the configuration field, used in error messages.

    Returns
    -------
    list of int or None
        Integer list if provided, otherwise None.
    """
    if value is None:
        return None

    if not isinstance(value, list):
        raise InvalidSensorConfigError(f"'{field_name}' must be a list or null.")

    try:
        result = [int(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise InvalidSensorConfigError(
            f"'{field_name}' must contain integers only."
        ) from exc

    return result


def validate_sensor_config(raw: Dict[str, Any], *, n_bands: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate and normalize the content of a sensor YAML/JSON configuration.

    If n_bands is provided, the function also checks that wavelengths and
    band_names have the same length as the raster band count.

    Parameters
    ----------
    raw :
        Raw dictionary loaded from YAML/JSON.
    n_bands :
        Optional number of bands read from the raster.

    Returns
    -------
    dict
        Normalized and validated sensor configuration.
    """
    missing = sorted(_REQUIRED_FIELDS - set(raw.keys()))
    if missing:
        raise InvalidSensorConfigError(
            f"Missing fields in sensor configuration: {missing}"
        )

    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise InvalidSensorConfigError("'name' must be a non-empty string.")
    name = name.strip()

    wavelengths = _as_float_list(raw["wavelengths"], "wavelengths")
    band_names = _as_str_list(raw["band_names"], "band_names")
    rgbnir_idx = _as_optional_int_list(raw["rgbnir_idx"], "rgbnir_idx")
    label_mapping = _validate_label_mapping(raw["label_mapping"])

    try:
        bit_depth = int(raw["bit_depth"])
    except (TypeError, ValueError) as exc:
        raise InvalidSensorConfigError("'bit_depth' must be an integer.") from exc

    try:
        gsd_m = float(raw["gsd_m"])
    except (TypeError, ValueError) as exc:
        raise InvalidSensorConfigError("'gsd_m' must be numeric.") from exc

    if bit_depth <= 0:
        raise InvalidSensorConfigError("'bit_depth' must be > 0.")

    if gsd_m <= 0:
        raise InvalidSensorConfigError("'gsd_m' must be > 0.")

    if len(wavelengths) != len(band_names):
        raise InvalidSensorConfigError(
            "'wavelengths' and 'band_names' must have the same length: "
            f"{len(wavelengths)} != {len(band_names)}"
        )

    if n_bands is not None and len(wavelengths) != n_bands:
        raise InvalidSensorConfigError(
            f"Configuration '{name}' has {len(wavelengths)} bands, "
            f"but the raster has {n_bands} bands."
        )

    if not wavelengths:
        raise MissingWavelengthError(
            "The 'wavelengths' list cannot be empty. "
            "Provide one central wavelength in µm for each band."
        )

    for i, wl in enumerate(wavelengths):
        if not 0.3 <= wl <= 20.0:
            raise InvalidSensorConfigError(
                f"Wavelength outside a plausible range at band {i}: {wl} µm. "
                "Use micrometers, not nanometers. Example: 0.665, not 665."
            )

    if rgbnir_idx is not None:
        if len(rgbnir_idx) != 4:
            raise InvalidSensorConfigError(
                "'rgbnir_idx' must contain exactly 4 values in the format [R, G, B, NIR]."
            )

        if len(set(rgbnir_idx)) != 4:
            raise InvalidSensorConfigError("'rgbnir_idx' must contain 4 distinct indices.")

        n = len(wavelengths)
        bad = [i for i in rgbnir_idx if i < 0 or i >= n]
        if bad:
            raise InvalidSensorConfigError(
                f"'rgbnir_idx' contains out-of-range indices: {bad}. "
                f"Valid range: 0..{n - 1}."
            )

    return {
        "name": name,
        "wavelengths": wavelengths,
        "bit_depth": bit_depth,
        "band_names": band_names,
        "gsd_m": gsd_m,
        "rgbnir_idx": rgbnir_idx,
        "label_mapping": label_mapping,
    }


def _validate_label_mapping(value: Any) -> Dict[str, int]:
    """
    Validate the raw label mapping of the sensor/datasets.

    Required format
    ---------------
        label_mapping:
          invalid_pixel: 0
          sealed_soil: 1
          non_sealed_soil: 2

    Notes
    -----
    The values are the RAW classes stored in the label GeoTIFF files.

    Training will always use:

        255 = ignore_index
        0   = sealed_soil
        1   = non_sealed_soil

    Parameters
    ----------
    value :
        Raw label mapping loaded from YAML/JSON.

    Returns
    -------
    dict
        Validated mapping with integer values.
    """
    if not isinstance(value, dict):
        raise InvalidSensorConfigError("'label_mapping' must be a dictionary.")

    required = {"invalid_pixel", "sealed_soil", "non_sealed_soil"}
    missing = required - set(value.keys())

    if missing:
        raise InvalidSensorConfigError(
            f"Incomplete 'label_mapping'. Missing: {sorted(missing)}"
        )

    result: Dict[str, int] = {}

    for key in required:
        try:
            result[key] = int(value[key])
        except (TypeError, ValueError) as exc:
            raise InvalidSensorConfigError(
                f"'label_mapping.{key}' must be an integer."
            ) from exc

    if len(set(result.values())) != len(result.values()):
        raise InvalidSensorConfigError(
            f"'label_mapping' values must be distinct. Received: {result}"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Raster helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gsd_from_transform(src) -> Optional[float]:
    """
    Estimate the GSD from the raster transform.

    Notes
    -----
    This value does NOT replace the manual gsd_m value defined in YAML/JSON.
    It is used only as a diagnostic check.

    Parameters
    ----------
    src :
        Open rasterio datasets.

    Returns
    -------
    float or None
        Estimated GSD in meters, if it can be computed.
    """
    try:
        t = src.transform
        gsd = (abs(t.a) + abs(t.e)) / 2.0

        if src.crs and src.crs.is_geographic:
            gsd *= 111_320.0

        if gsd <= 0:
            return None

        return round(float(gsd), 3)

    except Exception:
        return None


def _default_nodata(nodata: Any) -> int | float:
    """
    Normalize the raster nodata value.

    Parameters
    ----------
    nodata :
        Raw nodata value read from rasterio.

    Returns
    -------
    int or float
        Normalized nodata value. Returns 0 if nodata is missing or invalid.
    """
    if nodata is None:
        return 0

    try:
        as_float = float(nodata)
    except (TypeError, ValueError):
        return 0

    if as_float.is_integer():
        return int(as_float)
    return as_float


def _validate_raster_gsd(
    *,
    config_gsd_m: float,
    raster_gsd_m: Optional[float],
    tolerance_ratio: float,
    image_path: str,
) -> None:
    """
    Check that the configured GSD is compatible with the raster-estimated GSD.

    The check is optional and intentionally tolerant: many rasters are
    reprojected, resampled, or stored with transforms in geographic CRS.

    Parameters
    ----------
    config_gsd_m :
        Manual GSD value from the sensor configuration.
    raster_gsd_m :
        GSD estimated from the raster transform.
    tolerance_ratio :
        Maximum allowed relative difference.
    image_path :
        Raster path, used only for error messages.
    """
    if raster_gsd_m is None:
        return

    if tolerance_ratio <= 0:
        return

    diff_ratio = abs(raster_gsd_m - config_gsd_m) / config_gsd_m

    if diff_ratio > tolerance_ratio:
        raise InvalidSensorConfigError(
            "Raster GSD is very different from the GSD declared in the config.\n"
            f"  Raster : {image_path}\n"
            f"  Config : {config_gsd_m} m\n"
            f"  Raster : {raster_gsd_m} m\n"
            f"  Diff   : {diff_ratio:.2%}\n"
            "If the raster was resampled, increase gsd_tolerance_ratio "
            "or use validate_gsd=False."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def read_sensor(
    image_path: str | Path,
    sensor_config_path: str | Path,
    *,
    validate_gsd: bool = False,
    gsd_tolerance_ratio: float = 0.25,
) -> SensorConfig:
    """
    Read a raster and associate it with a manual YAML/JSON sensor configuration.

    Parameters
    ----------
    image_path :
        Path to the raster, for example .tif/.tiff.
    sensor_config_path :
        Path to the sensor .yaml/.yml/.json file.
    validate_gsd :
        If True, compare gsd_m from the config with the GSD estimated from the raster.
        Default is False because many rasters are resampled or reprojected.
    gsd_tolerance_ratio :
        Relative tolerance for validate_gsd. Default 0.25 = 25%.

    Returns
    -------
    SensorConfig
        Normalized and validated sensor configuration.
    """
    import rasterio

    image_path = Path(image_path)
    sensor_config_path = Path(sensor_config_path)

    raw_config = read_sensor_config_file(sensor_config_path)

    with rasterio.open(image_path) as src:
        n_bands = int(src.count)
        config = validate_sensor_config(raw_config, n_bands=n_bands)

        raster_gsd_m = _gsd_from_transform(src)

        if validate_gsd:
            _validate_raster_gsd(
                config_gsd_m=config["gsd_m"],
                raster_gsd_m=raster_gsd_m,
                tolerance_ratio=gsd_tolerance_ratio,
                image_path=str(image_path),
            )

        crs = src.crs.to_string() if src.crs else None
        dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
        nodata = _default_nodata(src.nodata)

    return SensorConfig(
        source_path=str(image_path),
        config_path=str(sensor_config_path),
        sensor_name=config["name"],
        n_bands=n_bands,
        band_names=config["band_names"],
        wavelengths=config["wavelengths"],
        bit_depth=config["bit_depth"],
        gsd_m=config["gsd_m"],
        nodata=nodata,
        rgbnir_idx=config["rgbnir_idx"],
        crs=crs,
        dtype=dtype,
        raster_gsd_m=raster_gsd_m,
        label_mapping=config["label_mapping"],
    )


@lru_cache(maxsize=512)
def read_sensor_cached(
    image_path: str,
    sensor_config_path: str,
    validate_gsd: bool = False,
    gsd_tolerance_ratio: float = 0.25,
) -> SensorConfig:
    """
    Cached version of read_sensor().

    Notes
    -----
    This function uses strings as arguments to keep the cache hashable and stable.

    Parameters
    ----------
    image_path :
        Path to the raster.
    sensor_config_path :
        Path to the sensor configuration file.
    validate_gsd :
        If True, compare configured GSD and raster-estimated GSD.
    gsd_tolerance_ratio :
        Relative tolerance used when validate_gsd=True.

    Returns
    -------
    SensorConfig
        Cached normalized sensor configuration.
    """
    return read_sensor(
        image_path=image_path,
        sensor_config_path=sensor_config_path,
        validate_gsd=validate_gsd,
        gsd_tolerance_ratio=gsd_tolerance_ratio,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible alias
# ─────────────────────────────────────────────────────────────────────────────

def get_or_infer_sensor(
    sensor_name: str,
    image_path: str,
    sensor_config_path: Optional[str] = None,
    wavelengths_override: Optional[List[float]] = None,
) -> SensorConfig:
    """
    Backward compatibility with the old API.

    Old behavior
    ------------
        get_or_infer_sensor(sensor_name, image_path, wavelengths_override=None)

    New behavior
    ------------
        sensor_config_path is required.

    Notes
    -----
    sensor_name and wavelengths_override are ignored because the complete
    configuration must come from the YAML/JSON file.

    Parameters
    ----------
    sensor_name :
        Kept only for backward compatibility. Ignored.
    image_path :
        Path to the raster.
    sensor_config_path :
        Path to the sensor YAML/JSON configuration file.
    wavelengths_override :
        Kept only for backward compatibility. Ignored.

    Returns
    -------
    SensorConfig
        Normalized and validated sensor configuration.
    """
    if sensor_config_path is None:
        raise InvalidSensorConfigError(
            "The new strategy requires sensor_config_path. "
            "Example: get_or_infer_sensor('SPOT', 'scene.tif', "
            "sensor_config_path='configs/sensors/spot.yaml')"
        )

    return read_sensor(
        image_path=image_path,
        sensor_config_path=sensor_config_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_USAGE = """
Usage:
    python sensor_configs.py <raster.tif> <sensor.yaml|sensor.json>

Example:
    python sensor_configs.py scene.tif configs/sensors/spot.yaml
""".strip()


def _main() -> int:
    """
    Command-line entry point.

    Returns
    -------
    int
        Exit code:
        0 for success,
        2 for sensor configuration errors,
        1 for unexpected errors.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Read a YAML/JSON sensor configuration and associate it with a raster.",
    )
    parser.add_argument("image_path", help="Path to the raster, e.g. scene.tif")
    parser.add_argument("sensor_config_path", help="Path to the sensor config .yaml/.yml/.json")
    parser.add_argument(
        "--validate-gsd",
        action="store_true",
        help="Compare the configured GSD with the GSD estimated from the raster.",
    )
    parser.add_argument(
        "--gsd-tolerance-ratio",
        type=float,
        default=0.25,
        help="Relative GSD tolerance. Default: 0.25 = 25%%.",
    )

    args = parser.parse_args()

    try:
        cfg = read_sensor(
            image_path=args.image_path,
            sensor_config_path=args.sensor_config_path,
            validate_gsd=args.validate_gsd,
            gsd_tolerance_ratio=args.gsd_tolerance_ratio,
        )
        print(cfg.summary())
        return 0

    except SensorConfigError as exc:
        print(f"[SENSOR CONFIG ERROR] {exc}")
        return 2

    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())