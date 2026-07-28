"""Spectral preprocessing and extinction-coefficient unmixing.

The model is the differential Beer-Lambert relation

    delta A(lambda) = L * sum(epsilon_i(lambda) * delta c_i) + nuisance

where epsilon must be a *reduced-minus-oxidised difference spectrum* sampled
through the actual LED/detector response. Peak coefficients copied from a paper
are not a substitute for instrument-specific spectral calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


CHANNELS_NM = np.array([450.0, 500.0, 550.0, 570.0, 600.0, 650.0])
AS7341_CHANNELS_NM = np.array([415.0, 445.0, 480.0, 515.0, 555.0, 590.0, 630.0, 680.0])
CYTOCHROME_LED_DETECTORS = {
    535: (555.0,),
    550: (555.0,),
    565: (555.0, 590.0),
    575: (555.0, 590.0),
    605: (630.0,),
    630: (630.0,),
}


def attenuation(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return differential optical attenuation, -log10(I / I0).

    Non-positive or non-finite values are returned as NaN instead of silently
    producing infinities.
    """
    signal = np.asarray(signal, dtype=float)
    reference = np.asarray(reference, dtype=float)
    signal, reference = np.broadcast_arrays(signal, reference)
    valid = np.isfinite(signal) & np.isfinite(reference) & (signal > 0) & (reference > 0)
    result = np.full(signal.shape, np.nan, dtype=float)
    result[valid] = -np.log10(signal[valid] / reference[valid])
    return result


def dark_correct(signal: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Subtract the matched dark spectrum."""
    return np.asarray(signal, dtype=float) - np.asarray(dark, dtype=float)


def haemoglobin_saturation(
    oxyhaemoglobin_um: float, deoxyhaemoglobin_um: float
) -> float:
    """Return tissue haemoglobin oxygen saturation in percent.

    Both inputs must be absolute non-negative concentrations. Differential
    concentration changes relative to an arbitrary baseline cannot determine an
    absolute saturation without baseline concentrations or endpoint calibration.
    """
    oxy = float(oxyhaemoglobin_um)
    deoxy = float(deoxyhaemoglobin_um)
    if not np.isfinite(oxy) or not np.isfinite(deoxy) or oxy < 0 or deoxy < 0:
        return float("nan")
    total = oxy + deoxy
    return 100.0 * oxy / total if total > 0 else float("nan")


def fibrosis_features(reflectance: pd.DataFrame) -> pd.DataFrame:
    """Return exploratory red-region reflectance features, not a fibrosis score.

    A fibrosis score must be trained and externally validated against histology
    for the same fresh/fixed tissue type, geometry and instrument.
    """
    numeric = np.asarray(reflectance.columns, dtype=float)
    if not {630.0, 680.0}.issubset(set(numeric)):
        raise ValueError("fibrosis features require 630 and 680 nm detector channels")
    values = reflectance.astype(float)
    safe = values.where(values > 0)
    result = pd.DataFrame(index=values.index)
    result["reflectance_630"] = safe[630.0]
    result["reflectance_680"] = safe[680.0]
    result["log_ratio_680_630"] = np.log(safe[680.0] / safe[630.0])
    result["red_slope_per_nm"] = (safe[680.0] - safe[630.0]) / 50.0
    return result


def cytochrome_led_indices(
    data: pd.DataFrame,
    baseline_cycles: int = 10,
) -> pd.DataFrame:
    """Calculate baseline-normalized narrow-LED differential attenuation.

    These are device indices, not concentrations. Detector selection reflects
    the nearest AS7341 bands and must be replaced by instrument-convolved
    coefficients for quantitative chromophore unmixing.
    """
    responses: dict[int, pd.Series] = {}
    for illumination, detector_channels in CYTOCHROME_LED_DETECTORS.items():
        spectra = paired_cycle_spectra(
            data,
            illumination_nm=illumination,
            channels_nm=detector_channels,
        )
        # Both members of each LED pair use the same detector band or
        # band-composite so filter response cannot create a false pair
        # difference.
        responses[illumination] = spectra.mean(axis=1)

    common = responses[535].index
    for series in responses.values():
        common = common.intersection(series.index)
    output = pd.DataFrame(index=common)
    for illumination, series in responses.items():
        values = series.loc[common].to_numpy(float)
        reference = float(np.nanmedian(values[: min(baseline_cycles, len(values))]))
        output[f"led_delta_a_{illumination}"] = attenuation(values, reference)

    output["cyt_c_led_index"] = output["led_delta_a_550"] - output["led_delta_a_535"]
    output["cyt_b_led_index"] = output["led_delta_a_565"] - output["led_delta_a_575"]
    output["cyt_aa3_led_index"] = output["led_delta_a_605"] - output["led_delta_a_630"]
    output.index.name = "cycle"
    return output.reset_index()


def plant_payload_features(
    data: pd.DataFrame,
    baseline_cycles: int = 5,
) -> pd.DataFrame:
    """Return device-specific germination and chlorophyll observables.

    The features are longitudinal changes relative to the dry, pre-rehydration
    baseline. They are not absolute chlorophyll concentration, germination
    percentage or photosynthetic quantum yield without biological calibration.
    """

    def response(illumination_nm: int, channel: int | str) -> pd.Series:
        signal_name = f"signal_{channel}"
        required = {"cycle", "illumination_nm", signal_name}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"plant features require columns: {sorted(missing)}")
        dark = data[data["illumination_nm"] == 0].set_index("cycle")[signal_name]
        light = data[data["illumination_nm"] == illumination_nm].set_index("cycle")[signal_name]
        common = dark.index.intersection(light.index)
        return light.loc[common].astype(float) - dark.loc[common].astype(float)

    measurements = {
        "fluorescence_680_counts": response(450, 680),
        "white_555": response(-1, 555),
        "white_680": response(-1, 680),
        "led_700_clear": response(700, "clear"),
        "led_730_clear": response(730, "clear"),
    }
    common = next(iter(measurements.values())).index
    for series in measurements.values():
        common = common.intersection(series.index)

    result = pd.DataFrame(index=common)
    for name, series in measurements.items():
        values = series.loc[common].to_numpy(float)
        reference = float(np.nanmedian(values[: min(baseline_cycles, len(values))]))
        result[name] = values
        result[f"delta_a_{name}"] = attenuation(values, reference)

    result["chlorophyll_red_absorption_delta"] = (
        result["delta_a_white_680"] - result["delta_a_white_555"]
    )
    result["red_edge_700_730_delta"] = (
        result["delta_a_led_700_clear"] - result["delta_a_led_730_clear"]
    )
    result["chlorophyll_fluorescence_680_log_change"] = -result[
        "delta_a_fluorescence_680_counts"
    ]
    result.index.name = "cycle"
    return result.reset_index()


def robust_reference(spectra: np.ndarray, count: int = 10) -> np.ndarray:
    """Median of the first baseline spectra, resistant to single-cycle spikes."""
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2 or spectra.shape[0] == 0:
        raise ValueError("spectra must be a non-empty two-dimensional array")
    return np.nanmedian(spectra[: min(count, len(spectra))], axis=0)


@dataclass(frozen=True)
class FitResult:
    concentrations_um: dict[str, float]
    standard_errors_um: dict[str, float]
    fitted_attenuation: np.ndarray
    residuals: np.ndarray
    rmse: float
    r_squared: float
    condition_number: float
    valid: bool
    warning: str = ""


class RedoxAnalyser:
    """Fit chromophore concentration changes by constrained ridge least squares.

    Coefficients are expected in mM^-1 cm^-1 and pathlength in cm. The returned
    concentration changes are micromolar. A positive result means an increase
    in the reduced-minus-oxidised component relative to the chosen reference.
    """

    def __init__(
        self,
        coefficients: pd.DataFrame,
        pathlength_cm: float,
        ridge: float = 1e-6,
        include_offset: bool = True,
        include_slope: bool = True,
        nonnegative: bool = False,
        maximum_condition: float = 1e6,
    ) -> None:
        if pathlength_cm <= 0:
            raise ValueError("pathlength_cm must be positive")
        if "wavelength_nm" not in coefficients:
            raise ValueError("coefficient table requires a wavelength_nm column")
        chromophores = [c for c in coefficients.columns if c != "wavelength_nm"]
        if not chromophores:
            raise ValueError("coefficient table has no chromophore columns")
        if coefficients[chromophores].isna().any().any():
            raise ValueError("coefficient table contains missing values")
        self.wavelengths = coefficients["wavelength_nm"].to_numpy(float)
        self.chromophores = chromophores
        self.extinction = coefficients[chromophores].to_numpy(float)
        self.pathlength_cm = float(pathlength_cm)
        self.ridge = float(ridge)
        self.include_offset = include_offset
        self.include_slope = include_slope
        self.nonnegative = nonnegative
        self.maximum_condition = maximum_condition

    @classmethod
    def from_csv(cls, filename: str | Path, **kwargs) -> "RedoxAnalyser":
        return cls(pd.read_csv(filename), **kwargs)

    def design_matrix(self) -> tuple[np.ndarray, int]:
        # epsilon [mM^-1 cm^-1] * L [cm] * c [uM] / 1000 = absorbance
        matrix = self.extinction * self.pathlength_cm / 1000.0
        chromophore_columns = matrix.shape[1]
        nuisance: list[np.ndarray] = []
        if self.include_offset:
            nuisance.append(np.ones(len(self.wavelengths)))
        if self.include_slope:
            scale = np.ptp(self.wavelengths)
            nuisance.append((self.wavelengths - np.mean(self.wavelengths)) / (scale or 1.0))
        if nuisance:
            matrix = np.column_stack([matrix, *nuisance])
        return matrix, chromophore_columns

    def fit(
        self,
        delta_attenuation: Iterable[float],
        standard_deviation: Iterable[float] | None = None,
    ) -> FitResult:
        y = np.asarray(list(delta_attenuation), dtype=float)
        if y.shape != self.wavelengths.shape:
            raise ValueError(f"expected {len(self.wavelengths)} attenuation values")
        valid = np.isfinite(y)
        if valid.sum() < len(self.chromophores) + int(self.include_offset) + int(self.include_slope):
            return self._invalid(y, "insufficient valid wavelengths for the requested model")

        x, chromophore_count = self.design_matrix()
        x, y_valid = x[valid], y[valid]
        if standard_deviation is not None:
            sd = np.asarray(list(standard_deviation), dtype=float)[valid]
            if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
                raise ValueError("standard deviations must be finite and positive")
            weights = 1.0 / sd
            x, y_valid = x * weights[:, None], y_valid * weights

        condition = float(np.linalg.cond(x[:, :chromophore_count]))
        # Ridge only regularises chromophores; offset and scattering slope remain free.
        if self.ridge > 0:
            penalty = np.zeros((chromophore_count, x.shape[1]))
            penalty[:, :chromophore_count] = np.eye(chromophore_count) * np.sqrt(self.ridge)
            x_fit = np.vstack([x, penalty])
            y_fit = np.concatenate([y_valid, np.zeros(chromophore_count)])
        else:
            x_fit, y_fit = x, y_valid

        lower = np.full(x.shape[1], -np.inf)
        if self.nonnegative:
            lower[:chromophore_count] = 0.0
        solution = lsq_linear(x_fit, y_fit, bounds=(lower, np.inf), method="trf")
        beta = solution.x

        full_x, _ = self.design_matrix()
        fitted = full_x @ beta
        residuals = y - fitted
        finite_residuals = residuals[valid]
        rmse = float(np.sqrt(np.mean(finite_residuals**2)))
        total = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
        r_squared = float(1.0 - np.sum(finite_residuals**2) / total) if total > 0 else np.nan

        degrees = max(valid.sum() - x.shape[1], 1)
        sigma2 = float(np.sum(finite_residuals**2) / degrees)
        covariance = sigma2 * np.linalg.pinv(x.T @ x)
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))[:chromophore_count]
        warning = ""
        is_valid = bool(solution.success and condition <= self.maximum_condition)
        if condition > self.maximum_condition:
            warning = (
                f"ill-conditioned coefficient matrix ({condition:.2g}); "
                "add independent wavelengths or instrument-specific calibration spectra"
            )
        elif not solution.success:
            warning = solution.message

        return FitResult(
            concentrations_um=dict(zip(self.chromophores, beta[:chromophore_count])),
            standard_errors_um=dict(zip(self.chromophores, errors)),
            fitted_attenuation=fitted,
            residuals=residuals,
            rmse=rmse,
            r_squared=r_squared,
            condition_number=condition,
            valid=is_valid,
            warning=warning,
        )

    def _invalid(self, values: np.ndarray, warning: str) -> FitResult:
        nan_map = {name: float("nan") for name in self.chromophores}
        return FitResult(
            concentrations_um=nan_map,
            standard_errors_um=nan_map.copy(),
            fitted_attenuation=np.full(values.shape, np.nan),
            residuals=np.full(values.shape, np.nan),
            rmse=float("nan"),
            r_squared=float("nan"),
            condition_number=float("inf"),
            valid=False,
            warning=warning,
        )


def paired_cycle_spectra(
    data: pd.DataFrame,
    illumination_nm: int = -1,
    channels_nm: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return dark-corrected spectra, pairing dark and illuminated rows by cycle."""
    if channels_nm is None:
        channels_nm = sorted(
            float(column.removeprefix("signal_"))
            for column in data.columns
            if column.startswith("signal_")
            and column.removeprefix("signal_").replace(".", "", 1).isdigit()
        )
    channels_nm = np.asarray(list(channels_nm), dtype=float)
    required = {"cycle", "illumination_nm", *(f"signal_{int(w)}" for w in channels_nm)}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing acquisition columns: {sorted(missing)}")
    channels = [f"signal_{int(w)}" for w in channels_nm]
    dark = data[data["illumination_nm"] == 0].set_index("cycle")[channels]
    light = data[data["illumination_nm"] == illumination_nm].set_index("cycle")[channels]
    common = dark.index.intersection(light.index)
    if common.empty:
        raise ValueError("no cycles contain both dark and illuminated measurements")
    corrected = light.loc[common] - dark.loc[common].to_numpy()
    corrected.columns = channels_nm
    return corrected
