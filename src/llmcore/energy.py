"""Energy and emissions per call (LLM_SPEC.md §6.5).

- ``ecologits``: estimate for API models from output tokens and latency. If the
  exact model is not registered, set ``energy.ecologits_model`` to the closest
  model of the same provider; ``check`` lists close names. For batch calls the
  latency is unknown, so EcoLogits uses its own generation-time estimate.
- ``codecarbon``: measurement for local models (whole machine, so one call at a time).
- ``none``: not measured; the columns are left empty.

Values are flagged in the energy record with ``method`` so estimates and
measurements are never mixed silently.
"""

from __future__ import annotations

import difflib
import logging
import math
import threading
from typing import Callable

from .config import ConfigError, ModelConfig
from .types import Response

log = logging.getLogger(__name__)


def _mid(value) -> tuple[float | None, float | None, float | None]:
    """EcoLogits returns a value or a RangeValue(min, max); keep the midpoint and the range."""
    if value is None:
        return None, None, None
    v = getattr(value, "value", value)
    if hasattr(v, "min") and hasattr(v, "max"):
        return (v.min + v.max) / 2, v.min, v.max
    return float(v), float(v), float(v)


class EnergyMeter:
    method = "none"

    def measure(self, call: Callable[[], Response]) -> Response:
        """Run ``call`` and attach ``response.energy``."""
        response = call()
        response.energy = self.estimate(response)
        return response

    def estimate(self, response: Response) -> dict | None:
        """Energy for a response obtained without running ``call`` here (batch results)."""
        return None

    def close(self) -> None:
        pass


class EcoLogitsMeter(EnergyMeter):
    method = "ecologits"

    def __init__(self, cfg: ModelConfig, provider_name: str | None):
        from ecologits.model_repository import models

        self.provider = cfg.energy.ecologits_provider or provider_name
        self.model = cfg.energy.ecologits_model or cfg.model
        self.zone = cfg.energy.electricity_mix_zone
        if not self.provider:
            raise ConfigError(f"model {cfg.id!r}: set energy.ecologits_provider")
        if models.find_model(provider=self.provider, model_name=self.model) is None:
            names = [m.name for m in models.list_models() if str(getattr(m.provider, "value", m.provider)) == self.provider]
            close = difflib.get_close_matches(self.model, names, n=5, cutoff=0.3)
            raise ConfigError(
                f"model {cfg.id!r}: EcoLogits has no model {self.model!r} for provider {self.provider!r}; "
                f"set energy.ecologits_model to the closest one, e.g. {close or sorted(names)[:5]}"
            )

    def estimate(self, response: Response) -> dict | None:
        from ecologits.tracers.utils import llm_impacts

        if not response.tokens_out:
            return None
        latency = response.latency_s if response.latency_s is not None else math.inf
        impacts = llm_impacts(
            provider=self.provider,
            model_name=self.model,
            output_token_count=response.tokens_out,
            request_latency=latency,
            electricity_mix_zone=self.zone,
        )
        if impacts.errors:
            return {"method": self.method, "error": str(impacts.errors[0])}
        energy, energy_min, energy_max = _mid(impacts.energy)
        gwp, gwp_min, gwp_max = _mid(impacts.gwp)
        water, _, _ = _mid(getattr(impacts, "wcf", None))
        return {
            "method": self.method,
            "estimated": True,
            "ecologits_model": f"{self.provider}/{self.model}",
            "energy_kwh": energy,
            "energy_kwh_min": energy_min,
            "energy_kwh_max": energy_max,
            "co2e_kg": gwp,
            "co2e_kg_min": gwp_min,
            "co2e_kg_max": gwp_max,
            "water_l": water,
            "warnings": [w.code for w in impacts.warnings or []],
        }


class CodeCarbonMeter(EnergyMeter):
    """One CodeCarbon tracker per model, one task per call (``start_task``/``stop_task``).

    CodeCarbon measures the whole machine, so calls must not overlap
    (``max_concurrency: 1``, enforced in the config). On macOS without sudo,
    CodeCarbon cannot read powermetrics and falls back to a CPU power estimate;
    the manifest records which method it used.
    """

    method = "codecarbon"

    def __init__(self, cfg: ModelConfig):
        from codecarbon import OfflineEmissionsTracker

        self._tracker_cls = OfflineEmissionsTracker
        self.country = cfg.energy.country_iso_code
        self.project = f"strategy-{cfg.id}"
        self._tracker = None
        self._lock = threading.Lock()
        self._n = 0
        self._warned = False

    def _ensure(self):
        if self._tracker is None:
            self._tracker = self._tracker_cls(
                country_iso_code=self.country,
                project_name=self.project,
                save_to_file=False,
                log_level="error",
                measure_power_secs=1,
            )
            self._tracker.start()
        return self._tracker

    def measure(self, call):
        with self._lock:
            tracker = self._ensure()
            self._n += 1
            task = f"call-{self._n}"
            tracker.start_task(task)
            try:
                response = call()
            finally:
                data = tracker.stop_task(task)
        gpu_count = getattr(data, "gpu_count", None) or 0
        if gpu_count == 0 and not self._warned:
            self._warned = True
            log.warning(
                "CodeCarbon found no GPU it can measure: GPU energy is not included (on a Mac, Ollama runs on the "
                "Apple GPU, so values are a CPU/RAM estimate and undercount). Values are flagged estimated=true."
            )
        response.energy = {
            "method": self.method,
            # Without GPU tracking the energy of GPU inference is not measured, only estimated from CPU/RAM.
            "estimated": gpu_count == 0,
            "gpu_measured": gpu_count > 0,
            "energy_kwh": getattr(data, "energy_consumed", None),
            "co2e_kg": getattr(data, "emissions", None),
            "cpu_energy_kwh": getattr(data, "cpu_energy", None),
            "gpu_energy_kwh": getattr(data, "gpu_energy", None),
            "ram_energy_kwh": getattr(data, "ram_energy", None),
            "cpu_power_w": getattr(data, "cpu_power", None),
            "gpu_power_w": getattr(data, "gpu_power", None),
            "cpu_model": getattr(data, "cpu_model", None),
            "gpu_model": getattr(data, "gpu_model", None),
            "gpu_count": gpu_count,
            "country_iso_code": getattr(data, "country_iso_code", None),
            "codecarbon_version": getattr(data, "codecarbon_version", None),
        }
        return response

    def close(self) -> None:
        with self._lock:
            if self._tracker is not None:
                self._tracker.stop()
                self._tracker = None


def make_meter(cfg: ModelConfig, provider_ecologits_name: str | None) -> EnergyMeter:
    if cfg.energy.method == "ecologits":
        return EcoLogitsMeter(cfg, provider_ecologits_name)
    if cfg.energy.method == "codecarbon":
        return CodeCarbonMeter(cfg)
    return EnergyMeter()
