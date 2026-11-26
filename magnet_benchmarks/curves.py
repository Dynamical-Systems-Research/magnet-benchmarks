"""Correlated intrinsic J(H, T) material-curve interface for aligned 0% HRE NdFeB.

Provides general-physics, lot-specific 2nd-quadrant intrinsic demagnetization
curves J(H_parallel, H_perp, T) across temperature, field obliquity, magnetic
history (descending, recoil, and thermal return), and parent-block-to-part
machining/coating transfer, together with the 2-record living interface
(MaterialCurveRecord and OperatingPointRecord) for automatic candidate rescoring.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

RHO_TD = 7.62
JGB_LEVELS = (0.0, 0.6, 1.2)
_H_REF_SIM_T = 2.627829
_X_SIM_T = 0.674357
C_GRID = np.array([0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0], dtype=float)
_R_HCI = {
    0.0: np.array([0.4, 0.402, 0.418, 0.491, 0.632, 0.798, 0.911, 1.0], dtype=float),
    0.6: np.array([0.4, 0.377, 0.368, 0.403, 0.496, 0.593, 0.663, 0.719], dtype=float),
    1.2: np.array([0.4, 0.348, 0.308, 0.306, 0.338, 0.369, 0.399, 0.42], dtype=float),
}
R_ALPHA = {
    J: (_R_HCI[J] * _H_REF_SIM_T + _X_SIM_T) / (_H_REF_SIM_T + _X_SIM_T)
    for J in JGB_LEVELS
}


def kuzmin_js(T_K: Any, Js0: Any, Tc: float = 585.0, s: Any = 0.5, p: float = 2.5) -> np.ndarray:
    tau = np.clip(np.asarray(T_K, dtype=float) / Tc, 0.0, 0.999)
    return Js0 * np.clip(1.0 - s * tau**1.5 - (1.0 - s) * tau**p, 1e-6, None) ** (1.0 / 3.0)

MU0 = 4.0e-7 * math.pi  # T * m / A
KOE_TO_A_M = 1.0e6 / (4.0 * math.pi)  # 79577.47154594767 A/m per kOe
A_M_TO_KOE = 4.0 * math.pi * 1.0e-6

VALID_BRANCHES = ("descending", "recoil", "return")
VALID_EVIDENCE_CLASSES = (
    "twin_posterior",
    "microstructure_physics",
    "lab_measured",
    "external_micromagnetic",
    "catalog_prior",
)

# Deterministic 2D quadrature nodes (normal quantiles & weights) for grain-ensemble
# integration over log-normal grain size (15 nodes) x local GB/texture spread (11 nodes).
_GH_D_X, _GH_D_W = np.polynomial.hermite.hermgauss(15)
_Z_D = np.sqrt(2.0) * _GH_D_X
_W_D = _GH_D_W / np.sqrt(np.pi)

_GH_A_X, _GH_A_W = np.polynomial.hermite.hermgauss(11)
_Z_A = np.sqrt(2.0) * _GH_A_X
_W_A = _GH_A_W / np.sqrt(np.pi)

_QUAD_ZD, _QUAD_ZA = np.meshgrid(_Z_D, _Z_A, indexing="ij")
_QUAD_ZD = _QUAD_ZD.ravel()
_QUAD_ZA = _QUAD_ZA.ravel()
_QUAD_W = (_W_D[:, None] * _W_A[None, :]).ravel()


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    xc = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-xc))


def angular_switching_factor(
    obliquity_rad: np.ndarray | float,
    pinning_fraction: float = 0.45,
) -> np.ndarray:
    """Mixed nucleation (Stoner-Wohlfarth) / pinning (Kondorsky) angular law.

    Returns the dimensionless multiplier ``f(psi) = H_sw(psi) / H_sw(0)`` for a
    field tilted by ``obliquity_rad`` from the grain easy axis.
    ``pinning_fraction = 0`` gives pure coherent nucleation (SW dip to 0.5 at 45 deg);
    ``pinning_fraction = 1`` gives domain-wall pinning (1/cos(psi), smoothly capped
    at high angles so transverse fields remain finite); ``0.40-0.55`` reproduces the
    measured shallow dip (~0.84-0.92 near 30-45 deg) and high-angle rise of aligned
    sintered (Nd,Pr)-Fe-B magnets.
    """
    psi = np.clip(np.asarray(obliquity_rad, dtype=float), 0.0, 0.5 * math.pi - 1e-4)
    c = np.cos(psi)
    s = np.sin(psi)
    # Stoner-Wohlfarth coherent rotation astroid
    f_sw = np.power(np.power(np.maximum(c, 1e-9), 2.0 / 3.0) + np.power(np.maximum(s, 0.0), 2.0 / 3.0), -1.5)
    # Kondorsky 1/cos(psi) with smooth transverse cap (~2.5x) reflecting finite oblique nucleation
    f_kond_raw = 1.0 / np.maximum(c, 0.15)
    f_kond = 1.0 + 1.5 * np.tanh((f_kond_raw - 1.0) / 1.5)
    kap = float(np.clip(pinning_fraction, 0.0, 1.0))
    # Harmonic/geometric blend stays smooth and strictly positive across [0, pi/2)
    return (1.0 - kap) * f_sw + kap * f_kond


@dataclass(frozen=True)
class MicrostructureState:
    """General physical state of an aligned 0% HRE (Nd,Pr)-Fe-B magnet lot."""

    name: str = "virgin_n42_baseline"
    pr_fraction: float = 0.22  # Pr / (Nd + Pr) molar fraction in 0% HRE didymium
    grain_median_um: float = 4.2
    grain_sigma_ln: float = 0.36
    alignment_cos: float = 0.955  # <cos(theta)> texture factor in [0.5, 0.99]
    density_rel: float = 0.9908  # rho / 7.62 g/cm^3 -> 7.55 g/cm^3
    gb_phase_fraction: float = 0.058  # non-magnetic RE-rich/GB volume fraction
    gb_continuity: float = 0.92  # grain-boundary wetting/decoupling continuity c in [0, 1]
    j_gb: float = 0.0  # intergranular exchange coupling level in [0.0, 1.2]
    alpha_k_base: float = 0.495  # Kronmueller microstructural nucleation factor at c=1, j_gb=0
    n_eff_b: float = 0.118  # Kronmueller stray-field log-grain slope (sintered aligned N_eff ~ 0.75-0.85)
    d_b_um: float = 0.004  # exchange length / defect scale (um)
    pinning_fraction: float = 0.45  # angular switching blend (0=SW nucleation, 1=Kondorsky pinning)
    alpha_defect_cv: float = 0.018  # baseline local grain nucleation CV
    kuzmin_s: float = 0.75
    kuzmin_p: float = 2.5
    gb_thermal_softening_per_K: float = 0.00055  # extra GB nucleation softening with T
    js_20c_override_T: float | None = None
    ha_20c_override_T: float | None = None
    tc_override_K: float | None = None
    ha_temp_exponent_override: float | None = None
    # Optional machining/coating transfer layers (populated for finished parts)
    damaged_skin_fraction: float = 0.0
    skin_hcj_factor: float = 0.50
    coating_volume_fraction: float = 0.0
    orientation_scatter_deg: float = 0.0
    piece_jr_factor: float = 1.0
    piece_hci_factor: float = 1.0

    def density_kg_m3(self) -> float:
        return float(self.density_rel * RHO_TD * 1000.0)

    def tc_K(self) -> float:
        if self.tc_override_K is not None:
            return float(self.tc_override_K)
        return 585.0 - 20.0 * float(np.clip(self.pr_fraction, 0.0, 0.5))

    def js_20c_T(self) -> float:
        if self.js_20c_override_T is not None:
            return float(self.js_20c_override_T)
        return 1.610 - 0.050 * float(np.clip(self.pr_fraction, 0.0, 0.5))

    def ha_20c_T(self) -> float:
        if self.ha_20c_override_T is not None:
            return float(self.ha_20c_override_T)
        return 7.30 + 1.40 * float(np.clip(self.pr_fraction, 0.0, 0.5))

    def ha_temp_exponent(self) -> float:
        if self.ha_temp_exponent_override is not None:
            return float(self.ha_temp_exponent_override)
        # Callen-Callen / experimental Nd2Fe14B H_A(T) scaling (~7.3 T @ 20 C -> ~5.0 T @ 100 C),
        # steepening with Pr content (higher H_A at 20 C but faster thermal decay).
        return 3.25 + 1.40 * float(np.clip(self.pr_fraction, 0.0, 0.5))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESET_MICROSTRUCTURES: dict[str, MicrostructureState] = {
    # Representative aligned 0% HRE virgin sintered (Nd,Pr)-Fe-B families spanning
    # commercial N42 through microstructure-engineered high-coercivity / high-remanence grades.
    "virgin_n42_baseline": MicrostructureState(
        name="virgin_n42_baseline",
        pr_fraction=0.22,
        grain_median_um=4.4,
        grain_sigma_ln=0.34,
        alignment_cos=0.940,
        density_rel=0.9908,
        gb_phase_fraction=0.118,
        gb_continuity=0.92,
        j_gb=0.05,
        alpha_k_base=0.490,
        alpha_defect_cv=0.018,
        pinning_fraction=0.45,
    ),
    "virgin_n42_high_pr": MicrostructureState(
        name="virgin_n42_high_pr",
        pr_fraction=0.34,
        grain_median_um=4.2,
        grain_sigma_ln=0.33,
        alignment_cos=0.942,
        density_rel=0.9910,
        gb_phase_fraction=0.118,
        gb_continuity=0.92,
        j_gb=0.05,
        alpha_k_base=0.490,
        alpha_defect_cv=0.018,
        pinning_fraction=0.45,
    ),
    "virgin_n45h_fine_grain": MicrostructureState(
        name="virgin_n45h_fine_grain",
        pr_fraction=0.20,
        grain_median_um=2.6,
        grain_sigma_ln=0.26,
        alignment_cos=0.955,
        density_rel=0.9920,
        gb_phase_fraction=0.102,
        gb_continuity=0.955,
        j_gb=0.02,
        alpha_k_base=0.465,
        alpha_defect_cv=0.014,
        pinning_fraction=0.50,
    ),
    "virgin_n48h_gb_engineered": MicrostructureState(
        name="virgin_n48h_gb_engineered",
        pr_fraction=0.18,
        grain_median_um=2.1,
        grain_sigma_ln=0.22,
        alignment_cos=0.966,
        density_rel=0.9935,
        gb_phase_fraction=0.088,
        gb_continuity=0.975,
        j_gb=0.0,
        alpha_k_base=0.458,
        alpha_defect_cv=0.012,
        pinning_fraction=0.52,
    ),
    "virgin_n50h_high_texture": MicrostructureState(
        name="virgin_n50h_high_texture",
        pr_fraction=0.15,
        grain_median_um=2.6,
        grain_sigma_ln=0.25,
        alignment_cos=0.974,
        density_rel=0.9945,
        gb_phase_fraction=0.078,
        gb_continuity=0.960,
        j_gb=0.02,
        alpha_k_base=0.450,
        alpha_defect_cv=0.015,
        pinning_fraction=0.48,
    ),
    "virgin_n52_high_br": MicrostructureState(
        name="virgin_n52_high_br",
        pr_fraction=0.12,
        grain_median_um=4.2,
        grain_sigma_ln=0.33,
        alignment_cos=0.980,
        density_rel=0.9950,
        gb_phase_fraction=0.068,
        gb_continuity=0.90,
        j_gb=0.10,
        alpha_k_base=0.495,
        alpha_defect_cv=0.020,
        pinning_fraction=0.42,
    ),
    "virgin_high_hcj_edge": MicrostructureState(
        name="virgin_high_hcj_edge",
        pr_fraction=0.20,
        grain_median_um=1.4,
        grain_sigma_ln=0.20,
        alignment_cos=0.935,
        density_rel=0.9910,
        gb_phase_fraction=0.120,
        gb_continuity=0.985,
        j_gb=0.0,
        alpha_k_base=0.475,
        alpha_defect_cv=0.011,
        pinning_fraction=0.55,
    ),
    "virgin_low_gb_continuity_risk": MicrostructureState(
        name="virgin_low_gb_continuity_risk",
        pr_fraction=0.22,
        grain_median_um=4.6,
        grain_sigma_ln=0.44,
        alignment_cos=0.932,
        density_rel=0.9880,
        gb_phase_fraction=0.118,
        gb_continuity=0.84,
        j_gb=0.25,
        alpha_k_base=0.550,
        alpha_defect_cv=0.036,
        pinning_fraction=0.38,
    ),
}


def _interp_ralpha(c_gb: float, j_gb: float) -> float:
    """Continuous interpolation of the micromagnetic grain-network r_alpha(c, j_gb)."""
    c = float(np.clip(c_gb, 0.0, 1.0))
    j = float(np.clip(j_gb, JGB_LEVELS[0], JGB_LEVELS[-1]))
    vals = [float(np.interp(c, C_GRID, R_ALPHA[lvl])) for lvl in JGB_LEVELS]
    return float(np.interp(j, JGB_LEVELS, vals))


def compute_curve_metrics(
    H_parallel_A_m: Sequence[float] | np.ndarray,
    J_parallel_T: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Derive J_r, H_cj, H_k90 (at J = 0.9 J_r), squareness, and BH_max directly from a 2nd-quadrant curve.

    Accepts ``H_parallel_A_m <= 0`` (descending from 0 toward negative demagnetizing fields).
    Returned ``hci_A_m``, ``hk90_A_m``, and ``hcb_A_m`` are positive magnitudes in A/m.
    """
    H = np.asarray(H_parallel_A_m, dtype=float)
    J = np.asarray(J_parallel_T, dtype=float)
    if H.ndim != 1 or J.ndim != 1 or len(H) != len(J) or len(H) < 2:
        raise ValueError("H_parallel_A_m and J_parallel_T must be 1D arrays of equal length >= 2")
    # Order by increasing opposing field magnitude |H| = -H
    H_mag = -H
    order = np.argsort(H_mag)
    H_mag = H_mag[order]
    J_ord = J[order]

    jr = float(np.interp(0.0, H_mag, J_ord)) if H_mag[0] <= 0.0 <= H_mag[-1] else float(J_ord[0])
    jr = max(jr, 0.0)

    def _crossing_mag(target_J: float) -> float:
        if jr <= target_J:
            return 0.0
        below = np.where(J_ord <= target_J)[0]
        if len(below) == 0:
            return math.nan
        idx = int(below[0])
        if idx == 0:
            return float(H_mag[0])
        j0, j1 = float(J_ord[idx - 1]), float(J_ord[idx])
        h0, h1 = float(H_mag[idx - 1]), float(H_mag[idx])
        if abs(j1 - j0) < 1e-15:
            return h1
        frac = (target_J - j0) / (j1 - j0)
        return float(h0 + np.clip(frac, 0.0, 1.0) * (h1 - h0))

    hci = _crossing_mag(0.0)
    hk90 = _crossing_mag(0.9 * jr)
    sq = hk90 / hci if (math.isfinite(hk90) and math.isfinite(hci) and hci > 1e-9) else math.nan

    B = J_ord - MU0 * H_mag
    # H_cb where B = 0
    b_below = np.where(B <= 0.0)[0]
    if len(b_below) == 0:
        hcb = math.nan
    else:
        i = int(b_below[0])
        if i == 0:
            hcb = float(H_mag[0])
        else:
            b0, b1 = float(B[i - 1]), float(B[i])
            h0, h1 = float(H_mag[i - 1]), float(H_mag[i])
            hcb = float(h0 + np.clip(-b0 / (b1 - b0), 0.0, 1.0) * (h1 - h0)) if abs(b1 - b0) > 1e-15 else h1

    bh_J_m3 = np.maximum(B, 0.0) * np.maximum(H_mag, 0.0)
    bhmax_kJ_m3 = float(np.max(bh_J_m3) / 1000.0)
    bhmax_MGOe = bhmax_kJ_m3 / 7.957747154594767

    # Recoil permeability near H = 0 (first 5% of H_k90 or first non-zero segment)
    if math.isfinite(hk90) and math.isfinite(hci) and hk90 > 0:
        h_probe = min(0.15 * hk90, 0.1 * hci)
    elif len(H_mag) > 1:
        h_probe = float(min(0.15 * H_mag[-1], H_mag[1])) if H_mag[1] > 0 else float(0.15 * H_mag[-1])
    else:
        h_probe = 0.0
    if h_probe > 1e-6:
        j_probe = float(np.interp(h_probe, H_mag, J_ord))
        dJ_dH = max((jr - j_probe) / h_probe, 0.0)
        mu_rec = 1.0 + dJ_dH / MU0
    else:
        mu_rec = 1.04

    return {
        "jr_T": jr,
        "hci_A_m": hci,
        "hci_kOe": hci * A_M_TO_KOE,
        "hk90_A_m": hk90,
        "hk90_kOe": hk90 * A_M_TO_KOE,
        "squareness_hk90_over_hci": sq,
        "hcb_A_m": hcb,
        "hcb_kOe": hcb * A_M_TO_KOE,
        "bhmax_kJ_m3": bhmax_kJ_m3,
        "bhmax_MGOe": bhmax_MGOe,
        "mu_rec_initial": mu_rec,
    }


@dataclass
class MaterialCurveRecord:
    """Record 1 (Twin -> Rotor): correlated intrinsic J(H, T) 2nd-quadrant curve."""

    curve_id: str
    parent_curve_id: str | None
    source_ref: str
    evidence_class: str
    lot_id: str
    specimen_id: str
    process_stage: str
    geometry: dict[str, Any]
    texture_axis: list[float]
    density_kg_m3: float
    T_C: float
    H_parallel_A_m: list[float]
    J_parallel_T: list[float]
    branch: str
    reversal_H_A_m: float | None
    uncertainty: dict[str, Any]
    joint_draw_id: str
    sharing_scope: str = "internal_anonymized"
    # Derived & auxiliary physics metadata (optional for external lab curves)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.branch not in VALID_BRANCHES:
            raise ValueError(f"branch must be one of {VALID_BRANCHES}, got {self.branch!r}")
        if len(self.H_parallel_A_m) != len(self.J_parallel_T) or len(self.H_parallel_A_m) < 2:
            raise ValueError("H_parallel_A_m and J_parallel_T must have equal length >= 2")
        if self.branch in ("recoil", "return") and self.reversal_H_A_m is None:
            raise ValueError(f"branch={self.branch!r} requires reversal_H_A_m")
        if "metrics" not in self.metadata and self.branch == "descending":
            self.metadata["metrics"] = compute_curve_metrics(self.H_parallel_A_m, self.J_parallel_T)

    @property
    def metrics(self) -> dict[str, float]:
        if "metrics" not in self.metadata:
            self.metadata["metrics"] = compute_curve_metrics(self.H_parallel_A_m, self.J_parallel_T)
        return self.metadata["metrics"]

    def evaluate_J(self, H_parallel_A_m: np.ndarray | float) -> np.ndarray:
        """Interpolate J_parallel_T(H_parallel_A_m) along this curve (2nd quadrant H <= 0)."""
        H_tab = -np.asarray(self.H_parallel_A_m, dtype=float)
        J_tab = np.asarray(self.J_parallel_T, dtype=float)
        order = np.argsort(H_tab)
        H_tab = H_tab[order]
        J_tab = J_tab[order]
        H_q = -np.asarray(H_parallel_A_m, dtype=float)
        if not np.all(np.isfinite(H_q)):
            raise ValueError("Query H_parallel_A_m must be finite")
        if np.any(H_q < H_tab[0]) or np.any(H_q > H_tab[-1]):
            h_lo = float(-H_tab[-1])
            h_hi = float(-H_tab[0])
            if h_hi == 0.0:
                h_hi = 0.0
            raise ValueError(
                f"Query H_parallel_A_m outside sampled curve range [{h_lo:.3f}, {h_hi:.3f}] A/m"
            )
        return np.interp(H_q, H_tab, J_tab)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialCurveRecord:
        known = {
            "curve_id",
            "parent_curve_id",
            "source_ref",
            "evidence_class",
            "lot_id",
            "specimen_id",
            "process_stage",
            "geometry",
            "texture_axis",
            "density_kg_m3",
            "T_C",
            "H_parallel_A_m",
            "J_parallel_T",
            "branch",
            "reversal_H_A_m",
            "uncertainty",
            "joint_draw_id",
            "sharing_scope",
            "metadata",
        }
        filtered = {k: data[k] for k in known if k in data}
        return cls(**filtered)


@dataclass
class OperatingPointBin:
    """One field-temperature sensitivity bin exported by an EM rotor solver."""

    T_C: float
    H_parallel_A_m: float
    H_perp_A_m: float
    dV_m3: float
    dScore_dJloss_per_T: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperatingPointBin:
        return cls(
            T_C=float(data["T_C"]),
            H_parallel_A_m=float(data["H_parallel_A_m"]),
            H_perp_A_m=float(data["H_perp_A_m"]),
            dV_m3=float(data["dV_m3"]),
            dScore_dJloss_per_T=float(data["dScore_dJloss_per_T"]),
        )


@dataclass
class OperatingPointRecord:
    """Record 2 (Rotor -> Twin & Lab): operating-point sensitivity bins for rescoring."""

    case_id: str
    geometry_id: str
    curve_id: str
    stock_case_id: str
    seat_condition: str
    finished_stack: dict[str, Any]
    winding_basis: dict[str, Any]
    thermal_state: dict[str, Any]
    mass_g: float
    score_kind: str
    score_ratio: float
    bins: list[OperatingPointBin]
    bench_observations: list[dict[str, Any]] = field(default_factory=list)
    sharing_scope: str = "internal_anonymized"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bins"] = [b.to_dict() if isinstance(b, OperatingPointBin) else dict(b) for b in self.bins]
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperatingPointRecord:
        bins = [
            b if isinstance(b, OperatingPointBin) else OperatingPointBin.from_dict(b)
            for b in data.get("bins", [])
        ]
        return cls(
            case_id=str(data["case_id"]),
            geometry_id=str(data["geometry_id"]),
            curve_id=str(data["curve_id"]),
            stock_case_id=str(data["stock_case_id"]),
            seat_condition=str(data["seat_condition"]),
            finished_stack=dict(data["finished_stack"]),
            winding_basis=dict(data["winding_basis"]),
            thermal_state=dict(data["thermal_state"]),
            mass_g=float(data["mass_g"]),
            score_kind=str(data["score_kind"]),
            score_ratio=float(data["score_ratio"]),
            bins=bins,
            bench_observations=list(data.get("bench_observations", [])),
            sharing_scope=str(data.get("sharing_scope", "internal_anonymized")),
        )


class AlignedCurveModel:
    """Callable physics model for a single correlated aligned 0% HRE NdFeB lot.

    Carries one microstructure/chemistry state consistently across temperature,
    field obliquity, magnetic reversal/return history, and parent-block-to-part
    transfer.
    """

    def __init__(
        self,
        state: MicrostructureState,
        *,
        lot_id: str | None = None,
        specimen_id: str = "bulk-witness",
        process_stage: str = "parent_block",
        source_ref: str = "curves",
        evidence_class: str = "microstructure_physics",
        joint_draw_id: str | None = None,
        parent_curve_id: str | None = None,
        geometry: Mapping[str, Any] | None = None,
        texture_axis: Sequence[float] = (0.0, 0.0, 1.0),
        uncertainty: Mapping[str, Any] | None = None,
        sharing_scope: str = "public_physics",
    ) -> None:
        self.state = state
        self.lot_id = lot_id or f"lot-{state.name}"
        self.specimen_id = specimen_id
        self.process_stage = process_stage
        self.source_ref = source_ref
        self.evidence_class = evidence_class
        self.joint_draw_id = joint_draw_id or _stable_id("draw", state.to_dict())
        self.parent_curve_id = parent_curve_id
        self.geometry = dict(geometry or {"kind": "parent_block"})
        self.texture_axis = [float(x) for x in texture_axis]
        self.uncertainty = dict(uncertainty or {})
        self.sharing_scope = sharing_scope

        # Precompute grain-ensemble static factors (bulk + optional machined surface skin)
        self._build_ensemble()

    def _build_ensemble(self) -> None:
        st = self.state
        cos_align = float(np.clip(st.alignment_cos, 0.50, 0.995))
        scatter_rad = math.radians(max(st.orientation_scatter_deg, 0.0))
        cos_eff = float(np.clip(cos_align * math.cos(scatter_rad), 0.50, 0.995))
        # Characteristic misorientation spread (radians) consistent with <cos(theta)>
        # For small angles <cos(theta)> ~ 1 - 0.5 * <theta^2> => theta_rms ~ sqrt(2*(1 - cos))
        sin2_mean = float(np.clip(1.0 - cos_eff * cos_eff, 0.005, 0.667))
        theta_rms = math.sqrt(max(2.0 * (1.0 - cos_eff), 1e-4))
        self._cos_eff = cos_eff
        self._sin2_mean = sin2_mean
        self._theta_rms = theta_rms

        # Grain size quadrature (um)
        D_k = np.maximum(st.grain_median_um * np.exp(st.grain_sigma_ln * _QUAD_ZD), 0.15)
        self._neff_k = st.n_eff_b * np.log(np.maximum(D_k / st.d_b_um, 1.5))

        # Grain-boundary continuity multiplier from micromagnetic grain-network table
        r_a_ref = _interp_ralpha(1.0, 0.0)
        r_a = _interp_ralpha(st.gb_continuity, st.j_gb) / max(r_a_ref, 1e-6)
        # Local GB defect spread widens when wetting continuity c_gb drops or j_gb rises
        defect_cv = st.alpha_defect_cv + 0.08 * (1.0 - st.gb_continuity) + 0.025 * st.j_gb
        alpha_mean = st.alpha_k_base * r_a * st.piece_hci_factor
        # Log-normal local nucleation factor across grains
        self._alpha_bulk_k = alpha_mean * np.exp(defect_cv * _QUAD_ZA - 0.5 * defect_cv * defect_cv)
        # Mild grain misorientation dispersion around texture axis
        self._theta_grain_k = np.clip(theta_rms * np.abs(_QUAD_ZA) * 0.65, 0.0, 0.45 * math.pi)
        self._w_base = _QUAD_W.copy()

        # Active magnetic volume fraction (coating displacement when dims are post-coat)
        self._active_vol_frac = float(np.clip(1.0 - st.coating_volume_fraction, 0.1, 1.0))
        self._f_skin = float(np.clip(st.damaged_skin_fraction, 0.0, 0.6))
        self._skin_factor = float(np.clip(st.skin_hcj_factor, 0.0, 1.0))

    def intrinsic_params_at_T(self, T_C: float) -> dict[str, float]:
        """Return temperature-dependent intrinsic J_s(T), H_A(T), J_r_ideal(T), and mu_rec(T)."""
        st = self.state
        T_K = float(T_C) + 273.15
        Tc = st.tc_K()
        rJ = float(
            kuzmin_js(T_K, 1.0, Tc=Tc, s=st.kuzmin_s, p=st.kuzmin_p)
            / kuzmin_js(293.15, 1.0, Tc=Tc, s=st.kuzmin_s, p=st.kuzmin_p)
        )
        Js_T = st.js_20c_T() * rJ
        p_ha = st.ha_temp_exponent()
        rHA = max(rJ ** p_ha, 0.02)
        HA_T_Tesla = st.ha_20c_T() * rHA
        HA_T_A_m = HA_T_Tesla / MU0
        # Extra GB-phase thermal nucleation softening
        gb_soft_rate = st.gb_thermal_softening_per_K * (1.0 + 1.5 * (1.0 - st.gb_continuity) + 0.8 * st.j_gb)
        alpha_T_mult = float(np.clip(1.0 - gb_soft_rate * (T_C - 20.0), 0.55, 1.15))

        # Ideal saturation remanence before self-demagnetization of zero-coercivity grains
        jr_ideal = (
            Js_T
            * st.density_rel
            * (1.0 - st.gb_phase_fraction)
            * self._cos_eff
            * self._active_vol_frac
            * st.piece_jr_factor
        )
        # Reversible recoil permeability from misaligned grain rotation + intragranular susceptibility
        chi_rot = (Js_T / max(HA_T_Tesla, 0.1)) * st.density_rel * (1.0 - st.gb_phase_fraction) * self._sin2_mean
        chi_base = 0.022 * (st.ha_20c_T() / max(HA_T_Tesla, 0.1)) * rJ
        mu_rec = 1.0 + (chi_rot + chi_base) * self._active_vol_frac
        return {
            "T_C": float(T_C),
            "rJ": rJ,
            "rHA": rHA,
            "Js_T": Js_T,
            "HA_T_Tesla": HA_T_Tesla,
            "HA_T_A_m": HA_T_A_m,
            "alpha_T_mult": alpha_T_mult,
            "jr_ideal_T": jr_ideal,
            "mu_rec": mu_rec,
        }

    def _grain_switching_fields(
        self,
        T_C: float,
        obliquity_rad: float,
        *,
        skin: bool = False,
    ) -> tuple[np.ndarray, float]:
        prm = self.intrinsic_params_at_T(T_C)
        alpha_k = self._alpha_bulk_k * prm["alpha_T_mult"]
        if skin:
            alpha_k = alpha_k * self._skin_factor
        # Kronmueller switching field along grain easy axis (A/m)
        H_sw_0 = alpha_k * prm["HA_T_A_m"] - self._neff_k * (prm["Js_T"] / MU0)
        # Combine external field obliquity with local grain misorientation
        psi_eff = np.sqrt(obliquity_rad * obliquity_rad + self._theta_grain_k * self._theta_grain_k)
        f_ang = angular_switching_factor(psi_eff, self.state.pinning_fraction)
        H_sw = H_sw_0 * f_ang
        # Thermal fluctuation broadening scale (~1.2% of characteristic H_sw, floor 4 kA/m)
        dH_th = max(0.012 * float(np.median(np.maximum(H_sw, 0.0))), 4000.0)
        return H_sw, dH_th

    def _grain_switched_probs(
        self,
        H_parallel_A_m: np.ndarray,
        H_perp_A_m: np.ndarray,
        T_C: float,
        *,
        skin: bool = False,
    ) -> np.ndarray:
        """Compute switched probability matrix shape (n_points, n_grains)."""
        H_par = np.asarray(H_parallel_A_m, dtype=float)
        H_perp = np.asarray(H_perp_A_m, dtype=float)
        H_opp = np.maximum(-H_par, 0.0)
        H_perp_abs = np.abs(H_perp)
        H_tot = np.hypot(H_opp, H_perp_abs)
        # When H_parallel >= 0 (1st quadrant magnetizing field), no demag switching from parallel component;
        # purely transverse field only switches if H_tot > H_sw(pi/2).
        obliq = np.where(H_tot > 1.0, np.arctan2(H_perp_abs, np.maximum(H_opp, 1e-3)), 0.0)

        prm = self.intrinsic_params_at_T(T_C)
        alpha_k = self._alpha_bulk_k * prm["alpha_T_mult"]
        if skin:
            alpha_k = alpha_k * self._skin_factor
        H_sw_0 = alpha_k * prm["HA_T_A_m"] - self._neff_k * (prm["Js_T"] / MU0)  # (G,)
        psi_eff = np.sqrt(obliq[:, None] ** 2 + self._theta_grain_k[None, :] ** 2)  # (N, G)
        f_ang = angular_switching_factor(psi_eff, self.state.pinning_fraction)
        H_sw = H_sw_0[None, :] * f_ang  # (N, G)
        dH_th = max(0.012 * float(np.median(np.maximum(H_sw_0, 0.0))), 4000.0)

        # Effective driving field magnitude (zero when H_par >= 0 and H_perp == 0)
        drive = np.where((H_par < 0.0) | (H_perp_abs > 1.0), H_tot, -1e9)
        probs = _sigmoid_stable((drive[:, None] - H_sw) / dH_th)
        # Ensure clean zero switching at zero applied field unless H_sw <= 0 (intrinsic self-demag)
        zero_mask = (H_par >= 0.0) & (H_perp_abs <= 1.0)
        if np.any(zero_mask):
            self_demag = _sigmoid_stable((-H_sw_0[None, :]) / dH_th)
            probs = np.where(zero_mask[:, None], self_demag, probs)
        return probs

    def switched_fraction(
        self,
        H_parallel_A_m: np.ndarray | float,
        T_C: float,
        H_perp_A_m: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """Irreversible switched volume fraction f_sw in [0, 1] along major descending branch."""
        scalar = np.ndim(H_parallel_A_m) == 0 and np.ndim(H_perp_A_m) == 0
        H_par, H_perp = np.broadcast_arrays(
            np.asarray(H_parallel_A_m, dtype=float),
            np.asarray(H_perp_A_m, dtype=float),
        )
        shape = H_par.shape
        hp = H_par.ravel()
        ht = H_perp.ravel()

        p_bulk = self._grain_switched_probs(hp, ht, T_C, skin=False)
        f_bulk = p_bulk @ self._w_base
        if self._f_skin > 0.0:
            p_skin = self._grain_switched_probs(hp, ht, T_C, skin=True)
            f_skin = p_skin @ self._w_base
            f_tot = (1.0 - self._f_skin) * f_bulk + self._f_skin * f_skin
        else:
            f_tot = f_bulk
        out = np.clip(f_tot.reshape(shape), 0.0, 1.0)
        return float(out) if scalar else out

    def J_descending(
        self,
        H_parallel_A_m: np.ndarray | float,
        T_C: float,
        H_perp_A_m: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """Intrinsic parallel polarization J_parallel (T) on the descending 2nd-quadrant curve."""
        scalar = np.ndim(H_parallel_A_m) == 0 and np.ndim(H_perp_A_m) == 0
        H_par, H_perp = np.broadcast_arrays(
            np.asarray(H_parallel_A_m, dtype=float),
            np.asarray(H_perp_A_m, dtype=float),
        )
        prm = self.intrinsic_params_at_T(T_C)
        f_sw = self.switched_fraction(H_par, T_C, H_perp)
        # Transverse field slightly tilts the unswitched magnetization away from the easy axis
        H_perp_abs = np.abs(H_perp)
        tilt_cos = 1.0 / np.sqrt(1.0 + (H_perp_abs / max(prm["HA_T_A_m"], 1.0)) ** 2)
        J_val = prm["jr_ideal_T"] * tilt_cos * (1.0 - 2.0 * f_sw) + MU0 * (prm["mu_rec"] - 1.0) * H_par
        return float(J_val) if scalar else J_val

    def J_recoil(
        self,
        H_parallel_A_m: np.ndarray | float,
        T_C: float,
        reversal_H_parallel_A_m: float,
        reversal_H_perp_A_m: float = 0.0,
    ) -> np.ndarray:
        """Intrinsic polarization along an isothermal recoil line from a reversal point."""
        scalar = np.ndim(H_parallel_A_m) == 0
        H_par = np.asarray(H_parallel_A_m, dtype=float)
        prm = self.intrinsic_params_at_T(T_C)
        f_rev = float(self.switched_fraction(reversal_H_parallel_A_m, T_C, reversal_H_perp_A_m))
        J_rec = prm["jr_ideal_T"] * (1.0 - 2.0 * f_rev) + MU0 * (prm["mu_rec"] - 1.0) * H_par
        # If H_par goes deeper than reversal_H_parallel_A_m, stay bounded above by major descending curve
        J_desc = self.J_descending(H_par, T_C, 0.0)
        J_val = np.minimum(J_rec, J_desc)
        return float(J_val) if scalar else J_val

    def J_return(
        self,
        H_parallel_A_m: np.ndarray | float,
        T_return_C: float,
        *,
        excursion_T_C: float,
        excursion_H_parallel_A_m: np.ndarray | float,
        excursion_H_perp_A_m: np.ndarray | float = 0.0,
        return_H_perp_A_m: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """Retained intrinsic polarization at T_return_C after a hot excursion (grain-by-grain memory).

        Captures both reversible thermal recovery (J_s(T_return)/J_s(T_hot)) and frozen
        permanent demagnetization of any grain class switched during the hot excursion.
        """
        scalar = (
            np.ndim(H_parallel_A_m) == 0
            and np.ndim(excursion_H_parallel_A_m) == 0
            and np.ndim(excursion_H_perp_A_m) == 0
            and np.ndim(return_H_perp_A_m) == 0
        )
        H_par, H_ret_perp, H_exc_par, H_exc_perp = np.broadcast_arrays(
            np.asarray(H_parallel_A_m, dtype=float),
            np.asarray(return_H_perp_A_m, dtype=float),
            np.asarray(excursion_H_parallel_A_m, dtype=float),
            np.asarray(excursion_H_perp_A_m, dtype=float),
        )
        shape = H_par.shape
        hp = H_par.ravel()
        ht = H_ret_perp.ravel()
        ep = H_exc_par.ravel()
        et = H_exc_perp.ravel()

        p_hot_b = self._grain_switched_probs(ep, et, excursion_T_C, skin=False)
        p_cold_b = self._grain_switched_probs(hp, ht, T_return_C, skin=False)
        f_bulk = np.maximum(p_hot_b, p_cold_b) @ self._w_base
        if self._f_skin > 0.0:
            p_hot_s = self._grain_switched_probs(ep, et, excursion_T_C, skin=True)
            p_cold_s = self._grain_switched_probs(hp, ht, T_return_C, skin=True)
            f_skin = np.maximum(p_hot_s, p_cold_s) @ self._w_base
            f_eff = (1.0 - self._f_skin) * f_bulk + self._f_skin * f_skin
        else:
            f_eff = f_bulk

        prm = self.intrinsic_params_at_T(T_return_C)
        tilt_cos = 1.0 / np.sqrt(1.0 + (np.abs(ht) / max(prm["HA_T_A_m"], 1.0)) ** 2)
        J_val = prm["jr_ideal_T"] * tilt_cos * (1.0 - 2.0 * np.clip(f_eff, 0.0, 1.0)) + MU0 * (prm["mu_rec"] - 1.0) * hp
        out = J_val.reshape(shape)
        return float(out) if scalar else out

    def irreversible_loss_fraction(
        self,
        excursion_H_parallel_A_m: np.ndarray | float,
        excursion_T_C: float,
        excursion_H_perp_A_m: np.ndarray | float = 0.0,
        *,
        reference_T_C: float = 20.0,
    ) -> np.ndarray:
        """Permanent fractional remanence loss at reference_T_C after hot excursion."""
        J_init = float(self.J_descending(0.0, reference_T_C, 0.0))
        J_ret = self.J_return(
            0.0,
            reference_T_C,
            excursion_T_C=excursion_T_C,
            excursion_H_parallel_A_m=excursion_H_parallel_A_m,
            excursion_H_perp_A_m=excursion_H_perp_A_m,
        )
        if J_init <= 1e-12:
            return np.zeros_like(J_ret) if np.ndim(J_ret) else 0.0
        loss = np.clip((J_init - J_ret) / J_init, 0.0, 2.0)
        return float(loss) if np.ndim(loss) == 0 else loss

    def emit_curve_record(
        self,
        T_C: float,
        *,
        branch: str = "descending",
        n_points: int = 161,
        H_max_mag_A_m: float | None = None,
        obliquity_deg: float = 0.0,
        reversal_H_A_m: float | None = None,
        return_from_T_C: float | None = None,
        reversal_H_perp_A_m: float = 0.0,
        curve_id: str | None = None,
    ) -> MaterialCurveRecord:
        """Emit a serialized Record-1 MaterialCurveRecord for one (T, branch, obliquity)."""
        if branch not in VALID_BRANCHES:
            raise ValueError(f"Unsupported branch: {branch!r}")
        obliq_rad = math.radians(float(obliquity_deg))

        if H_max_mag_A_m is None:
            # Automatically span well past H_cj at T_C so H_cj and H_k90 are always bracketed
            prm = self.intrinsic_params_at_T(T_C)
            est_hcj = max(
                float(np.median(self._alpha_bulk_k)) * prm["alpha_T_mult"] * prm["HA_T_A_m"]
                - float(np.median(self._neff_k)) * (prm["Js_T"] / MU0),
                2.0e5,
            )
            H_max_mag_A_m = max(1.45 * est_hcj, 4.0e5)

        if branch == "descending":
            H_mag = np.linspace(0.0, float(H_max_mag_A_m), int(n_points))
            H_par = -H_mag
            H_perp = H_mag * math.tan(obliq_rad) if obliquity_deg > 0.0 else np.zeros_like(H_par)
            J_arr = self.J_descending(H_par, T_C, H_perp)
            rev_val = None
            extra_meta: dict[str, Any] = {
                "obliquity_deg": float(obliquity_deg),
                "pinning_fraction": float(self.state.pinning_fraction),
                "microstructure": self.state.to_dict(),
            }
        elif branch == "recoil":
            if reversal_H_A_m is None or reversal_H_A_m > 0.0:
                raise ValueError("branch='recoil' requires reversal_H_A_m <= 0.0 (2nd quadrant)")
            H_par = np.linspace(0.0, float(reversal_H_A_m), int(n_points))
            J_arr = self.J_recoil(H_par, T_C, float(reversal_H_A_m), float(reversal_H_perp_A_m))
            rev_val = float(reversal_H_A_m)
            extra_meta = {
                "obliquity_deg": float(obliquity_deg),
                "reversal_H_perp_A_m": float(reversal_H_perp_A_m),
                "microstructure": self.state.to_dict(),
            }
        else:  # branch == "return"
            if reversal_H_A_m is None or reversal_H_A_m > 0.0:
                raise ValueError("branch='return' requires reversal_H_A_m <= 0.0 (2nd quadrant)")
            exc_T = float(return_from_T_C if return_from_T_C is not None else max(T_C, 90.0))
            H_par = np.linspace(0.0, float(reversal_H_A_m), int(n_points))
            J_arr = self.J_return(
                H_par,
                T_C,
                excursion_T_C=exc_T,
                excursion_H_parallel_A_m=float(reversal_H_A_m),
                excursion_H_perp_A_m=float(reversal_H_perp_A_m),
            )
            rev_val = float(reversal_H_A_m)
            extra_meta = {
                "return_from_T_C": exc_T,
                "reversal_H_perp_A_m": float(reversal_H_perp_A_m),
                "microstructure": self.state.to_dict(),
            }

        metrics = compute_curve_metrics(H_par, J_arr)
        extra_meta["metrics"] = metrics
        extra_meta["intrinsic_at_T"] = self.intrinsic_params_at_T(T_C)

        cid = curve_id or _stable_id(
            f"crv-{self.state.name}-{int(round(T_C))}C-{branch[:4]}",
            {
                "lot_id": self.lot_id,
                "specimen_id": self.specimen_id,
                "process_stage": self.process_stage,
                "T_C": round(float(T_C), 3),
                "branch": branch,
                "obliquity_deg": round(float(obliquity_deg), 3),
                "reversal_H_A_m": round(float(rev_val), 3) if rev_val is not None else None,
                "reversal_H_perp_A_m": round(abs(float(reversal_H_perp_A_m)), 3) if rev_val is not None else None,
                "return_from_T_C": round(float(exc_T), 3) if branch == "return" else None,
                "joint_draw_id": self.joint_draw_id,
                "microstructure": self.state.to_dict(),
                "geometry": dict(self.geometry),
            },
        )
        return MaterialCurveRecord(
            curve_id=cid,
            parent_curve_id=self.parent_curve_id,
            source_ref=self.source_ref,
            evidence_class=self.evidence_class,
            lot_id=self.lot_id,
            specimen_id=self.specimen_id,
            process_stage=self.process_stage,
            geometry=dict(self.geometry),
            texture_axis=list(self.texture_axis),
            density_kg_m3=self.state.density_kg_m3(),
            T_C=float(T_C),
            H_parallel_A_m=[round(float(x), 3) for x in H_par],
            J_parallel_T=[round(float(x), 6) for x in J_arr],
            branch=branch,
            reversal_H_A_m=rev_val,
            uncertainty=dict(self.uncertainty),
            joint_draw_id=self.joint_draw_id,
            sharing_scope=self.sharing_scope,
            metadata=extra_meta,
        )


def transfer_parent_block_to_part(
    parent: AlignedCurveModel,
    *,
    thickness_m: float,
    surface_area_m2: float | None = None,
    volume_m3: float | None = None,
    damaged_skin_m: float = 10.0e-6,
    skin_hcj_factor: float = 0.50,
    coating_thickness_m: float = 25.0e-6,
    coating_included_in_dims: bool = True,
    orientation_scatter_deg: float = 1.5,
    piece_jr_factor: float = 1.0,
    piece_hci_factor: float = 1.0,
    specimen_id: str = "finished-thin-part",
    process_stage: str = "machined_coated_part",
    parent_curve_id: str | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> AlignedCurveModel:
    """Transfer a parent-block material lot to a machined/coated thin finished part.

    Explicitly accounts for post-coating volume displacement, machining-damaged
    low-coercivity surface skin (which softens H_k90/H_cj on thin parts), and
    piece-to-piece orientation scatter while preserving lot_id and joint_draw_id.
    """
    if thickness_m <= 0.0:
        raise ValueError("thickness_m must be positive")
    if surface_area_m2 is not None and volume_m3 is not None and volume_m3 > 0.0:
        surf_to_vol = float(surface_area_m2) / float(volume_m3)
    else:
        # Thin-plate approximation (2 major faces + ~15% perimeter contribution)
        surf_to_vol = 2.15 / float(thickness_m)

    f_coat = float(np.clip(coating_thickness_m * surf_to_vol, 0.0, 0.45)) if coating_included_in_dims else 0.0
    # Active core volume has slightly higher surface/volume ratio after coating removal
    eff_t = max(float(thickness_m) - (2.0 * coating_thickness_m if coating_included_in_dims else 0.0), 1e-5)
    f_dmg = float(np.clip((damaged_skin_m * surf_to_vol) / max(1.0 - f_coat, 0.2), 0.0, 0.50))

    st = parent.state
    new_state = MicrostructureState(
        **{
            **st.to_dict(),
            "name": f"{st.name}_part",
            "damaged_skin_fraction": f_dmg,
            "skin_hcj_factor": float(skin_hcj_factor),
            "coating_volume_fraction": f_coat,
            "orientation_scatter_deg": float(orientation_scatter_deg),
            "piece_jr_factor": float(piece_jr_factor),
            "piece_hci_factor": float(piece_hci_factor),
        }
    )
    geom = dict(
        geometry
        or {
            "kind": "thin_finished_part",
            "thickness_m": float(thickness_m),
            "effective_magnetic_thickness_m": eff_t,
            "surface_to_volume_m_inv": surf_to_vol,
            "damaged_skin_m": float(damaged_skin_m),
            "coating_thickness_m": float(coating_thickness_m),
            "coating_included_in_dims": bool(coating_included_in_dims),
        }
    )
    ref_parent_cid = parent_curve_id or parent.emit_curve_record(20.0).curve_id
    return AlignedCurveModel(
        new_state,
        lot_id=parent.lot_id,
        specimen_id=specimen_id,
        process_stage=process_stage,
        source_ref=f"{parent.source_ref}:block_to_part_transfer",
        evidence_class=parent.evidence_class,
        joint_draw_id=parent.joint_draw_id,
        parent_curve_id=ref_parent_cid,
        geometry=geom,
        texture_axis=parent.texture_axis,
        uncertainty=dict(parent.uncertainty),
        sharing_scope=parent.sharing_scope,
    )


def generate_curve_family(
    model: AlignedCurveModel,
    temperatures_C: Sequence[float] = (20.0, 75.0, 90.0, 105.0, 120.0),
    *,
    obliquities_deg: Sequence[float] = (0.0,),
    recoil_H_A_m: Sequence[float] = (),
    return_T_C: float = 20.0,
    n_points: int = 161,
    H_max_mag_A_m: float | None = None,
) -> list[MaterialCurveRecord]:
    """Emit a complete correlated family of Record-1 curves for one material lot."""
    records: list[MaterialCurveRecord] = []
    for T_C in temperatures_C:
        for obliq in obliquities_deg:
            records.append(
                model.emit_curve_record(
                    float(T_C),
                    branch="descending",
                    n_points=n_points,
                    H_max_mag_A_m=H_max_mag_A_m,
                    obliquity_deg=float(obliq),
                )
            )
        for h_rev in recoil_H_A_m:
            h_neg = -abs(float(h_rev))
            records.append(
                model.emit_curve_record(
                    float(T_C),
                    branch="recoil",
                    n_points=max(41, n_points // 2),
                    reversal_H_A_m=h_neg,
                )
            )
            if abs(float(T_C) - float(return_T_C)) > 1e-6:
                records.append(
                    model.emit_curve_record(
                        float(return_T_C),
                        branch="return",
                        n_points=max(41, n_points // 2),
                        reversal_H_A_m=h_neg,
                        return_from_T_C=float(T_C),
                    )
                )
    return records


def sample_aligned_curve_ensemble(
    n_draws: int = 16,
    *,
    base_state: MicrostructureState | str = "virgin_n42_baseline",
    seed: int = 20261125,
    use_twin_posteriors: bool = False,
    posterior_dir: str | Path | None = None,
    bundle: Any | None = None,
) -> list[AlignedCurveModel]:
    """Sample correlated aligned 0% HRE curve models carrying posterior/lot uncertainty.

    When ``use_twin_posteriors=True``, requires an explicit ``bundle`` providing
    correlated intrinsic ``Js``, ``H_A``, ``n_B``, ``d_B``, and intergranular exchange
    ``j_gb`` weights with the grain-network coercivity link.
    """
    st0 = PRESET_MICROSTRUCTURES[base_state] if isinstance(base_state, str) else base_state
    rng = np.random.default_rng(int(seed))
    n = int(n_draws)
    if n < 1:
        raise ValueError("n_draws must be >= 1")

    bma_meta: list[dict[str, Any]] = [{} for _ in range(n)]
    if use_twin_posteriors:
        if bundle is None:
            raise ValueError("Pass an explicit bundle when use_twin_posteriors=True in standalone mode")
        b = bundle
        mlab, midx = b.assign_bma("magnetic", n, rng)
        js_scale = np.empty(n, dtype=float)
        ha_scale = np.empty(n, dtype=float)
        nB_arr = np.empty(n, dtype=float)
        dB_arr = np.empty(n, dtype=float)
        for i in range(n):
            mech = str(mlab[i])
            idx = int(midx[i])
            p = b.post[mech]
            js_scale[i] = float(p["Js"][idx]) / 16.1
            ha_scale[i] = float(p["HA"][idx]) / 75.0 if "HA" in p else 1.0
            nB_arr[i] = st0.n_eff_b * (float(p["nB"][idx]) / 0.17) if "nB" in p else st0.n_eff_b
            dB_arr[i] = float(p["dB"][idx]) if "dB" in p else st0.d_b_um
            bma_meta[i] = {
                "bma_mechanism": mech,
                "bma_posterior_index": idx,
                "posterior_hashes": dict(b.hashes),
                "coupling": "paired_bma_relative_Js_HA_nB_and_dB",
            }
        # Keep virgin aligned sintered GB mostly decoupled around base_state.j_gb, modulated by transfer prior
        jgb_draw = np.clip(st0.j_gb + rng.normal(0.0, 0.06, n), 0.0, 1.2)
        src_ref = f"curves:posterior:{b.hashes['posterior_tool'][:12]}"
        ev_class = "twin_posterior"
    else:
        nB_arr = np.clip(rng.normal(st0.n_eff_b, 0.015, n), 0.07, 0.22)
        dB_arr = np.clip(rng.normal(st0.d_b_um, 0.0004, n), 0.002, 0.008)
        js_scale = rng.normal(1.0, 0.008, n)
        ha_scale = rng.normal(1.0, 0.030, n)
        jgb_draw = np.clip(st0.j_gb + rng.normal(0.0, 0.04, n), 0.0, 1.2)
        src_ref = "curves:prior_ensemble"
        ev_class = "microstructure_physics"

    models: list[AlignedCurveModel] = []
    for i in range(n):
        pr_i = float(np.clip(st0.pr_fraction + rng.normal(0.0, 0.02), 0.0, 0.40))
        g_med_i = float(np.clip(st0.grain_median_um * math.exp(rng.normal(0.0, 0.08)), 0.8, 12.0))
        g_sig_i = float(np.clip(st0.grain_sigma_ln + rng.normal(0.0, 0.03), 0.15, 0.65))
        align_i = float(np.clip(st0.alignment_cos + rng.normal(0.0, 0.006), 0.85, 0.992))
        rho_i = float(np.clip(st0.density_rel + rng.normal(0.0, 0.003), 0.95, 0.998))
        c_gb_i = float(np.clip(st0.gb_continuity + rng.normal(0.0, 0.02), 0.70, 0.995))
        alpha_i = float(np.clip(st0.alpha_k_base * math.exp(rng.normal(0.0, 0.045)), 0.20, 0.65))
        pin_i = float(np.clip(st0.pinning_fraction + rng.normal(0.0, 0.08), 0.15, 0.85))
        js_ov = float(st0.js_20c_T() * js_scale[i])
        ha_ov = float(st0.ha_20c_T() * ha_scale[i])

        st_i = MicrostructureState(
            **{
                **st0.to_dict(),
                "name": f"{st0.name}_d{i:03d}",
                "pr_fraction": pr_i,
                "grain_median_um": g_med_i,
                "grain_sigma_ln": g_sig_i,
                "alignment_cos": align_i,
                "density_rel": rho_i,
                "gb_continuity": c_gb_i,
                "j_gb": float(jgb_draw[i]),
                "alpha_k_base": alpha_i,
                "n_eff_b": float(np.clip(nB_arr[i], 0.07, 0.22)),
                "d_b_um": float(np.clip(dB_arr[i], 0.002, 0.008)),
                "pinning_fraction": pin_i,
                "js_20c_override_T": js_ov,
                "ha_20c_override_T": ha_ov,
            }
        )
        draw_id = f"{st0.name}-seed{seed}-draw{i:03d}"
        models.append(
            AlignedCurveModel(
                st_i,
                lot_id=f"lot-{st0.name}-draw{i:03d}",
                specimen_id="aligned-lot-draw",
                process_stage="parent_block",
                source_ref=src_ref,
                evidence_class=ev_class,
                joint_draw_id=draw_id,
                uncertainty={"ensemble_seed": int(seed), "draw_index": i, "n_draws": n, **bma_meta[i]},
                sharing_scope="public_physics",
            )
        )
    return models


def _eval_curve_source(
    source: AlignedCurveModel | MaterialCurveRecord | Mapping[float, MaterialCurveRecord] | Sequence[MaterialCurveRecord],
    T_C: float,
    H_parallel_A_m: float,
    H_perp_A_m: float,
    *,
    return_T_C: float | None = None,
) -> tuple[float, float, str]:
    """Evaluate (J_at_T, J_return_at_cold, resolved_curve_id) for any Record-1 source."""
    if isinstance(source, AlignedCurveModel):
        j_hot = float(source.J_descending(H_parallel_A_m, T_C, H_perp_A_m))
        if return_T_C is not None:
            j_ret = float(
                source.J_return(
                    0.0,
                    return_T_C,
                    excursion_T_C=T_C,
                    excursion_H_parallel_A_m=H_parallel_A_m,
                    excursion_H_perp_A_m=H_perp_A_m,
                )
            )
        else:
            j_ret = j_hot
        return j_hot, j_ret, source.joint_draw_id

    if isinstance(source, MaterialCurveRecord):
        records = [source]
    elif isinstance(source, Mapping):
        records = list(source.values())
    else:
        records = list(source)

    desc = [r for r in records if r.branch == "descending"]
    if not desc:
        raise ValueError("Curve record collection must contain at least one descending curve")
    desc.sort(key=lambda r: r.T_C)

    # Effective field magnitude accounting for obliquity if curve is scalar 1D
    H_opp = max(-float(H_parallel_A_m), 0.0)
    H_perp_abs = abs(float(H_perp_A_m))
    psi = math.atan2(H_perp_abs, max(H_opp, 1e-3)) if (H_opp > 0 or H_perp_abs > 0) else 0.0
    pin = float(desc[0].metadata.get("pinning_fraction", 0.45))
    f_ang = float(angular_switching_factor(psi, pin))
    H_eff = -math.hypot(H_opp, H_perp_abs) / max(f_ang, 0.25) if H_parallel_A_m < 0.0 else float(H_parallel_A_m)

    # Several measured angular rays are not duplicate temperature samples.
    # At fixed total field evaluate each ray at its own parallel projection,
    # then interpolate angle and temperature. Applying f_ang as well would
    # count the oblique switching on the measured ray twice.
    angles = [float(r.metadata.get("obliquity_deg", 0.0)) for r in desc]
    has_angular_rays = any(abs(a) > 1e-8 for a in angles)
    by_T: dict[float, dict[float, MaterialCurveRecord]] = {}
    query = math.degrees(psi)
    def _bracket_indices(grid: np.ndarray, val: float, tol: float = 1e-6) -> list[int]:
        if len(grid) == 1 or val <= float(grid[0]):
            return [0]
        if val >= float(grid[-1]):
            return [len(grid) - 1]
        nearest = int(np.argmin(np.abs(grid - val)))
        if abs(float(grid[nearest]) - val) <= tol:
            return [nearest]
        hi = int(np.searchsorted(grid, val, side="right"))
        return [hi - 1, hi]

    if has_angular_rays:
        for record, ang in zip(desc, angles):
            row = by_T.setdefault(record.T_C, {})
            if ang in row:
                raise ValueError("Ambiguous duplicate temperature-angle ray in curve family")
            row[ang] = record
        if H_parallel_A_m > 0.0:
            raise ValueError("Oblique second-quadrant family requires non-positive H_parallel")
        temps = np.array(sorted(by_T), dtype=float)
        if float(T_C) < temps[0] - 1e-6 or float(T_C) > temps[-1] + 1e-6:
            raise ValueError(
                f"Temperature {T_C} C outside supplied curve family [{temps[0]}, {temps[-1]}] C"
            )
        total_H = math.hypot(H_opp, H_perp_abs)
        t_idx = _bracket_indices(temps, float(T_C))
        t_active = temps[t_idx]
        j_active: list[float] = []
        for t in t_active:
            rays = by_T[float(t)]
            axis = np.array(sorted(rays), dtype=float)
            if total_H > 0.0 and (query < axis[0] - 1e-6 or query > axis[-1] + 1e-6):
                raise ValueError("Obliquity outside supplied angular rays; add brackets or use aligned model")
            if total_H == 0.0:
                j_active.append(float(rays[float(axis[0])].evaluate_J(0.0)))
            else:
                a_idx = _bracket_indices(axis, query)
                a_active = axis[a_idx]
                ray_J = [
                    float(
                        rays[float(a)].evaluate_J(
                            float(H_parallel_A_m)
                            if abs(float(a) - query) <= 1e-6
                            else -total_H * math.cos(math.radians(float(a)))
                        )
                    )
                    for a in a_active
                ]
                j_active.append(float(np.interp(query, a_active, ray_J)))
        j_hot = float(np.interp(float(T_C), t_active, j_active))
        # Reversible fraction is only an estimate for a measured family
        # lacking matched minor-loop/return measurements.
        desc = [by_T[float(t)][float(min(by_T[float(t)]))] for t in temps]
    else:
        temps = np.array([r.T_C for r in desc], dtype=float)
        if float(T_C) < temps[0] - 1e-6 or float(T_C) > temps[-1] + 1e-6:
            raise ValueError(
                f"Temperature {T_C} C outside supplied curve family [{temps[0]}, {temps[-1]}] C"
            )
        t_idx = _bracket_indices(temps, float(T_C))
        t_active = temps[t_idx]
        j_active = [float(desc[i].evaluate_J(H_eff)) for i in t_idx]
        j_hot = float(np.interp(float(T_C), t_active, j_active))

    # Estimate irreversible switched fraction from departure below linear recoil at T_C
    jr_vals = np.array([float(r.metrics["jr_T"]) for r in desc], dtype=float)
    mu_vals = np.array([float(r.metrics.get("mu_rec_initial", 1.05)) for r in desc], dtype=float)
    jr_T = float(np.interp(float(T_C), temps, jr_vals))
    mu_T = float(np.interp(float(T_C), temps, mu_vals))
    j_lin_T = jr_T + MU0 * (mu_T - 1.0) * float(H_parallel_A_m)
    irrev_frac = float(np.clip((j_lin_T - j_hot) / max(2.0 * jr_T, 1e-9), 0.0, 1.0))

    if return_T_C is not None:
        by_hp: dict[float, dict[float, MaterialCurveRecord]] = {}
        for r in records:
            if r.branch != "return" or r.reversal_H_A_m is None:
                continue
            if "return_from_T_C" not in r.metadata or r.metadata["return_from_T_C"] is None:
                continue
            if abs(r.T_C - float(return_T_C)) > 1e-6:
                continue
            if abs(float(r.metadata["return_from_T_C"]) - float(T_C)) > 1e-6:
                continue
            hp_key = round(abs(float(r.metadata.get("reversal_H_perp_A_m", 0.0))), 3)
            rev_key = round(float(r.reversal_H_A_m), 3)
            bucket = by_hp.setdefault(hp_key, {})
            if rev_key in bucket:
                raise ValueError(
                    "Ambiguous duplicate return curve for the same temperature and reversal field"
                )
            bucket[rev_key] = r

        covering_hps = np.array(
            sorted(
                hp_key
                for hp_key, bucket in by_hp.items()
                if min(bucket) - 1e-3 <= float(H_parallel_A_m) <= max(bucket) + 1e-3
            ),
            dtype=float,
        )

        matched_ret = False
        if len(covering_hps) > 0 and covering_hps[0] - 1e-3 <= H_perp_abs <= covering_hps[-1] + 1e-3:
            hp_idx = _bracket_indices(covering_hps, H_perp_abs, tol=1e-3)
            hp_active = covering_hps[hp_idx]
            hp_js: list[float] = []
            for hp_val in hp_active:
                bucket = by_hp[float(hp_val)]
                rev_hs = np.array(sorted(bucket), dtype=float)
                r_idx = _bracket_indices(rev_hs, float(H_parallel_A_m), tol=1e-3)
                r_active = rev_hs[r_idx]
                r_js = np.array([float(bucket[float(h)].evaluate_J(0.0)) for h in r_active], dtype=float)
                hp_js.append(float(np.interp(float(H_parallel_A_m), r_active, r_js)))
            j_ret = float(np.interp(H_perp_abs, hp_active, hp_js))
            matched_ret = True

        if not matched_ret:
            if float(return_T_C) < temps[0] - 1e-6 or float(return_T_C) > temps[-1] + 1e-6:
                raise ValueError(
                    f"Return temperature {return_T_C} C outside supplied curve family [{temps[0]}, {temps[-1]}] C"
                )
            jr_cold = float(np.interp(float(return_T_C), temps, jr_vals))
            j_ret = jr_cold * (1.0 - 2.0 * irrev_frac)
    else:
        j_ret = j_hot

    # Pick closest temperature (and angle) record ID for provenance tracking
    nearest = desc[int(np.argmin(np.abs(temps - float(T_C))))]
    if has_angular_rays:
        row = by_T[nearest.T_C]
        nearest = row[min(row, key=lambda a: (abs(a - query), a))]
    return j_hot, j_ret, nearest.curve_id


def rescore_operating_point(
    record: OperatingPointRecord,
    new_curve: AlignedCurveModel | MaterialCurveRecord | Mapping[float, MaterialCurveRecord] | Sequence[MaterialCurveRecord],
    reference_curve: AlignedCurveModel | MaterialCurveRecord | Mapping[float, MaterialCurveRecord] | Sequence[MaterialCurveRecord],
    *,
    cold_return_T_C: float = 20.0,
) -> dict[str, Any]:
    """Automatically rescore an OperatingPointRecord under a new material curve.

    Uses each bin's exact ``(T_C, H_parallel_A_m, H_perp_A_m, dV_m3, dScore_dJloss_per_T)``
    sensitivity to compute the updated hot operating score, retained cold-return
    score after thermal excursion, and volume-weighted irreversible polarization loss.
    """
    if not record.bins:
        raise ValueError(f"OperatingPointRecord {record.case_id!r} has no operating-point bins")

    total_dV = 0.0
    weighted_dJ_hot = 0.0
    weighted_dJ_ret = 0.0
    delta_score_hot = 0.0
    delta_score_ret = 0.0
    worst_irrev_loss_T = 0.0
    resolved_cid = ""

    for b in record.bins:
        j_ref_hot, j_ref_ret, _ = _eval_curve_source(
            reference_curve, b.T_C, b.H_parallel_A_m, b.H_perp_A_m, return_T_C=cold_return_T_C
        )
        j_new_hot, j_new_ret, resolved_cid = _eval_curve_source(
            new_curve, b.T_C, b.H_parallel_A_m, b.H_perp_A_m, return_T_C=cold_return_T_C
        )
        # Positive dJ_loss means new curve has LOWER polarization than reference
        dJ_loss_hot = j_ref_hot - j_new_hot
        dJ_loss_ret = j_ref_ret - j_new_ret
        delta_score_hot += b.dScore_dJloss_per_T * dJ_loss_hot
        delta_score_ret += b.dScore_dJloss_per_T * dJ_loss_ret

        # Also measure absolute permanent loss on new curve relative to its own unexposed cold Jr
        j_new_cold0, _, _ = _eval_curve_source(new_curve, cold_return_T_C, 0.0, 0.0, return_T_C=cold_return_T_C)
        irrev_loss_T = max(j_new_cold0 - j_new_ret, 0.0)
        worst_irrev_loss_T = max(worst_irrev_loss_T, irrev_loss_T)

        total_dV += b.dV_m3
        weighted_dJ_hot += b.dV_m3 * dJ_loss_hot
        weighted_dJ_ret += b.dV_m3 * dJ_loss_ret

    mean_dJ_hot = weighted_dJ_hot / total_dV if total_dV > 0.0 else 0.0
    mean_dJ_ret = weighted_dJ_ret / total_dV if total_dV > 0.0 else 0.0
    return {
        "case_id": record.case_id,
        "geometry_id": record.geometry_id,
        "reference_curve_id": record.curve_id,
        "new_curve_id": resolved_cid,
        "score_kind": record.score_kind,
        "original_score_ratio": float(record.score_ratio),
        "rescored_hot_ratio": float(record.score_ratio + delta_score_hot),
        "delta_score_hot": float(delta_score_hot),
        "rescored_return_delta": float(delta_score_ret),
        "mean_delta_J_loss_hot_T": float(mean_dJ_hot),
        "mean_delta_J_loss_return_T": float(mean_dJ_ret),
        "worst_bin_permanent_J_loss_T": float(worst_irrev_loss_T),
        "total_magnet_volume_m3": float(total_dV),
    }


def rescore_catalog(
    records: Sequence[OperatingPointRecord],
    new_curve: AlignedCurveModel | MaterialCurveRecord | Mapping[float, MaterialCurveRecord] | Sequence[MaterialCurveRecord],
    reference_curves: Mapping[str, AlignedCurveModel | MaterialCurveRecord | Sequence[MaterialCurveRecord]] | AlignedCurveModel,
    *,
    cold_return_T_C: float = 20.0,
) -> list[dict[str, Any]]:
    """Rescore all linked OperatingPointRecords under a new material curve and rank them."""
    results: list[dict[str, Any]] = []
    for rec in records:
        if isinstance(reference_curves, AlignedCurveModel):
            ref = reference_curves
        else:
            if rec.curve_id not in reference_curves:
                raise KeyError(f"Reference curve_id {rec.curve_id!r} missing for case {rec.case_id!r}")
            ref = reference_curves[rec.curve_id]
        results.append(rescore_operating_point(rec, new_curve, ref, cold_return_T_C=cold_return_T_C))
    results.sort(key=lambda row: row["rescored_hot_ratio"], reverse=True)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    return results


__all__ = [
    "A_M_TO_KOE",
    "KOE_TO_A_M",
    "MU0",
    "PRESET_MICROSTRUCTURES",
    "VALID_BRANCHES",
    "VALID_EVIDENCE_CLASSES",
    "AlignedCurveModel",
    "MaterialCurveRecord",
    "MicrostructureState",
    "OperatingPointBin",
    "OperatingPointRecord",
    "angular_switching_factor",
    "compute_curve_metrics",
    "generate_curve_family",
    "rescore_catalog",
    "rescore_operating_point",
    "sample_aligned_curve_ensemble",
    "transfer_parent_block_to_part",
]
