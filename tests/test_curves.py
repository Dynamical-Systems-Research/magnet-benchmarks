"""Physics and contract tests for aligned 0% HRE J(H, T) curves & 2-record interface."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from magnet_benchmarks import (
    PRESET_MICROSTRUCTURES,
    AlignedCurveModel,
    MaterialCurveRecord,
    OperatingPointBin,
    OperatingPointRecord,
    angular_switching_factor,
    generate_curve_family,
    rescore_catalog,
    rescore_operating_point,
    sample_aligned_curve_ensemble,
    transfer_parent_block_to_part,
)


def test_preset_families_have_realistic_0pct_hre_temperature_and_hk90() -> None:
    """Verify virgin 0% HRE presets produce physical Br, Hcj, beta_Hcj, and derived H_k90."""
    n42 = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n42_baseline"])
    c20 = n42.emit_curve_record(20.0)
    c90 = n42.emit_curve_record(90.0)
    c120 = n42.emit_curve_record(120.0)

    m20, m90, m120 = c20.metrics, c90.metrics, c120.metrics
    # Commercial N42 virgin 0% HRE: Br(20C) ~ 1.28-1.34 T, Hcj(20C) ~ 11.5-14.5 kOe, BHmax ~ 40-44 MGOe
    assert 1.25 <= m20["jr_T"] <= 1.35
    assert 11.0 <= m20["hci_kOe"] <= 15.0
    assert 39.0 <= m20["bhmax_MGOe"] <= 45.0
    assert 0.80 <= m20["squareness_hk90_over_hci"] <= 0.96

    # Derived H_k90 must satisfy J(-H_k90) == 0.9 * J_r to tight interpolation accuracy
    j_at_hk90 = float(c20.evaluate_J(-m20["hk90_A_m"]))
    assert abs(j_at_hk90 - 0.9 * m20["jr_T"]) < 2e-3

    # Temperature coefficients must reflect realistic 0% HRE degradation (not flat -0.12%/C)
    alpha_br_20_90 = (m90["jr_T"] / m20["jr_T"] - 1.0) / 70.0
    alpha_br_90_120 = (m120["jr_T"] / m90["jr_T"] - 1.0) / 30.0
    beta_hci_20_90 = (m90["hci_A_m"] / m20["hci_A_m"] - 1.0) / 70.0
    assert -0.00145 <= alpha_br_20_90 <= -0.00090
    assert alpha_br_90_120 < alpha_br_20_90  # nonlinear steepening toward Tc
    assert -0.0072 <= beta_hci_20_90 <= -0.0048
    assert m120["hci_A_m"] < m90["hci_A_m"] < m20["hci_A_m"]


def test_microstructure_ordering_and_pr_crossover() -> None:
    """Finer grain + higher GB continuity raises Hcj and squareness; high-Br trades coercivity."""
    n42 = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n42_baseline"])
    n48h = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n48h_gb_engineered"])
    n52 = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n52_high_br"])
    high_pr = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n42_high_pr"])

    m42_20 = n42.emit_curve_record(20.0).metrics
    m48_20 = n48h.emit_curve_record(20.0).metrics
    m52_20 = n52.emit_curve_record(20.0).metrics
    m42_105 = n42.emit_curve_record(105.0).metrics
    m48_105 = n48h.emit_curve_record(105.0).metrics
    mpr_20 = high_pr.emit_curve_record(20.0).metrics
    mpr_120 = high_pr.emit_curve_record(120.0).metrics

    assert m52_20["jr_T"] > m48_20["jr_T"] > m42_20["jr_T"]
    assert m48_20["hci_A_m"] > m42_20["hci_A_m"] > m52_20["hci_A_m"]
    assert m48_20["squareness_hk90_over_hci"] > m42_20["squareness_hk90_over_hci"]
    assert m48_105["hk90_A_m"] > m42_105["hk90_A_m"]

    # Higher Pr raises room-temperature Hcj via H_A(20C) but has steeper relative thermal drop to 120 C
    assert mpr_20["hci_A_m"] > m42_20["hci_A_m"]
    beta_pr = (mpr_120["hci_A_m"] / mpr_20["hci_A_m"] - 1.0) / 100.0
    beta_base = (n42.emit_curve_record(120.0).metrics["hci_A_m"] / m42_20["hci_A_m"] - 1.0) / 100.0
    assert beta_pr < beta_base


def test_recoil_and_thermal_return_separate_reversible_and_permanent_loss() -> None:
    """Safe excursion recovers 100% cold Jr; past-knee hot excursion causes permanent cold loss."""
    model = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n42_baseline"])
    m90 = model.emit_curve_record(90.0).metrics
    jr20 = model.emit_curve_record(20.0).metrics["jr_T"]

    # Well inside knee at 90 C -> negligible irreversible loss on return to 20 C
    h_safe = -0.35 * m90["hk90_A_m"]
    loss_safe = model.irreversible_loss_fraction(h_safe, 90.0, reference_T_C=20.0)
    assert loss_safe < 0.005

    # Beyond H_k90 at 90 C -> substantial permanent loss on return to 20 C
    h_deep = -1.05 * m90["hci_A_m"]
    loss_deep = model.irreversible_loss_fraction(h_deep, 90.0, reference_T_C=20.0)
    assert loss_deep > 0.45

    # Recoil curve at 90 C from h_deep must be linear with slope mu0*(mu_rec - 1) and below major curve
    rec_crv = model.emit_curve_record(90.0, branch="recoil", reversal_H_A_m=h_deep)
    ret_crv = model.emit_curve_record(20.0, branch="return", reversal_H_A_m=h_deep, return_from_T_C=90.0)
    assert rec_crv.branch == "recoil"
    assert ret_crv.branch == "return"
    assert rec_crv.J_parallel_T[0] < m90["jr_T"]
    assert ret_crv.J_parallel_T[0] < jr20 * 0.65
    # Recoil slope is positive (J decreases slightly as H becomes more negative)
    assert rec_crv.J_parallel_T[0] >= rec_crv.J_parallel_T[-1]


def test_angular_switching_and_parent_block_to_part_transfer() -> None:
    """Verify angular switching behavior and thin-part machining/coating transfer."""
    sw_45 = float(angular_switching_factor(math.radians(45.0), pinning_fraction=0.0))
    kond_45 = float(angular_switching_factor(math.radians(45.0), pinning_fraction=1.0))
    mix_45 = float(angular_switching_factor(math.radians(45.0), pinning_fraction=0.45))
    assert abs(sw_45 - 0.5) < 1e-6
    assert kond_45 > 1.25
    assert sw_45 < mix_45 < kond_45

    parent = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n45h_fine_grain"])
    thin_part = transfer_parent_block_to_part(
        parent,
        thickness_m=0.0020,
        damaged_skin_m=20.0e-6,
        skin_hcj_factor=0.45,
        coating_thickness_m=25.0e-6,
        coating_included_in_dims=True,
        orientation_scatter_deg=2.0,
    )
    p20 = parent.emit_curve_record(20.0)
    t20 = thin_part.emit_curve_record(20.0)
    assert thin_part.lot_id == parent.lot_id
    assert thin_part.joint_draw_id == parent.joint_draw_id
    assert thin_part.parent_curve_id is not None
    # Post-coat displacement + orientation scatter reduces effective remanence; damaged skin softens H_k90
    assert t20.metrics["jr_T"] < p20.metrics["jr_T"]
    assert t20.metrics["hk90_A_m"] < p20.metrics["hk90_A_m"]


def test_two_record_serialization_and_automatic_rescoring() -> None:
    """Verify exact 2-record schema JSON round-trip and automatic rescoring across curves."""
    base_model = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n42_baseline"])
    better_model = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n48h_gb_engineered"])
    weak_model = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_low_gb_continuity_risk"])

    base_family = generate_curve_family(base_model, temperatures_C=(20.0, 90.0, 105.0), recoil_H_A_m=(4.5e5,))
    better_family = generate_curve_family(better_model, temperatures_C=(20.0, 90.0, 105.0))
    weak_family = generate_curve_family(weak_model, temperatures_C=(20.0, 90.0, 105.0), H_max_mag_A_m=1.4e6)

    # Round-trip Record 1 via JSON and check all 18 required keys exist
    required_r1 = {
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
    }
    raw_r1 = json.loads(json.dumps(base_family[0].to_dict()))
    assert required_r1.issubset(raw_r1.keys())
    rehydrated_r1 = MaterialCurveRecord.from_dict(raw_r1)
    assert rehydrated_r1.curve_id == base_family[0].curve_id

    # Construct two Record-2 candidates: one moderate-field, one high-edge-field
    op_safe = OperatingPointRecord(
        case_id="case-safe",
        geometry_id="geom-a",
        curve_id=base_family[0].curve_id,
        stock_case_id="case-stock",
        seat_condition="fixed_seat",
        finished_stack={"coating_um": 25.0},
        winding_basis={"footprint": "fixed_reference"},
        thermal_state={"T_mag_C": 105.0},
        mass_g=295.0,
        score_kind="retained_winding_linkage_ratio",
        score_ratio=1.00,
        bins=[
            OperatingPointBin(105.0, -2.5e5, 0.3e5, 1.8e-5, -0.55),
            OperatingPointBin(105.0, -4.2e5, 0.8e5, 0.2e-5, -0.15),
        ],
    )
    op_edge = OperatingPointRecord(
        case_id="case-edge",
        geometry_id="geom-b",
        curve_id=base_family[0].curve_id,
        stock_case_id="case-stock",
        seat_condition="fixed_gap",
        finished_stack={"coating_um": 25.0},
        winding_basis={"footprint": "fixed_reference"},
        thermal_state={"T_mag_C": 105.0},
        mass_g=270.0,
        score_kind="retained_winding_linkage_ratio",
        score_ratio=1.02,
        bins=[
            OperatingPointBin(105.0, -3.0e5, 0.4e5, 1.5e-5, -0.50),
            OperatingPointBin(105.0, -5.8e5, 1.2e5, 0.3e-5, -0.35),
        ],
    )
    raw_r2 = json.loads(json.dumps(op_safe.to_dict()))
    required_r2 = {
        "case_id",
        "geometry_id",
        "curve_id",
        "stock_case_id",
        "seat_condition",
        "finished_stack",
        "winding_basis",
        "thermal_state",
        "mass_g",
        "score_kind",
        "score_ratio",
        "bins",
        "bench_observations",
        "sharing_scope",
    }
    assert required_r2.issubset(raw_r2.keys())
    rehydrated_r2 = OperatingPointRecord.from_dict(raw_r2)
    assert len(rehydrated_r2.bins) == 2

    # Rescore using ONLY serialized Record-1 curve lists (as external lab data would arrive)
    res_better = rescore_operating_point(op_edge, better_family, base_family)
    res_weak = rescore_operating_point(op_edge, weak_family, base_family)
    assert res_better["rescored_hot_ratio"] > op_edge.score_ratio
    assert res_weak["rescored_hot_ratio"] < op_edge.score_ratio

    # Weak GB-continuity curve should degrade high-field case-edge much more than case-safe, flipping rank
    ranked_better = rescore_catalog([op_safe, op_edge], better_family, {base_family[0].curve_id: base_family})
    ranked_weak = rescore_catalog([op_safe, op_edge], weak_family, {base_family[0].curve_id: base_family})
    assert ranked_better[0]["case_id"] == "case-edge"
    assert ranked_weak[0]["case_id"] == "case-safe"


def test_correlated_prior_ensemble() -> None:
    """Verify correlated standalone prior ensemble draws carry distinct joint_draw_ids."""
    models = sample_aligned_curve_ensemble(n_draws=6, base_state="virgin_n42_baseline", seed=42, use_twin_posteriors=False)
    assert len(models) == 6
    draw_ids = {m.joint_draw_id for m in models}
    assert len(draw_ids) == 6
    hcj_20 = [m.emit_curve_record(20.0).metrics["hci_kOe"] for m in models]
    hcj_90 = [m.emit_curve_record(90.0).metrics["hci_kOe"] for m in models]
    assert all(h90 < h20 for h20, h90 in zip(hcj_20, hcj_90))
    assert np.std(hcj_20) > 0.1
    assert all(m.evidence_class == "microstructure_physics" for m in models)
    assert all("draw_index" in m.uncertainty and "ensemble_seed" in m.uncertainty for m in models)
    with pytest.raises(ValueError, match="bundle"):
        sample_aligned_curve_ensemble(n_draws=2, use_twin_posteriors=True)


@pytest.mark.parametrize("T_C", [90.0, 105.0])
@pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_multiray_angular_record_lookup_matches_model_without_double_counting(
    T_C: float, angle_deg: float
) -> None:
    """Multi-angle Record-1 lookup interpolates measured rays directly without double-counting f_ang."""
    from magnet_benchmarks.curves import _eval_curve_source

    model = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n42_baseline"])
    records = [
        model.emit_curve_record(t, obliquity_deg=a, n_points=801)
        for t in (90.0, 105.0)
        for a in (0.0, 15.0, 30.0, 45.0, 60.0)
    ]
    hp = -420.0e3 * math.cos(math.radians(angle_deg))
    ht = 420.0e3 * math.sin(math.radians(angle_deg))
    j_eval, _, _ = _eval_curve_source(records, T_C, hp, ht)
    assert j_eval == pytest.approx(model.J_descending(hp, T_C, ht), abs=8e-4)


def test_multiray_angular_lookup_order_invariance_and_no_extrapolation() -> None:
    """Record-order invariance holds for multi-ray families and unbracketed angles raise ValueError."""
    from magnet_benchmarks.curves import _eval_curve_source

    model = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n42_baseline"])
    records = [
        model.emit_curve_record(t, obliquity_deg=a, n_points=401)
        for t in (90.0, 105.0)
        for a in (0.0, 15.0, 30.0, 45.0, 60.0)
    ]
    j_fwd = _eval_curve_source(records, 97.5, -410.0e3, 150.0e3)[0]
    j_rev = _eval_curve_source(list(reversed(records)), 97.5, -410.0e3, 150.0e3)[0]
    assert j_fwd == pytest.approx(j_rev, rel=0.0, abs=1e-12)
    with pytest.raises(ValueError, match="Obliquity"):
        _eval_curve_source(records, 90.0, -1.0, 500.0e3)


def test_distinct_part_ids_truncated_metrics_and_return_records() -> None:
    """Part curve IDs distinguish skin thickness, truncated curves return nan crossings, and temperature/return bounds are honored."""
    from magnet_benchmarks import compute_curve_metrics
    from magnet_benchmarks.curves import _eval_curve_source

    parent = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n48h_gb_engineered"])
    p10 = transfer_parent_block_to_part(parent, thickness_m=0.002, damaged_skin_m=10e-6)
    p20 = transfer_parent_block_to_part(parent, thickness_m=0.002, damaged_skin_m=20e-6)
    c10 = p10.emit_curve_record(105.0)
    c20 = p20.emit_curve_record(105.0)
    assert c10.curve_id != c20.curve_id

    # Truncated curve that stays above 0.9 * Jr must report nan crossings rather than endpoint H
    trunc_m = compute_curve_metrics(np.array([0.0, -50_000.0, -100_000.0]), np.array([1.30, 1.29, 1.28]))
    assert math.isnan(trunc_m["hk90_A_m"])
    assert math.isnan(trunc_m["hci_A_m"])
    assert math.isnan(trunc_m["hcb_A_m"])
    assert math.isnan(trunc_m["squareness_hk90_over_hci"])

    # Out-of-range temperature raises ValueError rather than silently clamping
    fam = generate_curve_family(parent, temperatures_C=(20.0, 90.0, 105.0))
    with pytest.raises(ValueError, match="outside supplied curve family"):
        _eval_curve_source(fam, 120.0, -400_000.0, 0.0)

    # Supplied return record overrides descending linear-recoil approximation
    ret_lo = parent.emit_curve_record(20.0, branch="return", reversal_H_A_m=-450_000.0, return_from_T_C=105.0)
    ret_shifted = MaterialCurveRecord.from_dict({**ret_lo.to_dict(), "J_parallel_T": [0.95] * len(ret_lo.J_parallel_T)})
    _, j_ret_default, _ = _eval_curve_source(fam, 105.0, -450_000.0, 0.0, return_T_C=20.0)
    _, j_ret_supplied, _ = _eval_curve_source([*fam, ret_shifted], 105.0, -450_000.0, 0.0, return_T_C=20.0)
    assert j_ret_supplied == pytest.approx(0.95, abs=1e-6)
    assert abs(j_ret_supplied - j_ret_default) > 0.2


def test_return_curve_identity_order_invariance_and_strict_coverage() -> None:
    """Return/recoil IDs distinguish H_perp, selection is order-invariant, and unbracketed fields/temps are not silently clamped."""
    from magnet_benchmarks.curves import _eval_curve_source

    n52 = transfer_parent_block_to_part(
        AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n52_high_br"]),
        thickness_m=0.002,
        damaged_skin_m=15e-6,
    )
    fam = generate_curve_family(n52, temperatures_C=(20.0, 90.0, 105.0))

    # Distinct curve_ids when reversal_H_perp_A_m differs (both return and recoil branches)
    ret_0 = n52.emit_curve_record(
        20.0, branch="return", reversal_H_A_m=-350_000.0, reversal_H_perp_A_m=0.0, return_from_T_C=105.0
    )
    ret_50 = n52.emit_curve_record(
        20.0, branch="return", reversal_H_A_m=-350_000.0, reversal_H_perp_A_m=50_000.0, return_from_T_C=105.0
    )
    rec_0 = n52.emit_curve_record(105.0, branch="recoil", reversal_H_A_m=-350_000.0, reversal_H_perp_A_m=0.0)
    rec_50 = n52.emit_curve_record(105.0, branch="recoil", reversal_H_A_m=-350_000.0, reversal_H_perp_A_m=50_000.0)
    assert ret_0.curve_id != ret_50.curve_id
    assert rec_0.curve_id != rec_50.curve_id

    # Order invariance when both transverse-field return records are supplied
    op = OperatingPointRecord(
        case_id="c-n52-oblique",
        geometry_id="g-slice",
        curve_id=fam[0].curve_id,
        stock_case_id="stock",
        seat_condition="fixed",
        finished_stack={},
        winding_basis={},
        thermal_state={"T_mag_C": 105.0},
        mass_g=10.0,
        score_kind="test",
        score_ratio=1.0,
        bins=[OperatingPointBin(105.0, -350_000.0, 50_000.0, 1e-6, -1.0)],
    )
    r_ab = rescore_operating_point(op, [*fam, ret_0, ret_50], fam)
    r_ba = rescore_operating_point(op, [*fam, ret_50, ret_0], fam)
    assert r_ab["worst_bin_permanent_J_loss_T"] == pytest.approx(r_ba["worst_bin_permanent_J_loss_T"], abs=1e-12)
    assert r_ab["mean_delta_J_loss_return_T"] == pytest.approx(r_ba["mean_delta_J_loss_return_T"], abs=1e-12)
    _, j_ret_ab, _ = _eval_curve_source([*fam, ret_0, ret_50], 105.0, -350_000.0, 50_000.0, return_T_C=20.0)
    assert j_ret_ab == pytest.approx(ret_50.evaluate_J(0.0), abs=1e-6)

    # Duplicate return record at the exact same (T_C, return_from_T_C, H_par, H_perp) raises ValueError
    with pytest.raises(ValueError, match="Ambiguous duplicate return curve"):
        _eval_curve_source([*fam, ret_50, ret_50], 105.0, -350_000.0, 50_000.0, return_T_C=20.0)

    # Unbracketed -450 kA/m return record is NOT silently clamped onto a -350 kA/m query
    ret_450 = n52.emit_curve_record(
        20.0, branch="return", reversal_H_A_m=-450_000.0, reversal_H_perp_A_m=50_000.0, return_from_T_C=105.0
    )
    _, j_ret_base, _ = _eval_curve_source(fam, 105.0, -350_000.0, 50_000.0, return_T_C=20.0)
    _, j_ret_unbracketed, _ = _eval_curve_source([*fam, ret_450], 105.0, -350_000.0, 50_000.0, return_T_C=20.0)
    assert j_ret_unbracketed == pytest.approx(j_ret_base, abs=1e-12)

    # Bracketed [-450 kA/m, -300 kA/m] return records at matching H_perp interpolate cleanly at -350 kA/m
    ret_300 = n52.emit_curve_record(
        20.0, branch="return", reversal_H_A_m=-300_000.0, reversal_H_perp_A_m=50_000.0, return_from_T_C=105.0
    )
    _, j_ret_bracketed, _ = _eval_curve_source(
        [*fam, ret_450, ret_300], 105.0, -350_000.0, 50_000.0, return_T_C=20.0
    )
    expected_interp = (1.0 / 3.0) * float(ret_450.evaluate_J(0.0)) + (2.0 / 3.0) * float(ret_300.evaluate_J(0.0))
    assert j_ret_bracketed == pytest.approx(expected_interp, abs=1e-6)

    # A 20 C return record is NOT silently accepted for a 19.5 C return query (raises out-of-range)
    with pytest.raises(ValueError, match="Return temperature 19.5 C outside"):
        _eval_curve_source([*fam, ret_50], 105.0, -350_000.0, 50_000.0, return_T_C=19.5)


def test_truncated_descending_and_return_records_raise_out_of_range() -> None:
    """Queries outside a curve's sampled H range (including zero field on return records) raise ValueError."""
    from magnet_benchmarks.curves import _eval_curve_source

    n52 = transfer_parent_block_to_part(
        AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n52_high_br"]),
        thickness_m=0.002,
    )
    full = [n52.emit_curve_record(t) for t in (20.0, 105.0)]
    limited = [n52.emit_curve_record(t, H_max_mag_A_m=350_000.0) for t in (20.0, 105.0)]

    # Within sampled [-350, 0] kA/m range succeeds
    j_at_350, _, _ = _eval_curve_source(limited, 105.0, -350_000.0, 0.0, return_T_C=20.0)
    assert j_at_350 == pytest.approx(float(full[1].evaluate_J(-350_000.0)), abs=5e-3)

    # Beyond sampled -350 kA/m (including small boundary overshoot -350000.25) raises rather than clamping
    for h_oob in (-450_000.0, -350_000.25):
        with pytest.raises(ValueError, match="outside sampled curve range"):
            _eval_curve_source(limited, 105.0, h_oob, 0.0, return_T_C=20.0)
    op_deep = OperatingPointRecord(
        case_id="c-deep",
        geometry_id="g-slice",
        curve_id=full[0].curve_id,
        stock_case_id="stock",
        seat_condition="fixed",
        finished_stack={},
        winding_basis={},
        thermal_state={"T_mag_C": 105.0},
        mass_g=1.0,
        score_kind="test",
        score_ratio=1.0,
        bins=[OperatingPointBin(105.0, -450_000.0, 0.0, 1.0, -1.0)],
    )
    with pytest.raises(ValueError, match="outside sampled curve range"):
        rescore_operating_point(op_deep, limited, full)

    # Synthetic return record omitting H=0 (even ending at -0.25 A/m) raises rather than reusing endpoint as J(0)
    ret_valid = n52.emit_curve_record(
        20.0, branch="return", reversal_H_A_m=-450_000.0, reversal_H_perp_A_m=50_000.0, return_from_T_C=105.0
    )
    for h_top in (-50_000.0, -0.25):
        ret_trunc = MaterialCurveRecord.from_dict(
            {
                **ret_valid.to_dict(),
                "H_parallel_A_m": [-450_000.0, -250_000.0, h_top],
                "J_parallel_T": [0.70, 0.72, 0.74],
            }
        )
        with pytest.raises(ValueError, match="outside sampled curve range"):
            _eval_curve_source([*full, ret_trunc], 105.0, -450_000.0, 50_000.0, return_T_C=20.0)
        op_ret = OperatingPointRecord(
            case_id="c-ret-trunc",
            geometry_id="g-slice",
            curve_id=full[0].curve_id,
            stock_case_id="stock",
            seat_condition="fixed",
            finished_stack={},
            winding_basis={},
            thermal_state={"T_mag_C": 105.0},
            mass_g=1.0,
            score_kind="test",
            score_ratio=1.0,
            bins=[OperatingPointBin(105.0, -450_000.0, 50_000.0, 1.0, -1.0)],
        )
        with pytest.raises(ValueError, match="outside sampled curve range"):
            rescore_operating_point(op_ret, [*full, ret_trunc], full)

    # An unused truncated return record at another reversal field does NOT invalidate an exact valid match
    ret_unused_trunc = MaterialCurveRecord.from_dict(
        {
            **n52.emit_curve_record(
                20.0, branch="return", reversal_H_A_m=-300_000.0, reversal_H_perp_A_m=50_000.0, return_from_T_C=105.0
            ).to_dict(),
            "H_parallel_A_m": [-300_000.0, -150_000.0, -50_000.0],
            "J_parallel_T": [0.70, 0.72, 0.74],
        }
    )
    _, j_ret_exact, _ = _eval_curve_source([*full, ret_valid], 105.0, -450_000.0, 50_000.0, return_T_C=20.0)
    _, j_ret_with_unused, _ = _eval_curve_source(
        [*full, ret_valid, ret_unused_trunc], 105.0, -450_000.0, 50_000.0, return_T_C=20.0
    )
    assert j_ret_with_unused == pytest.approx(j_ret_exact, abs=1e-12)
    res_only = rescore_operating_point(op_ret, [*full, ret_valid], full)
    res_with_unused = rescore_operating_point(op_ret, [*full, ret_valid, ret_unused_trunc], full)
    assert res_with_unused["mean_delta_J_loss_return_T"] == pytest.approx(
        res_only["mean_delta_J_loss_return_T"], abs=1e-12
    )

    # Exact sampled angular-ray endpoint must not fail on trig round-trip ULP drift
    n42 = AlignedCurveModel(PRESET_MICROSTRUCTURES["virgin_n42_baseline"])
    angle = 15.0
    axial = -350_000.0
    transverse = -axial * math.tan(math.radians(angle))
    angular = [n42.emit_curve_record(t, H_max_mag_A_m=-axial, obliquity_deg=angle) for t in (20.0, 105.0)]
    full_n42 = generate_curve_family(n42, temperatures_C=(20.0, 105.0), H_max_mag_A_m=1.4e6)
    op_ang = OperatingPointRecord(
        case_id="c-ang-end",
        geometry_id="g-slice",
        curve_id=full_n42[0].curve_id,
        stock_case_id="stock",
        seat_condition="fixed",
        finished_stack={},
        winding_basis={},
        thermal_state={"T_mag_C": 105.0},
        mass_g=1.0,
        score_kind="test",
        score_ratio=1.0,
        bins=[OperatingPointBin(105.0, axial, transverse, 1.0, -1.0)],
    )
    res_ang = rescore_operating_point(op_ang, angular, full_n42)
    assert math.isfinite(res_ang["rescored_hot_ratio"])

    # Single-record return tolerance boundary (both H_perp and H_parallel) never raises IndexError
    ret_100k = n42.emit_curve_record(
        20.0, branch="return", reversal_H_A_m=-450_000.0, reversal_H_perp_A_m=100_000.0, return_from_T_C=105.0
    )
    for h_par_q, h_perp_q in ((-450_000.0, 100_000.001), (-450_000.001, 100_000.0), (-449_999.999, 99_999.999)):
        op_tol = OperatingPointRecord(
            case_id="c-ret-tol",
            geometry_id="g-slice",
            curve_id=full_n42[0].curve_id,
            stock_case_id="stock",
            seat_condition="fixed",
            finished_stack={},
            winding_basis={},
            thermal_state={"T_mag_C": 105.0},
            mass_g=1.0,
            score_kind="test",
            score_ratio=1.0,
            bins=[OperatingPointBin(105.0, h_par_q, h_perp_q, 1.0, -1.0)],
        )
        res_tol = rescore_operating_point(op_tol, [*full_n42, ret_100k], full_n42)
        assert math.isfinite(res_tol["rescored_return_delta"])




