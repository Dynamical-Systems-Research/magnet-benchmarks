"""Open standalone demagnetization curve model for 0% HRE NdFeB magnets."""

from .curves import (
    PRESET_MICROSTRUCTURES,
    VALID_BRANCHES,
    VALID_EVIDENCE_CLASSES,
    AlignedCurveModel,
    MaterialCurveRecord,
    MicrostructureState,
    OperatingPointBin,
    OperatingPointRecord,
    angular_switching_factor,
    compute_curve_metrics,
    generate_curve_family,
    rescore_catalog,
    rescore_operating_point,
    sample_aligned_curve_ensemble,
    transfer_parent_block_to_part,
)

__all__ = [
    "AlignedCurveModel",
    "MaterialCurveRecord",
    "MicrostructureState",
    "OperatingPointBin",
    "OperatingPointRecord",
    "PRESET_MICROSTRUCTURES",
    "VALID_BRANCHES",
    "VALID_EVIDENCE_CLASSES",
    "angular_switching_factor",
    "compute_curve_metrics",
    "generate_curve_family",
    "rescore_catalog",
    "rescore_operating_point",
    "sample_aligned_curve_ensemble",
    "transfer_parent_block_to_part",
]
