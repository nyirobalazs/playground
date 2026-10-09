"""Montage constants and the default configuration.

Everything trial-specific (calibration spans, validation spans) defaults to ``None`` and is either derived from the
recording by a stated rule or passed in by the caller.  ``merge_config`` deep-merges user overrides (for example a
JSON file given to the CLI with ``--config``) into :data:`DEFAULT_CONFIG`.
"""
from __future__ import annotations

import copy
import json

# --------------------------------------------------------------------------------------------- montage constants
#: Cap spelling -> MNE standard 10-20 spelling.
MNE_NAMES = {"FP1": "Fp1", "FP2": "Fp2"}

#: Sensorimotor channels consumed by the downstream band-power decoder (selected BY NAME, never by index).
DECODER_17 = ["F3", "Fz", "F4", "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4",
              "CP5", "CP1", "CP2", "CP6", "P3", "Pz", "P4"]

#: Robust common-average set: no ocular poles (FP/AF/F7/F8), no T7/T8 (reference-adjacent / EMG-prone) and no
#: bridged duplicates.  Channels dropped by the bridge detector are removed from this set at run time.
ROBUST_CAR20 = ["F3", "Fz", "F4", "FC5", "FC1", "FC2", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
                "P7", "P3", "Pz", "P4", "P8", "PO7", "Oz"]

#: Channels used as EOG proxies.  They are delivered uncorrected, typed ``eog``, and excluded from references and
#: features.
EOG_POLES = ["FP1", "FP2", "F7", "F8"]

#: Bipolar EOG proxy definitions (computed from the untouched recording, so they are reference-free).
VEOG_POS, VEOG_NEG = ["FP1", "FP2"], ["PO7", "PO3", "PO4", "PO8", "Oz"]
HEOG_POS, HEOG_NEG = ["F8"], ["F7"]

#: Independent ocular proxy that the correction never uses (for validation only).
HOLDOUT = ("AF4", "AF3")

#: Small Laplacians for mu checks (C4 avoids FC6, which is commonly a bridged duplicate of F4 on this cap).
LAPLACIANS = {"C3": ["FC5", "FC1", "CP5", "CP1"], "Cz": ["FC1", "FC2", "CP1", "CP2"], "C4": ["FC2", "CP2", "CP6", "Cz"]}

#: Amplifier DI-trigger latency relative to the software markers measured on the development setup (seconds).
#: Used only as a fallback when the recording carries no usable DI codes; re-measure for every setup.
DEFAULT_DI_OFFSET_S = 0.0775

# --------------------------------------------------------------------------------------------- default config
DEFAULT_CONFIG = {
    # zero-phase Butterworth (SOS) bands; pad = odd extension of pad_s seconds
    "band": [0.5, 45.0],             # delivered X
    "band_broadband": [0.5, 150.0],  # delivered X_broadband (HF mask source, A7-type ratios)
    "band_hp01": [0.1, 45.0],        # internal only (slow-shift QA; not valid for MRCP from single trials)
    "filter_order": 4,
    "pad_s": 3.0,
    # time axis
    "crop_label": "trial_reset",     # marker that ends the trial (samples after it belong to the next trial)
    "trial_start_label": "trial_start",
    "target_prefix": "target_on",    # first marker whose label starts with this = target onset
    "di_offset": "auto",             # 'auto' (measure from DI codes) or seconds
    "epoch": {"pre_s": 0.6, "post_s": 1.4},          # decoder epoch around the DI-corrected reach onset
    "decoder_channels": list(DECODER_17),
    # bridges
    "bridge": {"band": [1.0, 30.0], "thr_uV2": 5.0},
    # ocular regression
    "ocular": {
        "regressors": ["VEOG", "HEOG"],
        "lp_hz": 7.0,                # fit AND subtraction band upper edge. Never above 7 Hz (see README)
        "lp_order": 4,
        "ridge_lambda_frac": 0.01,   # lambda = frac * trace of the (standardised) proxy covariance
        "standardise": True,
        "poles": list(EOG_POLES),
        "fit_spans": None,           # list of [t0, t1] (s, EEG axis) or None = automatic rule below
        "auto_edge_s": 0.2,          # automatic rule: trial crop minus edge_s at both ends ...
        "auto_head_thr_deg_s": 10.0,  # ... minus head speed above this (padded like the motion mask) ...
        "auto_hand_frac": 0.10,      # ... minus hand speed above this fraction of its peak (padded 0.1 s)
        "auto_min_run_s": 0.25,      # drop calibration fragments shorter than this
        "auto_min_s": 4.0,           # warn when fewer seconds remain
    },
    # reference
    "reference": {"car_set": list(ROBUST_CAR20)},
    # rule-based masks (EEG = DI-corrected axis)
    "motion": {
        "head_thr_deg_s": 10.0, "head_pad_before_s": 0.1, "head_pad_after_s": 0.4,
        "hf_band": [55.0, 95.0], "hf_smooth_s": 0.1, "hf_quiet_pct": 25.0, "hf_thr_db": 6.0, "hf_pad_s": 0.1,
        "alpha_band": [7.5, 10.5], "alpha_channels": ["Fz", "FC1", "C3", "Cz", "Pz"], "alpha_thr_x": 3.0,
        "alpha_quiet_pct": 25.0, "alpha_smooth_s": 0.1, "alpha_pad_s": 0.1,
        # 'never' | 'always' | 'if_union_misses_validation_span'
        "alpha_apply": "never",
        "validation_spans": None, "validation_min_coverage": 0.95,
    },
    "transient": {"thr_uV": 100.0, "win_s": 0.25, "hp_hz": 0.5},
    "blink": {"band": [0.5, 15.0], "height_uV": 100.0, "prominence_uV": 80.0, "min_dist_s": 0.15,
              "width_s": [0.03, 0.25], "half_window_s": 0.15},
    # features
    "features": {
        "win_s": 0.5, "step_s": 0.05,
        "bands": {"eog_low": [0.5, 7.0], "mu": [8.0, 13.0], "beta_low": [13.0, 20.0], "beta_high": [20.0, 30.0],
                  "gamma": [30.0, 40.0]},
        "require_fully_valid": True,
    },
    "usability": {
        "exclude_bands": ["eog_low", "gamma"],
        "knife_edge_frac": 0.10,
        "beta_high_exclude_ratio": 2.0, "beta_high_caution_ratio": 3.0,
        "beta_low_exclude_ratio": 1.5, "beta_low_caution_ratio": 3.0,
        "blink_caution_db": 3.0, "posture_caution_db": 6.0,
        "ocular_gain_caution": 0.10,  # |VEOG gain| above this -> blink energy > lp_hz is left -> mu 'caution'
        "blink_in_epoch_caution_db": 1.0,  # blink inside the valid decoder epoch + local blink mu contrast >= this -> mu 'caution'
        "min_independent_windows": 2.0,  # valid epoch time / window length below this -> 'use' is capped at 'caution'
        "policy": {},                # optional {"channel": {"band": "exclude"|"caution"}} overrides
    },
    # optional band-limited ICA cross-check (QA only; never applied to X)
    "ica_check": {"enabled": False, "band": [1.0, 40.0], "n_components": 15, "method": "infomax",
                  "extended": True, "random_state": 97, "max_iter": 2000, "abs_r_thr": 0.6,
                  "blink_subspace_r": 0.4, "alpha_veto_db": 3.0, "max_remove": 3, "lp_hz": 7.0},
}


def merge_config(base: dict | None = None, override: dict | None = None) -> dict:
    """Deep-merge ``override`` into a copy of ``base`` (default :data:`DEFAULT_CONFIG`)."""
    out = copy.deepcopy(DEFAULT_CONFIG if base is None else base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None) -> dict:
    """Default config, optionally overridden by a JSON file."""
    if not path:
        return merge_config()
    with open(path) as fh:
        return merge_config(None, json.load(fh))
