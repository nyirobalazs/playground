"""eeg_vr_clean - artefact handling for EEG recorded during VR reaching (HMD + wireless EEG cap).

Band-limited EOG regression, robust CAR, gel-bridge handling, rule-based motion / transient masks on the
DI-corrected EEG axis, masked band-power features with usability flags, and a fixed before/after figure set.

Typical use::

    from eeg_vr_clean import load_trial, run_trial, save_trial_outputs
    D = load_trial("trial.hdf5", "trial_vr.json")
    res = run_trial(D)
    save_trial_outputs(res, D, "out/trial")

or ``python -m eeg_vr_clean run --eeg trial.hdf5 --vr trial_vr.json --out out/trial``.
"""
__version__ = "1.0.0"

from .config import DEFAULT_CONFIG, load_config, merge_config  # noqa: E402,F401
from .io import load_trial, trial_from_arrays  # noqa: E402,F401
from .ocular import fit_gains, load_calibration, save_calibration  # noqa: E402,F401
from .pipeline import run_session, run_trial, save_trial_outputs  # noqa: E402,F401
