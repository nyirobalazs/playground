"""Band-limited EOG regression on synthetic data with injected blinks and saccades / gaze offsets."""
import numpy as np
import pytest
import synthetic as S
from scipy.signal import welch

import eeg_vr_clean as P
from eeg_vr_clean import dsp
from eeg_vr_clean import pipeline as PL


@pytest.fixture(scope="module")
def cleaned(rec):
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH)
    res = P.run_trial(D)
    D0 = P.trial_from_arrays(rec["X0"], S.FS, S.CH)
    R = PL.reference_raw(D, res["cfg"], res["car_set"])          # artefacted, same band / reference, no correction
    R0 = PL.reference_raw(D0, res["cfg"], res["car_set"])        # artefact-free ground truth, same processing
    dec = [c for c in res["cfg"]["decoder_channels"] if c not in res["drop"]]
    return dict(D=D, res=res, R=R, R0=R0, idx=[S.CH.index(c) for c in dec], dec=dec)


def test_blinks_and_saccades_removed_below_7hz(cleaned):
    X, R, R0, idx = cleaned["res"]["X"], cleaned["R"], cleaned["R0"], cleaned["idx"]
    resid = dsp.bandpass(X - R0, S.FS, 1.0, 7.0)[idx].var(1)       # what is left of the artefact (+ removed EEG)
    art = dsp.bandpass(R - R0, S.FS, 1.0, 7.0)[idx].var(1)         # the injected artefact after the robust CAR
    big = art > 5.0                                                # channels the artefact actually reaches
    assert big.sum() >= 10
    ratio = resid[big] / art[big]
    assert np.median(ratio) < 0.05, ratio
    assert ratio.max() < 0.15, ratio
    frontal = [k for k, i in enumerate(idx) if S.CH[i] in ("F3", "Fz", "F4")]
    assert (resid[frontal] / art[frontal]).max() < 0.03


def test_13_40_hz_untouched(cleaned):
    X, R, idx = cleaned["res"]["X"], cleaned["R"], cleaned["idx"]
    f, Px = welch(X[idx], S.FS, nperseg=500)
    Pr = welch(R[idx], S.FS, nperseg=500)[1]
    Pm = welch((R - X)[idx], S.FS, nperseg=500)[1]
    m = (f >= 13) & (f <= 40)
    ratio = Px[:, m].sum(1) / Pr[:, m].sum(1)
    assert np.all(np.abs(ratio - 1) < 1e-3), ratio
    assert (Pm[:, m].sum(1) / Pr[:, m].sum(1)).max() < 1e-4


def test_mu_beta_power_preserved(cleaned):
    X, R0, idx = cleaned["res"]["X"], cleaned["R0"], cleaned["idx"]
    f, Px = welch(X[idx], S.FS, nperseg=500)
    P0 = welch(R0[idx], S.FS, nperseg=500)[1]
    m = (f >= 8) & (f <= 13)
    # 8-13 Hz holds the transition-band remainder of the blinks (left on purpose) - never less than ground truth
    assert np.median(Px[:, m].sum(1) / P0[:, m].sum(1)) == pytest.approx(1.0, abs=0.1)


def test_gains_recover_true_propagation(cleaned):
    res = cleaned["res"]
    car = [S.CH.index(c) for c in res["car_set"]]
    g = np.array([S.vgain(c) for c in S.CH])
    true = (g - g[car].mean()) / (1.0 - 0.02)                      # VEOG proxy = (1 - posterior gain) x blink
    B = res["model"]["B"][res["model"]["regressors"].index("VEOG")]
    for c in ("F3", "Fz", "F4", "FC1", "Pz"):
        assert B[S.CH.index(c)] == pytest.approx(true[S.CH.index(c)], abs=0.03), c


def test_poles_uncorrected_and_typed_eog(cleaned):
    res = cleaned["res"]
    B = res["model"]["B"]
    for c in ("FP1", "FP2", "F7", "F8"):
        i = S.CH.index(c)
        assert np.all(B[:, i] == 0)
        assert res["ch_types"][i] == "eog"
    assert sum(res["ch_types"] == "eog") == 4


def test_deterministic_and_rank(cleaned, rec):
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH)
    again = P.run_trial(D, qa=False)
    assert np.array_equal(again["X"], cleaned["res"]["X"])
    assert cleaned["res"]["rank"] == len(S.CH) - 1 - len(cleaned["res"]["drop"])


def test_explicit_fit_spans_used(rec):
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH)
    res = P.run_trial(D, fit_spans=[[0.5, 4.0], [6.0, 11.0]], qa=False)
    cal = res["model"]["calibration"][0]
    assert cal["spans_s"] == [[0.5, 4.0], [6.0, 11.0]]
    assert cal["n_samples"] == pytest.approx(8.5 * S.FS, abs=2)


def test_ica_crosscheck_band_limited(rec):
    """Optional QA path: ICA finds the blink component and, being band-limited, leaves 13-40 Hz alone."""
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH)
    cfg = P.merge_config(None, {"ica_check": {"n_components": 10, "max_iter": 400}})
    res = P.run_trial(D, cfg, ica_check=True, qa=False)
    ic = res["ica_check"]
    assert ic["removed_ics"], "no ocular IC selected"
    assert max(abs(s["r_veog"]) for s in ic["removed_ics"]) > 0.8
    r = ic["ica_over_regression_power_dec_median"]
    for b in ("13-20", "20-30", "30-40"):
        assert abs(r[b] - 1) < 2e-3, (b, r[b])
    assert ic["X_ica"].shape == res["X"].shape
