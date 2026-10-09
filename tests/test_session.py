"""Session mode: one bridge decision and one frozen gain model for several trials; calibration-file round trip."""
import numpy as np
import pytest
import synthetic as S

import eeg_vr_clean as P
from eeg_vr_clean import ocular as OC


def _trials(seeds, bridge=("F4", "FC6")):
    out = []
    for k, s in enumerate(seeds):
        blinks = list(np.sort(np.random.default_rng(s).uniform(0.5, 11.5, 12)))
        rec = S.make_recording(seed=s, blinks=blinks, bridge=bridge)
        out.append(P.trial_from_arrays(rec["X"], S.FS, S.CH))
    return out


@pytest.fixture(scope="module")
def session():
    trials = _trials([11, 12, 13])
    return trials, P.run_session(trials)


def test_session_fits_one_model_on_concatenated_calibration(session):
    trials, ses = session
    m = ses["model"]
    assert len(m["calibration"]) == 3
    assert m["n_fit_s"] == pytest.approx(sum(c["n_samples"] for c in m["calibration"]) / S.FS)
    assert m["n_fit_s"] > 3 * 10.0
    assert ses["summary"]["drop"] == ["FC6"]
    car = [S.CH.index(c) for c in m["car_set"]]
    g = np.array([S.vgain(c) for c in S.CH])
    true = (g - g[car].mean()) / 0.98
    B = m["B"][m["regressors"].index("VEOG")]
    for c in ("F3", "Fz", "F4"):
        assert B[S.CH.index(c)] == pytest.approx(true[S.CH.index(c)], abs=0.02)


def test_session_applies_frozen_model_per_trial(session):
    trials, ses = session
    for D, r in zip(trials, ses["results"]):
        assert r["model"] is ses["model"]
        solo = P.run_trial(D, model=ses["model"], bridges=ses["bridges"], qa=False)
        assert np.array_equal(solo["X"], r["X"])
        assert r["X"].shape == D["X"].shape


def test_calibration_file_round_trip(session, tmp_path):
    trials, ses = session
    path = OC.save_calibration(ses["model"], str(tmp_path / "gains.json"))
    m2 = OC.load_calibration(path, S.CH)
    assert np.allclose(m2["B"], ses["model"]["B"])
    ses2 = P.run_session(trials, calibration=path)
    for a, b in zip(ses["results"], ses2["results"]):
        assert np.max(np.abs(a["X"] - b["X"])) < 1e-9


def test_dedicated_calibration_recording():
    trials = _trials([21, 22])
    calib = _trials([23])
    ses = P.run_session(trials, calib_trials=calib)
    assert ses["model"]["fitted_on"] == "calibration recordings"
    assert len(ses["model"]["calibration"]) == 1
    # gains from another recording still remove the low-band ocular artefact of the session trials
    for r in ses["results"]:
        assert r["qa"]["power_ratio_vs_reference_raw"]["1-7"]["median"] < 0.8
        assert r["qa"]["power_ratio_vs_reference_raw"]["20-30"]["median"] == pytest.approx(1.0, abs=1e-3)


def test_channel_mismatch_rejected():
    a = _trials([31])[0]
    rec = S.make_recording(seed=32)
    b = P.trial_from_arrays(rec["X"][::-1], S.FS, S.CH[::-1])
    with pytest.raises(ValueError):
        P.run_session([a, b])
