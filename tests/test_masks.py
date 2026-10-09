"""Rule-based masks: head-speed rule on the DI-corrected axis, HF envelope rule, transients, next-trial crop,
and enforcement of the masks in the feature extractor."""
import numpy as np
import pytest
import synthetic as S

import eeg_vr_clean as P
from eeg_vr_clean import masks as MK


def _head_speed(t, a, b, peak=40.0):
    """Raised-cosine head-speed bump on the host axis from a to b."""
    x = np.clip((t - a) / (b - a), 0, 1)
    return peak * 0.5 * (1 - np.cos(2 * np.pi * x))


def test_head_speed_mask_on_di_axis():
    rec = S.make_recording(seed=5, bridge=None)
    t = rec["t"]
    hs = _head_speed(t, 5.0, 5.6)
    di = 0.08
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, head_speed=hs, di_offset=di)
    cfg = P.merge_config()
    m, raw_m, hs_eeg = MK.head_speed_mask(D, cfg["motion"])
    above = t[hs > 10.0]                                  # host axis
    a0, a1 = above[0] + di, above[-1] + di                 # expected supra-threshold span on the EEG axis
    on = t[raw_m]
    assert on[0] == pytest.approx(a0, abs=2 / S.FS) and on[-1] == pytest.approx(a1, abs=2 / S.FS)
    pad = t[m]
    assert pad[0] == pytest.approx(a0 - 0.1, abs=2 / S.FS)
    assert pad[-1] == pytest.approx(a1 + 0.4, abs=2 / S.FS)
    assert np.nanargmax(hs_eeg) == pytest.approx(np.argmax(hs) + di * S.FS, abs=1)


def test_motion_mask_union_in_pipeline():
    rec = S.make_recording(seed=6, bridge=None, hf_burst=(8.0, 8.5))
    t = rec["t"]
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, head_speed=_head_speed(t, 5.0, 5.6), di_offset=0.08)
    res = P.run_trial(D, qa=False)
    spans = res["motion"]["spans"]["BAD_motion"]
    valid = res["valid"]
    assert not valid[(t >= 5.2) & (t < 5.9)].any()       # head turn (+DI, padded)
    assert not valid[(t >= 8.0) & (t < 8.5)].any()       # HF burst
    hf = [s for s in res["motion"]["spans"]["hf_padded"] if s[1] > 7.5]
    assert hf and hf[0][0] > 7.7 and hf[-1][1] < 8.8
    assert valid[(t >= 1.0) & (t < 4.5)].mean() > 0.9     # quiet data stays usable
    assert all(d.startswith(("BAD_", "EYE_")) for d in res["ann"][2])
    assert spans


def test_transient_window_flagged():
    rec = S.make_recording(seed=7, bridge=None)
    X = rec["X"].copy()
    i = S.CH.index("C3")
    k = int(2.1 * S.FS)
    X[i, k:k + 10] += 400.0                               # electrode pop on a decoder channel (not ocular)
    D = P.trial_from_arrays(X, S.FS, S.CH)
    res = P.run_trial(D, qa=False)
    assert any(abs(a - 2.0) < 1e-9 for a, _ in res["transient"])
    assert not res["valid"][(rec["t"] >= 2.0) & (rec["t"] < 2.25)].any()


def test_next_trial_cropped():
    rec = S.make_recording(seed=8, bridge=None)
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, events={"trial_reset": 11.0})
    assert D["n_crop"] == int(11.0 * S.FS)
    res = P.run_trial(D, qa=False)
    on, du, de = res["ann"]
    k = list(de).index("BAD_next_trial")
    assert on[k] == pytest.approx(11.0)
    assert not res["valid"][int(11.0 * S.FS):].any()


def test_blinks_flagged_not_masked(rec):
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH)
    res = P.run_trial(D, qa=False)
    assert np.allclose(np.sort(res["blinks"]), rec["blinks"], atol=0.01)
    assert res["valid"].all()                             # EYE_blink is informational only


def test_features_enforce_masks():
    rec = S.make_recording(seed=9, bridge=None)
    t = rec["t"]
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, head_speed=_head_speed(t, 5.0, 5.6), di_offset=0.08,
                            move_onset=5.5, events={"trial_start": 1.0, "target_on_1": 3.0})
    res = P.run_trial(D)
    ft = res["features"]
    assert ft["n_windows"] == 31
    wv = ft["window_valid"]
    assert 0 < wv.sum() < len(wv)
    for b, F in ft["features"].items():
        assert np.isnan(F[:, ~wv]).all(), b
        assert np.isfinite(F[:, wv]).all(), b
    v = res["valid"]
    for s, ok in zip(ft["starts"], wv):
        assert ok == bool(v[(t >= s) & (t < s + 0.5)].all())
    assert ft["baseline"]["n_valid"] > 0
    rows = [r for r in ft["rows"] if r["window_valid"]]
    assert all(r["rel_baseline_db"] is not None for r in rows)
