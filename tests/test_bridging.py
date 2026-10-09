"""Gel-bridge detection by electrical distance, keeper rule and spline re-estimation."""
import numpy as np
import pytest
import synthetic as S

import eeg_vr_clean as P
from eeg_vr_clean import bridging as BR
from eeg_vr_clean import dsp


def _trial(bridge, seed=3):
    rec = S.make_recording(seed=seed, bridge=bridge)
    return P.trial_from_arrays(rec["X"], S.FS, S.CH)


def test_bridged_pair_detected_and_duplicate_dropped():
    br = BR.detect_bridges(_trial(("F4", "FC6")))
    assert [(d["a"], d["b"]) for d in br["bridged"]] == [("F4", "FC6")]
    assert br["bridged"][0]["ed_uV2"] < 1.0
    assert br["drop"] == ["FC6"]                         # both decoder channels: F4 wins as robust-CAR member
    assert br["components"][0]["rule_applied"] == "robust-CAR member"
    assert br["gap_ratio_next_pair"] > 10


def test_keeper_rule_prefers_car_member():
    br = BR.detect_bridges(_trial(("PO7", "PO3")))
    assert br["drop"] == ["PO3"]


def test_no_false_bridges():
    br = BR.detect_bridges(_trial(None))
    assert br["bridged"] == [] and br["drop"] == []
    assert br["pairs_lowest"][0]["ed_uV2"] > 5.0


def test_reestimated_channel_is_spline_of_sources_not_duplicate():
    D = _trial(("F4", "FC6"))
    res = P.run_trial(D, qa=False)
    X = res["X"]
    i4, i6 = S.CH.index("F4"), S.CH.index("FC6")
    ed = np.var(dsp.bandpass(X[i4] - X[i6], S.FS, 1, 30))
    assert ed > 5.0                                          # no longer a duplicate of F4
    si, w, src = res["interp"]["FC6"]
    assert "F4" in src and "FP1" not in src and "F8" not in src   # uncorrected EOG poles never feed a re-estimate
    assert w.sum() == pytest.approx(1.0, abs=0.05)
    assert res["rank"] == 32 - 1 - 1


def test_session_bridges_median_ed():
    trials = [_trial(("F4", "FC6"), seed=s) for s in (4, 5)]
    br = BR.session_bridges(trials)
    assert br["drop"] == ["FC6"] and br["n_trials"] == 2
    assert len(br["per_trial_ed_of_bridged_uV2"]["F4-FC6"]) == 2
