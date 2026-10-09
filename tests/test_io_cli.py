"""Loader (HDF5 + Unity JSON, DI alignment, Unity-clock head kinematics) and the command line interface, on
synthetic files written to a temporary directory."""
import json
import os

import numpy as np
import pytest
import synthetic as S

import eeg_vr_clean as P
from eeg_vr_clean import cli


def test_load_trial_di_alignment_and_kinematics(files):
    D = P.load_trial(files["eeg"], files["vr"])
    assert D["X"].shape == (32, files["rec"]["X"].shape[1]) and D["ch"] == S.CH
    assert np.allclose(D["X"], files["rec"]["X"].astype(np.float32), atol=1e-3)
    rep = D["di_report"]
    assert rep["n"] == 2 and not rep["fallback_used"]
    assert D["di_offset"] == pytest.approx(files["di_lag"], abs=1.5 / S.FS)
    assert any(e["code"] == 70 for e in rep["excluded"])          # queued pulse is not used
    assert D["n_crop"] == int(files["crop_s"] * S.FS)
    a, b, _ = files["turn"]
    hs = D["head_speed_unity"]
    assert np.nanmax(hs) == pytest.approx(files["peak_deg_s"], rel=0.05)
    assert D["t"][np.nanargmax(hs)] == pytest.approx((a + b) / 2, abs=0.03)    # host axis
    assert D["move_onset"] == pytest.approx(5.3, abs=0.1)
    assert D["move_onset_di"] - D["move_onset"] == pytest.approx(D["di_offset"])
    assert D["decoder_epoch_di"][1] - D["decoder_epoch_di"][0] == pytest.approx(2.0)
    assert D["touch_time"] == pytest.approx(3.0 + 2.6)


def test_fixed_di_offset_override(files):
    D = P.load_trial(files["eeg"], files["vr"], di_offset=0.05)
    assert D["di_offset"] == 0.05
    assert D["di_report"]["measured_offset_s"] == pytest.approx(files["di_lag"], abs=1.5 / S.FS)


def test_load_without_vr(files):
    D = P.load_trial(files["eeg"])
    assert np.isnan(D["move_onset"]) and np.all(np.isnan(D["head_speed_unity"]))
    res = P.run_trial(D)
    assert res["features"]["n_windows"] == 0 and res["qa"]["usable_epoch_s"] is None


def test_cli_run_outputs(files, tmp_path):
    out = str(tmp_path / "run")
    assert cli.main(["run", "--eeg", files["eeg"], "--vr", files["vr"], "--out", out]) == 0
    for f in ("cleaned.npz", "cleaned_raw.fif", "info.json", "gains.json", "features.csv", "feature_usability.csv"):
        assert os.path.exists(os.path.join(out, f)), f
    figs = sorted(os.listdir(os.path.join(out, "figures")))
    assert len(figs) == 7 and all(f.endswith(".png") for f in figs)
    z = np.load(os.path.join(out, "cleaned.npz"))
    for k in ("X", "fs", "ch", "ann_onset", "ann_duration", "ann_desc", "ch_types", "valid", "X_broadband"):
        assert k in z.files
    assert z["X"].shape == (32, files["rec"]["X"].shape[1]) and np.isfinite(z["X"]).all()
    info = json.load(open(os.path.join(out, "info.json")))
    assert info["timing"]["di_offset_s"] == pytest.approx(files["di_lag"], abs=1.5 / S.FS)
    assert "BAD_next_trial" in [a["desc"] for a in info["annotations"]]
    assert info["dropped_and_reestimated"] == ["FC6"]
    # the head turn (+DI, padded) is masked
    t = np.arange(z["X"].shape[1]) / S.FS
    assert not z["valid"][(t >= 5.2) & (t < 5.6)].any()


def test_cli_calibrate_then_run_with_calibration(files, tmp_path):
    gains = str(tmp_path / "gains.json")
    assert cli.main(["calibrate", "--eeg", files["eeg"], "--vr", files["vr"], "--out", gains,
                     "--fit-spans", "0.5-4.5,8.0-11.5"]) == 0
    g = json.load(open(gains))
    assert g["format"] == "eeg_vr_clean.ocular_gains/1" and g["drop"] == ["FC6"]
    out = str(tmp_path / "run_cal")
    assert cli.main(["run", "--eeg", files["eeg"], "--vr", files["vr"], "--out", out, "--calibration", gains,
                     "--no-figs"]) == 0
    info = json.load(open(os.path.join(out, "info.json")))
    assert info["ocular_model"]["source"] == gains
    assert info["ocular_model"]["calibration"][0]["spans_s"] == [[0.5, 4.5], [8.0, 11.5]]


def test_cli_session_and_qc(tmp_path):
    a = S.write_files(str(tmp_path), seed=41)
    d2 = tmp_path / "second"
    d2.mkdir()
    b = S.write_files(str(d2), seed=42)
    out = str(tmp_path / "ses")
    assert cli.main(["session", "--eeg", a["eeg"], b["eeg"], "--vr", a["vr"], b["vr"], "--out", out, "--no-figs"]) == 0
    summ = json.load(open(os.path.join(out, "session_summary.json")))
    assert summ["n_trials"] == 2 and summ["drop"] == ["FC6"]
    assert os.path.exists(os.path.join(out, "session_gains.json"))
    assert len([d for d in os.listdir(out) if d.startswith("00")]) == 2
    qo = str(tmp_path / "qc")
    assert cli.main(["qc", "--eeg", a["eeg"], "--out", qo]) == 0
    q = json.load(open(os.path.join(qo, "qc.json")))
    assert [(d["a"], d["b"]) for d in q["bridges"]["bridged"]] == [("F4", "FC6")]
