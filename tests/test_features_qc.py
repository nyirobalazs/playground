"""Usability flags and channel QC on synthetic data."""
import numpy as np
import synthetic as S

import eeg_vr_clean as P
from eeg_vr_clean import features as FT
from eeg_vr_clean import qc as QC


def test_band_power_windows_nan_outside_valid():
    fs = 500.0
    t = np.arange(5000) / fs
    X = np.sin(2 * np.pi * 10 * t)[None, :] * 10
    valid = np.ones(len(t), bool)
    valid[(t >= 2.0) & (t < 2.5)] = False
    st = np.array([1.0, 1.8, 2.6])
    F, wv = FT.band_power_windows(X, fs, t, st, [0], {"mu": (8, 13), "beta": (13, 20)}, 0.5, valid)
    assert list(wv) == [True, False, True]
    assert np.isnan(F["mu"][0, 1]) and np.isfinite(F["mu"][0, [0, 2]]).all()
    assert F["mu"][0, 0] > F["beta"][0, 0] + 20                     # a 10 Hz sine lives in the mu band


def test_usability_rules(rec):
    X = rec["X"].copy()
    rng = np.random.default_rng(0)
    i = S.CH.index("CP5")
    X[i] += 15 * S.band_noise(rng, X.shape[1], S.FS, 20.0, 150.0)   # EMG / contact-like broadband floor
    D = P.trial_from_arrays(X, S.FS, S.CH, move_onset=6.0, events={"trial_start": 1.0, "target_on_1": 3.0})
    res = P.run_trial(D)
    st = {(r["channel"], r["band"]): r["status"] for r in res["usability"]}
    assert all(st[("FC6", b)] == "exclude" for b in ("mu", "beta_low", "beta_high"))   # re-estimated channel
    assert all(st[(c, "eog_low")] == "exclude" and st[(c, "gamma")] == "exclude" for c in ("Cz", "C3"))
    assert st[("CP5", "beta_high")] == "exclude"
    assert st[("Cz", "beta_low")] == "use" and st[("Cz", "mu")] == "use"
    assert st[("F3", "mu")] == "caution"                            # large VEOG gain: > 7 Hz blink energy left
    q = QC.channel_qc(D, P.merge_config(), res["bridges"])
    assert any("hf_excess" in f for f in q["channels"]["CP5"]["flags"])
    assert "bridged" in q["channels"]["FC6"]["flags"]


def test_policy_override(rec):
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, move_onset=6.0)
    cfg = P.merge_config(None, {"usability": {"policy": {"C4": {"beta_high": "exclude"}}}})
    res = P.run_trial(D, cfg)
    st = {(r["channel"], r["band"]): r for r in res["usability"]}
    assert st[("C4", "beta_high")]["status"] == "exclude"
    assert st[("C4", "beta_high")]["reason"].startswith("policy override")


def test_ocular_leakage_reduced(rec):
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH)
    res = P.run_trial(D)
    lk = res["qa"]["ocular_leakage"]
    assert lk["cleaned"]["AF_proxies"]["median_abs_bVEOG"] < 0.25 * lk["reference_raw"]["AF_proxies"]["median_abs_bVEOG"]
    assert lk["cleaned"]["brief_proxies"]["frontal_max_abs_bVEOG"] < 0.05


def test_decoder_export_by_name_and_resampled(rec):
    from eeg_vr_clean.pipeline import export_decoder_arrays
    t = rec["t"]
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, move_onset=6.0, events={"trial_reset": 11.0})
    res = P.run_trial(D, qa=False)
    ex = export_decoder_arrays(res, D, channels=["Cz", "C3", "FC6"], fs_out=250.0)
    assert list(ex["ch"]) == ["Cz", "C3", "FC6"] and ex["fs"] == 250.0
    assert ex["X"].shape == (3, len(t) // 2)
    k = S.CH.index("Cz")
    assert np.corrcoef(ex["X"][0, 100:-100], res["X"][k, ::2][100:-100])[0, 1] > 0.99
    assert not ex["valid"][int(11.0 * 250):].any() and ex["valid"][:int(10.9 * 250)].all()
    assert ex["reach_onset_sample"] == int(round(6.0 * 250))
    assert list(ex["reestimated_channels"]) == ["FC6"]


def test_ratios_use_decoder_epoch_windows(rec):
    """Beta/HF ratios come from the fully valid 0.5-s decoder-epoch windows, not from all valid samples."""
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, move_onset=6.0)
    res = P.run_trial(D)
    m = FT.channel_metrics(D, res, P.merge_config())
    assert "decoder-epoch windows" in m["Cz"]["ratio_source"]
    assert m["Cz"]["n_independent_epoch_windows"] >= 2


def test_too_little_epoch_data_caps_use_at_caution(rec):
    """A head turn masking most of the decoder epoch leaves < 2 independent windows: nothing may be 'use'."""
    t = rec["t"]
    head = np.where((t >= 5.2) & (t < 6.7), 30.0, 1.0)          # deg/s; mask pads 0.1 s before / 0.4 s after
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, move_onset=6.0, head_speed=head)
    res = P.run_trial(D)
    rows = [r for r in res["usability"] if r["band"] in ("mu", "beta_low", "beta_high")]
    assert rows and all(r["status"] != "use" for r in rows)
    assert any("too little data to certify" in r["reason"] for r in rows)


def test_blink_inside_epoch_flags_mu(rec):
    """mu becomes 'caution' when a blink lies in the valid epoch and the channel's local blink contrast is >= 1 dB."""
    D = P.trial_from_arrays(rec["X"], S.FS, S.CH, move_onset=6.0)
    res = P.run_trial(D)
    cfg = P.merge_config()
    m = FT.channel_metrics(D, res, cfg)
    m["Cz"].update(blink_in_epoch_frac=0.3, local_blink_mu_db=1.5, veog_gain=0.0)
    m["Pz"].update(blink_in_epoch_frac=0.0, local_blink_mu_db=1.5, veog_gain=0.0)
    st = {(r["channel"], r["band"]): r for r in FT.usability(D, res, cfg, metrics=m)}
    assert st[("Cz", "mu")]["status"] == "caution" and "inside the valid decoder epoch" in st[("Cz", "mu")]["reason"]
    assert st[("Pz", "mu")]["status"] == "use"
