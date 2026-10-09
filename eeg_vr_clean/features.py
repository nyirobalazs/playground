"""Masked log band-power window features and per-channel usability flags.

Masks are ENFORCED here: a window that overlaps any BAD_* span gets NaN features (``require_fully_valid``), so a
downstream decoder cannot silently train on the motion bursts that are still present in X.  In the development
recording, ignoring the masks inflated the epoch mu feature by 3-4 dB relative to fixation.

Usability flags (``use`` / ``caution`` / ``exclude`` per decoder channel x band) are rule-based and computed from the
cleaned output of the trial itself:

* re-estimated (bridged) channels: exclude every band (not an independent measurement);
* ``eog_low`` (0.5-7 Hz) and ``gamma`` (30-40 Hz): exclude everywhere by default - eye-head-coupled activity in the
  movement epoch cannot be certified as ocular-free, the head-posture slow shift is not ocular, and saccadic spike
  potentials / EMG bursts sit above 30 Hz;
* ``mu``: caution where blink energy above the correction band is left (|VEOG gain| >= 0.10, or a local blink
  contrast >= +3 dB);
* ``beta_low`` (13-20 Hz): exclude where 13-20/60-90 Hz < 1.5 (white HF floor), caution below 3, with large local
  blink contrast, or with a posture-locked 55-95 Hz change > 6 dB;
* ``beta_high`` (20-30 Hz): exclude where 20-30/60-90 Hz < 2 on the valid decoder-epoch samples (EMG / contact floor; the same
  ratio is >= 3 on medial channels with genuine beta), caution below 3 or with blink / posture effects;
* values within +-10 % of a threshold are indeterminate: at an exclusion threshold they are excluded
  (conservative), at a caution threshold they get 'caution'.
"""
from __future__ import annotations

import numpy as np

from . import dsp


def window_starts(a, b, win_s, step_s):
    if not (np.isfinite(a) and np.isfinite(b)) or b - a < win_s:
        return np.zeros(0)
    return a + step_s * np.arange(int(round((b - a - win_s) / step_s)) + 1)


def band_power_windows(X, fs, t, starts, channels_idx, bands, win_s=0.5, valid=None, require_fully_valid=True):
    """log10 band power (dB re 1 uV^2) of Hann-windowed segments: 10*log10(mean |FFT|^2 in band), nfft = 2*fs.

    Returns (features {band: (n_ch, n_win)}, window_valid (n_win,)); features of invalid windows are NaN."""
    n = int(round(win_s * fs))
    nfft = int(round(2 * fs))
    w = np.hanning(n)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    out = {b: np.full((len(channels_idx), len(starts)), np.nan) for b in bands}
    wv = np.zeros(len(starts), bool)
    for j, s in enumerate(starts):
        i0 = int(round(s * fs))
        if i0 < 0 or i0 + n > X.shape[1]:
            continue
        ok = True if valid is None else bool(valid[(t >= s) & (t < s + win_s)].all())
        wv[j] = ok
        if require_fully_valid and not ok:
            continue
        seg = X[channels_idx, i0:i0 + n]
        seg = seg - seg.mean(1, keepdims=True)
        Pw = np.abs(np.fft.rfft(seg * w, n=nfft, axis=1)) ** 2
        for b, (lo, hi) in bands.items():
            out[b][:, j] = 10 * np.log10(Pw[:, (f >= lo) & (f < hi)].mean(1))
    return out, wv


def epoch_features(D, res, cfg, baseline=None):
    """Decoder-epoch windows (DI axis) and baseline-relative features.

    ``baseline`` = (t0, t1) on the EEG axis; default = the fixation period between the DI-corrected trial-start and
    target-onset markers.  Returns a dict of arrays plus long-format rows for CSV export."""
    fc = cfg["features"]
    fs, t, ch = D["fs"], D["t"], list(D["ch"])
    chans = [c for c in cfg["decoder_channels"] if c in ch]
    idx = [ch.index(c) for c in chans]
    X = res["X"] - np.median(res["X"], 1, keepdims=True)
    e0, e1 = D["decoder_epoch_di"]
    st = window_starts(e0, e1, fc["win_s"], fc["step_s"])
    F, wv = band_power_windows(X, fs, t, st, idx, fc["bands"], fc["win_s"], res["valid"], fc["require_fully_valid"])
    if baseline is None:
        ev = D.get("events_di", {})
        ts = ev.get(cfg["trial_start_label"])
        tt = next((v for k, v in ev.items() if k.startswith(cfg["target_prefix"])), None)
        baseline = (ts, tt) if ts is not None and tt is not None else None
    base = None
    if baseline is not None:
        sb = window_starts(baseline[0], baseline[1], fc["win_s"], fc["step_s"])
        Fb, vb = band_power_windows(X, fs, t, sb, idx, fc["bands"], fc["win_s"], res["valid"], True)
        if vb.any():
            base = {b: np.nanmean(Fb[b][:, vb], 1) for b in fc["bands"]}
        baseline_info = dict(span_s=[float(baseline[0]), float(baseline[1])], n_windows=int(len(sb)),
                             n_valid=int(vb.sum()))
    else:
        baseline_info = None
    rows = []
    for j, s in enumerate(st):
        for b in fc["bands"]:
            for k, c in enumerate(chans):
                v = F[b][k, j]
                rows.append(dict(window_start_s=round(float(s), 4), window_end_s=round(float(s + fc["win_s"]), 4),
                                 window_valid=bool(wv[j]), channel=c, band=b,
                                 power_db=None if not np.isfinite(v) else round(float(v), 4),
                                 rel_baseline_db=None if (base is None or not np.isfinite(v))
                                 else round(float(v - base[b][k]), 4)))
    return dict(starts=st, window_valid=wv, features=F, channels=chans, baseline=baseline_info,
                baseline_db=base, n_windows=int(len(st)), n_valid_windows=int(wv.sum()), rows=rows)


# ------------------------------------------------------------------------------------------------ usability
def local_blink_contrast(X, D, blinks, lo, hi, cfg):
    """Band power within +-150 ms of each blink vs 0.35-0.65 s either side (other blinks excluded), dB per channel."""
    fs, t = D["fs"], D["t"]
    F = dsp.bandpass(X - np.median(X, 1, keepdims=True), fs, lo, hi, order=cfg["filter_order"], pad_s=cfg["pad_s"]) ** 2
    win = lambda a, b: (t >= a) & (t < b)  # noqa: E731
    bl = np.zeros(len(t), bool)
    for b in blinks:
        bl |= win(b - 0.15, b + 0.15)
    pin, pout = [], []
    for b in blinks:
        mo = (win(b - 0.65, b - 0.35) | win(b + 0.35, b + 0.65)) & ~bl
        if mo.sum() < 0.1 * fs:
            continue
        pin.append(F[:, win(b - 0.15, b + 0.15)].mean(1))
        pout.append(F[:, mo].mean(1))
    if not pin:
        return np.zeros(X.shape[0])
    return 10 * np.log10(np.mean(pin, 0) / np.mean(pout, 0))


def channel_metrics(D, res, cfg):
    """Per decoder channel numbers used by the usability rules (all from this trial's cleaned output)."""
    from scipy.signal import welch
    fs, t, ch, nc = D["fs"], D["t"], list(D["ch"]), int(D["n_crop"])
    X = res["X"] - np.median(res["X"], 1, keepdims=True)
    Xbb = res["X_broadband"] - np.median(res["X_broadband"], 1, keepdims=True)
    valid = res["valid"] & (np.arange(len(t)) < nc)
    hi = min(90.0, 0.45 * fs)
    f, Pb = welch(Xbb[:, :nc], fs, nperseg=int(round(2 * fs)), detrend="linear")
    fv, Pv = dsp.welch_runs(Xbb, valid, fs)
    loc = {bn: local_blink_contrast(X, D, res["blinks"], lo, h, cfg)
           for bn, (lo, h) in (("mu", (8, 13)), ("b13_20", (13, 20)), ("b20_30", (20, 30)))}
    ev = D.get("events_di", {})
    fix = (ev.get(cfg["trial_start_label"]), next((v for k, v in ev.items() if k.startswith(cfg["target_prefix"])), None))
    mov = (D.get("move_onset_di", np.nan), D["decoder_epoch_di"][1])

    # Beta/HF ratios are evaluated where AND how the decoder features are computed: the same fully valid 0.5-s Hann
    # windows of the decoder epoch (band_power_windows). Trial-wide valid samples are dominated by rest/fixation, where
    # EMG and contact noise differ from the head-turned, arm-moving posture of the epoch; on the development trial the
    # epoch ratios were 1.1-9.4x lower. On sub-second spans the ratio also depends strongly on the estimator (2-6x
    # between Hann windows, filter-based and multitaper estimates), so the feature estimator itself is used.
    fc = cfg["features"]
    ep_mask = (t >= D["decoder_epoch_di"][0]) & (t < D["decoder_epoch_di"][1]) & valid
    hf_hi = min(90.0, 0.45 * fs)
    rb = {"b13_20": (13.0, 20.0), "b20_30": (20.0, 30.0), "hf": (60.0, hf_hi)}
    dec_idx = [ch.index(c) for c in cfg["decoder_channels"] if c in ch]
    st_ep = window_starts(D["decoder_epoch_di"][0], D["decoder_epoch_di"][1], fc["win_s"], fc["step_s"])
    Fw, wv_ep = band_power_windows(Xbb, fs, t, st_ep, dec_idx, rb, fc["win_s"], valid, True)
    ep_ok = bool(wv_ep.any())
    lin = {b: (np.nanmean(10 ** (Fw[b][:, wv_ep] / 10), 1) if ep_ok else None) for b in rb}
    # independent evidence: valid epoch time / window length (overlapping windows are not independent)
    n_indep = float(ep_mask.sum() / fs / fc["win_s"])

    bl_ep = np.zeros(len(t), bool)
    for b in res["blinks"]:
        bl_ep |= (t >= b - 0.15) & (t < b + 0.15)
    blink_in_epoch_frac = float((bl_ep & ep_mask).sum() / max(ep_mask.sum(), 1))

    def posture(i, lo, h):
        if fix[0] is None or fix[1] is None or not np.isfinite(mov[0]):
            return float("nan")
        F = dsp.bandpass(Xbb[i], fs, lo, h, order=cfg["filter_order"], pad_s=cfg["pad_s"])
        a = (t >= mov[0]) & (t < mov[1]) & valid
        b = (t >= fix[0]) & (t < fix[1]) & valid
        if a.sum() < 0.25 * fs or b.sum() < 0.25 * fs:
            return float("nan")
        return float(10 * np.log10((F[a] ** 2).mean() / (F[b] ** 2).mean()))

    gV = None
    if "model" in res and "VEOG" in res["model"]["regressors"]:
        gV = res["model"]["B"][res["model"]["regressors"].index("VEOG")]
    out = {}
    for c in cfg["decoder_channels"]:
        if c not in ch:
            continue
        i = ch.index(c)
        bm = lambda P, ff, lo, h: P[(ff >= lo) & (ff <= h)].mean()  # noqa: E731
        whole = float(bm(Pb[i], f, 20, 30) / bm(Pb[i], f, 60, hi))
        if fv is not None:
            r2030t = float(bm(Pv[i], fv, 20, 30) / bm(Pv[i], fv, 60, hi))
            r1320t = float(bm(Pv[i], fv, 13, 20) / bm(Pv[i], fv, 60, hi))
        else:
            r2030t = whole
            r1320t = float(bm(Pb[i], f, 13, 20) / bm(Pb[i], f, 60, hi))
        if ep_ok:
            k = dec_idx.index(i)
            r2030v, r1320v = float(lin["b20_30"][k] / lin["hf"][k]), float(lin["b13_20"][k] / lin["hf"][k])
            src = f"{int(wv_ep.sum())} valid 0.5-s decoder-epoch windows"
        else:
            r2030v, r1320v, src = r2030t, r1320t, "whole-trial valid samples (decoder epoch has too little valid data)"
        out[c] = dict(ratio_20_30_over_60_90_valid=r2030v, ratio_20_30_over_60_90_wholecrop=whole,
                      ratio_13_20_over_60_90=r1320v, ratio_source=src,
                      ratio_20_30_over_60_90_trial_valid=r2030t, ratio_13_20_over_60_90_trial_valid=r1320t,
                      blink_in_epoch_frac=blink_in_epoch_frac, n_independent_epoch_windows=n_indep,
                      local_blink_mu_db=float(loc["mu"][i]),
                      local_blink_b13_20_db=float(loc["b13_20"][i]), local_blink_b20_30_db=float(loc["b20_30"][i]),
                      posture_55_95_db=posture(i, 55.0, min(95.0, 0.45 * fs)),
                      veog_gain=None if gV is None else float(gV[i]),
                      reestimated=c in res.get("drop", []))
    return out


def usability(D, res, cfg, metrics=None):
    """Rows of (channel, band, status, reason)."""
    uc = cfg["usability"]
    m_all = channel_metrics(D, res, cfg) if metrics is None else metrics
    edge = uc["knife_edge_frac"]
    near = lambda v, thr: np.isfinite(v) and abs(v - thr) <= edge * abs(thr)  # noqa: E731
    rows = []
    for c, m in m_all.items():
        pol = uc.get("policy", {}).get(c, {})
        for band, (lo, hi) in cfg["features"]["bands"].items():
            if m["reestimated"]:
                st, why = "exclude", "spherical-spline re-estimate of a bridged site: not an independent channel"
            elif band in uc["exclude_bands"]:
                st, why = "exclude", {"eog_low": "0.5-7 Hz: ocular / eye-head-coupled residual cannot be certified; "
                                                 "head-posture slow shift is not ocular",
                                      "gamma": "30-40 Hz: saccadic spike potentials, HF/EMG bursts, posture-locked "
                                               "contact noise are not removed"}.get(band, "excluded by configuration")
            elif band == "mu":
                lb, g = m["local_blink_mu_db"], m["veog_gain"]
                hit = []
                if lb >= uc["blink_caution_db"] or near(lb, uc["blink_caution_db"]):
                    hit.append(f"local blink contrast {lb:+.1f} dB")
                if g is not None and abs(g) >= uc["ocular_gain_caution"]:
                    hit.append(f"|VEOG gain| {abs(g):.2f}: blink energy above the correction band is left")
                if m["blink_in_epoch_frac"] > 0 and lb >= uc["blink_in_epoch_caution_db"]:
                    hit.append(f"a blink lies inside the valid decoder epoch ({100 * m['blink_in_epoch_frac']:.0f}% of "
                               f"its samples) and this channel's local blink mu contrast is {lb:+.1f} dB")
                st = "caution" if hit else "use"
                why = "; ".join(hit) if hit else f"local blink contrast {lb:+.1f} dB"
            elif band == "beta_low":
                r, lb, pz = m["ratio_13_20_over_60_90"], m["local_blink_b13_20_db"], m["posture_55_95_db"]
                if r < uc["beta_low_exclude_ratio"] or near(r, uc["beta_low_exclude_ratio"]):
                    st, why = "exclude", f"13-20/60-90 Hz {r:.2f} ({m['ratio_source']}): low beta at (or indeterminably near) the HF floor"
                elif (r < uc["beta_low_caution_ratio"] or near(r, uc["beta_low_caution_ratio"])
                      or lb >= uc["blink_caution_db"]
                      or (np.isfinite(pz) and abs(pz) > uc["posture_caution_db"])):
                    st, why = "caution", f"13-20/60-90 Hz {r:.2f}; local blink {lb:+.1f} dB; 55-95 Hz posture {pz:+.1f} dB"
                else:
                    st, why = "use", f"13-20/60-90 Hz {r:.2f}; 55-95 Hz posture {pz:+.1f} dB"
            elif band == "beta_high":
                r = m["ratio_20_30_over_60_90_valid"]
                lb, pz = m["local_blink_b20_30_db"], m["posture_55_95_db"]
                if r < uc["beta_high_exclude_ratio"] or near(r, uc["beta_high_exclude_ratio"]):
                    st, why = ("exclude", f"20-30/60-90 Hz {r:.2f} on {m['ratio_source']} (whole trial "
                                          f"{m['ratio_20_30_over_60_90_wholecrop']:.2f}): EMG / contact floor")
                elif (r < uc["beta_high_caution_ratio"] or near(r, uc["beta_high_caution_ratio"])
                      or lb >= uc["blink_caution_db"]
                      or (np.isfinite(pz) and abs(pz) > uc["posture_caution_db"])):
                    st, why = "caution", f"20-30/60-90 Hz {r:.2f}; local blink {lb:+.1f} dB; 55-95 Hz posture {pz:+.1f} dB"
                else:
                    st, why = "use", f"20-30/60-90 Hz {r:.2f}"
            else:
                st, why = "use", "no rule for this band"
            nmin = uc["min_independent_windows"]
            if st == "use" and m["n_independent_epoch_windows"] < nmin:
                st, why = "caution", (f"{why}; only {m['n_independent_epoch_windows']:.1f} independent valid 0.5-s "
                                      f"windows in the decoder epoch (< {nmin}): too little data to certify")
            if band in pol:
                st, why = pol[band], f"policy override ({why})"
            rows.append(dict(channel=c, band=band, band_hz=[lo, hi], status=st, reason=why))
    return rows
