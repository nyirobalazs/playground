"""Channel QC and cleaning QA: electrical distance, amplitude, high-frequency floor, line noise, ocular leakage,
band-power preservation and output rank.

All thresholds are relative to the montage median (quiet-anchored where possible) because single-trial absolute
levels are not comparable across sessions.  Note that native-reference amplitude z-scores are misleading on a cap
referenced to one ear/mastoid: channels next to the reference (T8, P8, PO8 on a right-ear reference) look 'dead'
but are normal after re-referencing, so amplitude is judged after the robust CAR.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from . import dsp
from .bridging import detrended, electrical_distance
from .config import HEOG_NEG, HEOG_POS, HOLDOUT, VEOG_NEG, VEOG_POS
from .reference import car_set, robust_car


def _psd(Y, fs, nc):
    return welch(Y[:, :nc], fs, nperseg=int(round(2 * fs)), detrend="linear")


def channel_qc(D, cfg=None, bridges=None, line_hz=50.0):
    """Per-channel table and flags (native and robust-CAR amplitudes, HF floor, 20-30/60-90 Hz ratio, line noise)."""
    cfg = cfg or {}
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    Zd, p = detrended(D)
    sel = car_set(ch, (cfg.get("reference") or {}).get("car_set"), (bridges or {}).get("drop", ()))
    hi = min(150.0, 0.45 * fs)
    Xn = dsp.bandpass(Zd, fs, 1.0, 40.0)[:, :nc]
    Xr = robust_car(dsp.bandpass(Zd, fs, 0.5, hi), ch, sel)
    f, P = _psd(Xr, fs, nc)
    line = np.zeros(len(f), bool)
    for k in range(1, 4):
        line |= np.abs(f - k * line_hz) <= 2.0
    bp = lambda lo, h: P[:, (f >= lo) & (f <= h) & ~line].mean(1)  # noqa: E731
    hf = bp(30.0, hi)
    hf_rel = hf / np.median(hf)
    r2030 = bp(20, 30) / bp(60, min(90, hi))
    lb = np.argmin(np.abs(f - line_hz))
    fl = (np.abs(f - line_hz) >= 1.0) & (np.abs(f - line_hz) <= 3.0)
    line_ratio = P[:, lb] / P[:, fl].mean(1)
    mad = lambda Y: 1.4826 * np.median(np.abs(Y - np.median(Y, 1, keepdims=True)), 1)  # noqa: E731
    mad_nat, mad_car = mad(Xn), mad(Xr[:, :nc])
    drift = p[0] * fs
    rows = {}
    bridged = {c for d in (bridges or {}).get("bridged", []) for c in (d["a"], d["b"])}
    for i, c in enumerate(ch):
        flags = []
        if c in bridged:
            flags.append("bridged")
        if hf_rel[i] > 2.0:
            flags.append("hf_excess(>2x median 30-150 Hz: EMG or poor contact)")
        if r2030[i] < 2.0:
            flags.append("beta_at_hf_floor(20-30/60-90 < 2)")
        if line_ratio[i] > 5.0:
            flags.append("line_noise(>5x flanks: check impedance)")
        if mad_car[i] > 3 * np.median(mad_car):
            flags.append("high_amplitude_after_CAR")
        rows[c] = dict(mad_native_1_40_uV=round(float(mad_nat[i]), 2), mad_robust_car_uV=round(float(mad_car[i]), 2),
                       hf_30_150_rel_median=round(float(hf_rel[i]), 2), ratio_20_30_over_60_90=round(float(r2030[i]), 2),
                       line_ratio=round(float(line_ratio[i]), 2), drift_uV_per_s=round(float(drift[i]), 2), flags=flags)
    return dict(channels=rows, car_set=sel, line_hz=line_hz, median_line_ratio=round(float(np.median(line_ratio)), 2),
                median_drift_uV_per_s=round(float(np.median(drift)), 2),
                note="hf/line ratios on the robust-CAR 0.5-150 Hz copy over the trial crop; native-reference "
                     "amplitude is not used for flagging (reference-adjacent channels look low)")


def raw_proxies(D, band=(1.0, 7.0)):
    """VEOG, HEOG and the hold-out AF4-AF3 proxy from the untouched recording (median removed), band-passed."""
    ch, fs = list(D["ch"]), D["fs"]
    ix = {c: i for i, c in enumerate(ch)}
    Z = D["X"] - np.median(D["X"], axis=1, keepdims=True)
    v = Z[[ix[c] for c in VEOG_POS]].mean(0) - Z[[ix[c] for c in VEOG_NEG]].mean(0)
    h = Z[ix[HEOG_POS[0]]] - Z[ix[HEOG_NEG[0]]]
    rows = [v, h]
    if all(c in ix for c in HOLDOUT) and all(c in ix for c in ("P7", "P8")):
        rows += [Z[[ix["AF3"], ix["AF4"]]].mean(0) - Z[[ix["P7"], ix["P8"]]].mean(0), Z[ix["AF4"]] - Z[ix["AF3"]]]
    return dsp.bandpass(np.vstack(rows), fs, *band)


def ocular_leakage(arrays, D, channels, frontal=("F3", "Fz", "F4")):
    """Absolute ocular leakage: slope of each channel's 1-7 Hz signal on the raw 1-7 Hz proxies (whole crop).
    'brief' = the proxies the regression used (partly circular for a regression method); 'AF' = AF-based proxies
    the method never saw (non-circular check)."""
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    P = raw_proxies(D)
    sel = [ch.index(c) for c in channels if c in ch]
    fr = [k for k, i in enumerate(sel) if ch[i] in frontal]
    sets = {"brief_proxies": P[:2, :nc]}
    if P.shape[0] >= 4:
        sets["AF_proxies"] = P[2:4, :nc]
    out = {}
    for name, X in arrays.items():
        Y = dsp.bandpass(X - np.median(X, 1, keepdims=True), fs, 1.0, 7.0)[sel][:, :nc]
        r = {}
        for pn, E in sets.items():
            A = np.c_[E.T, np.ones(nc)]
            b = np.linalg.lstsq(A, Y.T, rcond=None)[0][0]
            r[pn] = dict(median_abs_bVEOG=round(float(np.median(np.abs(b))), 4),
                         frontal_max_abs_bVEOG=round(float(np.abs(b[fr]).max()), 4) if fr else None)
        out[name] = r
    return out


def band_power_ratios(X, R, D, channels, bands=None):
    """Median over ``channels`` of cleaned / reference power per band (whole crop, Welch 1 s)."""
    bands = bands or {"1-7": (1, 7), "8-13": (8, 13), "13-20": (13, 20), "20-30": (20, 30), "30-40": (30, 40)}
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    sel = [ch.index(c) for c in channels if c in ch]
    f, Px = welch(X[sel][:, :nc], fs, nperseg=int(fs), nfft=int(4 * fs))
    Pr = welch(R[sel][:, :nc], fs, nperseg=int(fs), nfft=int(4 * fs))[1]
    out = {}
    for b, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        rat = Px[:, m].sum(1) / Pr[:, m].sum(1)
        out[b] = dict(median=round(float(np.median(rat)), 4),
                      per_channel={ch[i]: round(float(x), 4) for i, x in zip(sel, rat)})
    return out


def bridge_check(D, cfg=None):
    """Quick setup check (e.g. right after HMD donning): electrical distance of every neighbour pair."""
    bc = (cfg or {}).get("bridge", {"band": [1.0, 30.0], "thr_uV2": 5.0})
    pairs = electrical_distance(D, bc["band"])
    thr = bc["thr_uV2"]
    return dict(threshold_uV2=thr, bridged=[dict(a=a, b=b, ed_uV2=round(e, 3)) for e, a, b in pairs if e < thr],
                lowest=[dict(a=a, b=b, ed_uV2=round(e, 3)) for e, a, b in pairs[:12]],
                all_pairs=[(round(e, 3), a, b) for e, a, b in pairs])
