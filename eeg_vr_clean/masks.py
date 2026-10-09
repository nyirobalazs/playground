"""Rule-based masks on the physical (DI-corrected) EEG axis.  Nothing trial-specific is hard-coded.

BAD_motion = [Unity head angular speed (moved by +DI) > 10 deg/s, padded 0.1 s before / 0.4 s after]
             UNION [global 55-95 Hz envelope (median over the robust-CAR channels of the CLEANED 0.5-150 Hz copy,
             100-ms boxcar) > +6 dB over its quiet-anchored 25th percentile, padded 0.1 s].
             Optional third component: 7.5-10.5 Hz envelope of the in-phase central mean (Fz/FC1/C3/Cz/Pz) > 3x its
             quiet level; computed always, applied only when configured.
BAD_transient = 250-ms windows (grid from sample 0, inside the trial) where any decoder channel |x| > 100 uV
             (0.5 Hz high-passed cleaned array; the EOG poles are raw by design and therefore not tested).
BAD_next_trial = from the crop marker (trial reset) to the end of the record.
EYE_blink  = +-150 ms around blinks detected on the raw VEOG (0.5-15 Hz); informational, NOT bad: blink energy
             above the 7 Hz correction band is left on frontal channels, so features in these windows deserve a
             trust flag rather than rejection.
reach_onset_DI = zero-length marker at the DI-corrected kinematic reach onset.

Why masks and not cleaning: head-turn/reach-locked ~9 Hz high-amplitude bursts survived CAR, Laplacian, EOG
regression and ICA in the development recording (the burst IC was unstable across fits and carried genuine
alpha), so "removing" them would either leave them or fabricate an ERD.  Masks must be enforced by the feature
extractor (see :mod:`eeg_vr_clean.features`), because the bursts are still present in X.
"""
from __future__ import annotations

import numpy as np

from . import dsp
from .io import kin_on_eeg_axis


def head_speed_mask(D, mc):
    """(padded mask, raw supra-threshold mask, head speed on the EEG axis)."""
    hs = kin_on_eeg_axis(D, "head_speed_unity")
    raw_m = np.nan_to_num(hs) > mc["head_thr_deg_s"]
    return dsp.pad_mask(raw_m, D["fs"], mc["head_pad_before_s"], mc["head_pad_after_s"]), raw_m, hs


def hf_envelope(Xsrc, D, mc, channels, cfg):
    """Global 55-95 Hz envelope: per-channel Hilbert envelope over ``channels``, 100-ms boxcar, median across
    channels.  Returns (env, quiet level (P25 over the trial), dB re level)."""
    from scipy.signal import hilbert
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    sel = [ch.index(c) for c in channels]
    H = dsp.bandpass(Xsrc[sel], fs, *mc["hf_band"], order=cfg["filter_order"], pad_s=cfg["pad_s"])
    env = dsp.boxcar(np.abs(hilbert(H, axis=1)), int(round(mc["hf_smooth_s"] * fs)))
    g = np.median(env, 0)
    lev = float(np.percentile(g[:nc], mc["hf_quiet_pct"]))
    return g, lev, 20 * np.log10(np.maximum(g, 1e-30) / lev)


def alpha_envelope(Xc, D, mc, cfg):
    """7.5-10.5 Hz envelope of the in-phase central mean, 100-ms boxcar; ratio to its quiet 25th percentile."""
    from scipy.signal import hilbert
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    s = Xc[[ch.index(c) for c in mc["alpha_channels"]]].mean(0)
    env = dsp.boxcar(np.abs(hilbert(dsp.bandpass(s, fs, *mc["alpha_band"], order=cfg["filter_order"],
                                                 pad_s=cfg["pad_s"]))), int(round(mc["alpha_smooth_s"] * fs)))
    lev = float(np.percentile(env[:nc], mc["alpha_quiet_pct"]))
    return env, lev, env / lev


def motion_masks(D, X_clean, X_bb_clean, cfg, hf_channels):
    """Rule-based BAD_motion; returns masks, spans and per-component evidence."""
    mc = cfg["motion"]
    t, fs, nc = D["t"], D["fs"], int(D["n_crop"])
    in_trial = np.arange(len(t)) < nc
    m_head, m_head_raw, hs = head_speed_mask(D, mc)
    g, lev, gdb = hf_envelope(X_bb_clean, D, mc, hf_channels, cfg)
    m_hf_raw = (gdb > mc["hf_thr_db"]) & in_trial
    m_hf = dsp.pad_mask(m_hf_raw, fs, mc["hf_pad_s"], mc["hf_pad_s"])
    union = (m_head | m_hf) & in_trial
    env_a, lev_a, ratio_a = alpha_envelope(X_clean, D, mc, cfg)
    m_al_raw = (ratio_a > mc["alpha_thr_x"]) & in_trial
    m_al = dsp.pad_mask(m_al_raw, fs, mc["alpha_pad_s"], mc["alpha_pad_s"])
    cover = []
    for a, b in mc.get("validation_spans") or ():
        w = (t >= a) & (t < b)
        cover.append(dict(span=[a, b], covered_by_union=round(float(union[w].mean()), 4) if w.any() else None))
    missed = any(c["covered_by_union"] is not None and c["covered_by_union"] < mc["validation_min_coverage"]
                 for c in cover)
    rule = mc["alpha_apply"]
    applied_alpha = bool(missed) if rule == "if_union_misses_validation_span" else (rule is True or rule == "always")
    final = union | (m_al & in_trial if applied_alpha else np.zeros_like(union))
    final &= in_trial
    out = dict(mask=final, head=m_head & in_trial, hf=m_hf, alpha=m_al, head_raw=m_head_raw, hf_raw=m_hf_raw,
               alpha_raw=m_al_raw, head_speed_eeg=hs, hf_env=g, hf_level=lev, hf_db=gdb, alpha_env=env_a,
               alpha_level=lev_a, alpha_ratio=ratio_a, applied_alpha=applied_alpha, validation_coverage=cover)
    out["spans"] = dict(head_speed_raw_gt_thr=dsp.runs(m_head_raw, t, nc), head_speed_padded=dsp.runs(m_head, t, nc),
                        hf_raw_gt_thr=dsp.runs(m_hf_raw, t, nc), hf_padded=dsp.runs(m_hf, t, nc),
                        alpha_raw_gt_thr=dsp.runs(m_al_raw, t, nc), alpha_padded=dsp.runs(m_al, t, nc),
                        BAD_motion=dsp.runs(final, t, nc))
    return out


def transient_windows(Xc, D, cfg):
    """250-ms grid windows inside the trial where any decoder channel |x| > thr (high-passed cleaned array)."""
    tc = cfg["transient"]
    fs, ch = D["fs"], list(D["ch"])
    L = int(round(tc["win_s"] * fs))
    nc = int(D["n_crop"])
    H = dsp.bandpass(Xc - np.median(Xc, 1, keepdims=True), fs, tc["hp_hz"], None, order=cfg["filter_order"],
                     pad_s=cfg["pad_s"])
    dec = [ch.index(c) for c in cfg["decoder_channels"] if c in ch]
    n_win = nc // L
    W = np.abs(H[dec][:, :n_win * L].reshape(len(dec), n_win, L)).max(-1).max(0)
    return [(w * L / fs, L / fs) for w in np.where(W > tc["thr_uV"])[0]], W


def blink_events(D, cfg):
    """Blink peaks on the raw VEOG (untouched recording, 0.5-15 Hz) inside the trial crop.
    Returns (times_s, peak_uV, veog_filtered)."""
    from scipy.signal import find_peaks
    from .bridging import detrended
    from .config import VEOG_NEG, VEOG_POS
    bc = cfg["blink"]
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    ix = {c: i for i, c in enumerate(ch)}
    Zd, _ = detrended(D)
    v = Zd[[ix[c] for c in VEOG_POS]].mean(0) - Zd[[ix[c] for c in VEOG_NEG]].mean(0)
    v = dsp.bandpass(v, fs, *bc["band"], order=cfg["filter_order"], pad_s=cfg["pad_s"])
    pk, _ = find_peaks(v[:nc], height=bc["height_uV"], prominence=bc["prominence_uV"],
                       distance=max(1, int(bc["min_dist_s"] * fs)),
                       width=(bc["width_s"][0] * fs, bc["width_s"][1] * fs), rel_height=0.5)
    return D["t"][pk], v[pk], v


def build_annotations(D, motion_spans, transient, blinks, cfg):
    """(onset, duration, description) arrays, sorted by onset."""
    fs = D["fs"]
    t_end = float(D["t"][-1]) + 1.0 / fs
    t_cr = D["n_crop"] / fs
    on, du, de = [], [], []
    for a, b in motion_spans:
        on.append(a); du.append(b - a); de.append("BAD_motion")
    for a, d in transient:
        on.append(a); du.append(d); de.append("BAD_transient")
    if D["n_crop"] < D["X"].shape[1]:
        on.append(t_cr); du.append(t_end - t_cr); de.append("BAD_next_trial")
    hw = cfg["blink"]["half_window_s"]
    for b in blinks:
        on.append(max(0.0, float(b) - hw)); du.append(2 * hw); de.append("EYE_blink")
    if np.isfinite(D.get("move_onset_di", np.nan)):
        on.append(float(D["move_onset_di"])); du.append(0.0); de.append("reach_onset_DI")
    o = np.argsort(on, kind="stable")
    return np.array(on, float)[o], np.array(du, float)[o], np.array(de, dtype="<U16")[o]


def bad_mask(t, onset, duration, desc):
    """True where any annotation whose description starts with 'BAD' covers the sample."""
    m = np.zeros(len(t), bool)
    for o, d, s in zip(onset, duration, desc):
        if str(s).lower().startswith("bad"):
            m |= (t >= o) & (t < o + d)
    return m
