"""Loader for one trial: neuropype trial-epoch HDF5 (EEG + markers + amplifier DI byte) and the Unity VR JSON.

Time axes
---------
``t``            seconds from the first EEG sample (``lsl_timestamps - lsl_timestamps[0]``).
host / marker    software markers (``sample_index / fs``) and VR host-clock kinematics are placed on ``t`` directly;
                 this is the axis on which ``events``, ``move_onset`` and ``head_speed_unity`` are stored.
EEG (physical)   the amplifier samples the DI trigger byte together with the EEG.  The DI pulses lag the software
                 markers by a constant latency (``di_offset``, ~77 ms on the development setup), so host-clock times
                 must be shifted by ``+di_offset`` to land on the physical EEG sample axis.  Keys ending in ``_di``
                 are already shifted.

Head kinematics are computed on the Unity clock (``head_coordinates.samples[*].t_us_since_trial_start``), which is
immune to the host-timestamp bunching that produces spurious angular-speed spikes when the host timestamps of
``head_position`` are differentiated.

All paths are arguments; nothing is read from fixed locations.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .config import DEFAULT_DI_OFFSET_S, merge_config

#: HDF5 root attributes copied into ``trial['meta']`` (technical fields only; identifiers are not propagated).
PUBLIC_ATTRS = ("fs_hz", "srate_hz", "lsl_nominal_srate_hz", "layout", "schema_version", "n_eeg_channels",
                "n_aux_channels", "aux_extrapolated_samples", "held_fraction", "stream_suffix",
                "session_meta/notch_hz", "baseline_samples")


# ------------------------------------------------------------------------------------------------ HDF5
def read_eeg_hdf5(path):
    """Raw arrays from a neuropype trial HDF5.  EEG is returned as (n_channels, n_samples) float64 in microvolts."""
    import h5py
    with h5py.File(path, "r") as f:
        eeg = f["eeg"][:]
        layout = str(f.attrs.get("layout", "samples_x_channels"))
        X = (eeg.T if layout == "samples_x_channels" else eeg).astype(np.float64)
        lsl = f["lsl_timestamps"][:].astype(np.float64)
        fs = float(f.attrs["fs_hz"]) if "fs_hz" in f.attrs else float(1.0 / np.median(np.diff(lsl)))
        ch = [c.decode() if isinstance(c, bytes) else str(c) for c in f.attrs["channel_labels"]]
        markers = []
        if "markers" in f:
            for r in f["markers"][:]:
                lab = r["label"]
                markers.append(dict(label=lab.decode() if isinstance(lab, bytes) else str(lab),
                                    code=int(r["code"]), sample_index=int(r["sample_index"]),
                                    lsl_timestamp=float(r["lsl_timestamp"])))
        get = lambda k: f[k][:] if k in f else None  # noqa: E731
        acc, cnt, lq, di = get("acc"), get("counter"), get("link_quality"), get("di_byte")
        meta = {}
        for k in PUBLIC_ATTRS:
            if k in f.attrs:
                v = f.attrs[k]
                meta[k] = v.item() if hasattr(v, "item") else (v.decode() if isinstance(v, bytes) else v)
    if X.shape[0] != len(ch):
        raise ValueError(f"EEG has {X.shape[0]} rows but {len(ch)} channel labels")
    return dict(X=X, lsl=lsl, fs=fs, ch=ch, markers=markers,
                acc=None if acc is None else np.asarray(acc, float),
                counter=None if cnt is None else np.asarray(cnt, float),
                link_quality=None if lq is None else np.asarray(lq, float),
                di=None if di is None else np.asarray(di).astype(int), meta=meta)


def read_vr_json(path):
    with open(path) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------ DI alignment
def estimate_di_offset(di, lsl, markers, *, heartbeat=100, queue_gap_s=0.05, max_lag_s=0.5,
                       fallback=DEFAULT_DI_OFFSET_S):
    """Latency of the amplifier DI code pulses relative to the software markers (seconds).

    A DI onset is a sample where the DI byte changes to a code other than 0 and the heartbeat code.  Every
    software marker with a non-zero code is matched to the first DI onset with the same code within
    [-10 ms, +max_lag_s].  Markers that follow the previous marker by less than ``queue_gap_s`` are skipped: their
    DI pulses are serialised behind the previous pulse on the trigger line, so their latency includes queueing.
    Returns a dict with the median latency (``offset_s``) and the per-code table; falls back to ``fallback`` when
    nothing can be matched."""
    rep = dict(method="median(DI onset - software marker) over matched, non-queued codes", fallback_used=False,
               per_code=[], excluded=[])
    if di is None or not markers:
        rep.update(offset_s=float(fallback), n=0, fallback_used=True, note="no DI byte or no markers in the file")
        return rep
    di = np.asarray(di).astype(int)
    on = np.where((di[1:] != di[:-1]) & (di[1:] != 0) & (di[1:] != heartbeat))[0] + 1
    on_t, on_c = lsl[on], di[on]
    lags = []
    prev_t = -np.inf
    for m in sorted(markers, key=lambda r: r["lsl_timestamp"]):
        ts, code = m["lsl_timestamp"], int(m["code"])
        if code == 0:                       # code 0 emits no DI pulse, so it cannot queue the next one
            continue
        queued = ts - prev_t < queue_gap_s
        prev_t = ts
        if queued:
            rep["excluded"].append(dict(label=m["label"], code=code, reason=f"< {queue_gap_s * 1e3:g} ms after the "
                                                                             "previous marker (queued DI pulse)"))
            continue
        k = np.where((on_c == code) & (on_t >= ts - 0.01) & (on_t <= ts + max_lag_s))[0]
        if not k.size:
            rep["excluded"].append(dict(label=m["label"], code=code, reason="no DI pulse with this code"))
            continue
        lag = float(on_t[k[0]] - ts)
        lags.append(lag)
        rep["per_code"].append(dict(label=m["label"], code=code, lag_ms=round(lag * 1e3, 3)))
    if not lags:
        rep.update(offset_s=float(fallback), n=0, fallback_used=True, note="no DI code could be matched")
        return rep
    lags = np.array(lags)
    rep.update(offset_s=float(np.median(lags)), n=int(lags.size), mean_ms=round(float(lags.mean() * 1e3), 3),
               sd_ms=round(float(lags.std(ddof=1) * 1e3), 3) if lags.size > 1 else None,
               range_ms=[round(float(lags.min() * 1e3), 3), round(float(lags.max() * 1e3), 3)])
    return rep


# ------------------------------------------------------------------------------------------------ kinematics
def head_unity(vr, trial_start_s, t):
    """Head yaw / elevation / roll (deg) and angular speed (deg/s) from the HMD forward vector on the Unity clock,
    resampled to a uniform grid, Savitzky-Golay differentiated, and mapped onto ``t`` via the trial-start marker.
    Returned arrays are on the host/marker axis (add the DI offset for the physical EEG axis)."""
    from scipy.signal import savgol_filter
    hc = vr["head_coordinates"]
    hs = hc["samples"]
    fsu = float(hc.get("fs", 60.0) or 60.0)
    tu = np.array([s["t_us_since_trial_start"] for s in hs], float) / 1e6 + trial_start_s
    q = np.array([[s["qx"], s["qy"], s["qz"], s["qw"]] for s in hs], float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    fx = 2 * (x * z + w * y)
    fy = 2 * (y * z - w * x)
    fz = 1 - 2 * (x * x + y * y)
    yaw = np.degrees(np.unwrap(np.arctan2(fx, fz)))
    elev = np.degrees(np.arcsin(np.clip(fy, -1, 1)))
    roll = np.degrees(np.arcsin(np.clip(2 * (x * y + w * z), -1, 1)))
    tg = np.arange(tu[0], tu[-1], 1 / fsu)
    Y = np.vstack([np.interp(tg, tu, v) for v in (yaw, elev, roll)])
    dY = savgol_filter(Y, 9, 2, deriv=1, delta=1 / fsu, axis=1)
    sp = np.sqrt((dY ** 2).sum(0))
    f = lambda v: np.interp(t, tg, v, left=np.nan, right=np.nan)  # noqa: E731
    return dict(yaw=f(Y[0]), elev=f(Y[1]), roll=f(Y[2]), speed=f(sp), yaw_rate=f(dY[0]))


def _robust_speed(t, xyz, fs_target=90.0):
    """Hand speed (m/s): resample uniformly, 5-point median filter (tracking glitches), Savitzky-Golay derivative."""
    from scipy.signal import medfilt, savgol_filter
    tu = np.arange(t[0], t[-1], 1 / fs_target)
    P = np.column_stack([np.interp(tu, t, xyz[:, k]) for k in range(3)])
    P = np.column_stack([medfilt(P[:, k], 5) for k in range(3)])
    V = savgol_filter(P, 11, 3, deriv=1, delta=1 / fs_target, axis=0)
    return tu, P, np.linalg.norm(V, axis=1)


def hand_speed_host(vr, lsl0, t):
    """Hand speed on ``t`` from ``hand_position`` host timestamps (host/marker axis)."""
    hp = vr["hand_position"]
    th = np.array([s["host_ts_ns"] for s in hp], float) / 1e9 - lsl0
    xyz = np.array([[s["x"], s["y"], s["z"]] for s in hp], float)
    o = np.argsort(th, kind="stable")
    th, xyz = th[o], xyz[o]
    tu, _, sp = _robust_speed(th, xyz)
    return np.interp(t, tu, sp, left=np.nan, right=np.nan)


def reach_span(t, hand_speed, t_target, fs, frac=0.10, gap_s=0.5):
    """Movement onset/offset: the contiguous span around the speed peak (after target onset) where hand speed
    exceeds ``frac`` x peak, bridging sub-threshold gaps shorter than ``gap_s`` (the out-and-back reach is one span).
    Returns (onset_s, offset_s, peak_speed)."""
    if not np.isfinite(t_target):
        t_target = -np.inf
    m = (t >= t_target) & np.isfinite(hand_speed)
    if not m.any():
        return float("nan"), float("nan"), float("nan")
    sp = np.where(m, hand_speed, 0.0)
    ipk = int(np.argmax(sp))
    peak = float(sp[ipk])
    above = sp > frac * peak
    gap = int(gap_s * fs)
    lo = ipk
    while lo > 0:
        prev = np.where(above[max(0, lo - gap):lo])[0]
        if not prev.size:
            break
        lo = max(0, lo - gap) + prev[0] if not above[lo - 1] else lo - 1
    hi = ipk
    while hi < len(t) - 1:
        nxt = np.where(above[hi + 1:hi + 1 + gap])[0]
        if not nxt.size:
            break
        hi = hi + 1 + nxt[-1] if not above[hi + 1] else hi + 1
    return float(t[lo]), float(t[hi]), peak


# ------------------------------------------------------------------------------------------------ assembly
def _finish(D, cfg):
    """Derived keys shared by every constructor: crop, DI-shifted events and the decoder epoch."""
    fs, ev = D["fs"], D["events"]
    lab = cfg["crop_label"]
    D["n_crop"] = int(round(ev[lab] * fs)) if lab in ev else int(D["X"].shape[1])
    D["n_crop"] = int(min(max(D["n_crop"], 1), D["X"].shape[1]))
    off = float(D["di_offset"])
    D["events_di"] = {k: v + off for k, v in ev.items()}
    D["move_onset_di"] = D["move_onset"] + off
    D["move_offset_di"] = D["move_offset"] + off
    ep = cfg["epoch"]
    D["decoder_epoch_di"] = (D["move_onset_di"] - ep["pre_s"], D["move_onset_di"] + ep["post_s"])
    return D


def load_trial(eeg_path, vr_path=None, cfg=None, *, di_offset=None):
    """Load one trial.

    Parameters
    ----------
    eeg_path, vr_path : paths to the trial HDF5 and the Unity JSON (``vr_path`` optional: without it no
        kinematics, no head-speed mask and no reach onset are available).
    cfg : configuration (see :mod:`eeg_vr_clean.config`); ``cfg['di_offset']`` = 'auto' or seconds.
    di_offset : overrides ``cfg['di_offset']``.

    Returns a dict (the 'trial'): X (n_ch, N) uV, fs, ch, t, lsl, markers, events (label -> s, host/marker axis),
    n_crop, di_offset (s), di_report, head_speed_unity / head_yaw / head_yaw_rate (host axis, NaN outside the VR
    record), hand_speed, move_onset / move_offset / touch_time (host axis), *_di keys, decoder_epoch_di, acc, meta.
    """
    cfg = merge_config(None, cfg) if cfg is None or "band" not in cfg else cfg
    raw = read_eeg_hdf5(eeg_path)
    fs, lsl = raw["fs"], raw["lsl"]
    t = lsl - lsl[0]
    events = {}
    for m in raw["markers"]:
        events[m["label"]] = m["sample_index"] / fs
    want = cfg["di_offset"] if di_offset is None else di_offset
    rep = estimate_di_offset(raw["di"], lsl, raw["markers"])
    if isinstance(want, str) and want.lower() == "auto":
        off = rep["offset_s"]
    else:
        off = float(want)
        rep = {**rep, "measured_offset_s": rep["offset_s"], "offset_s": off, "note": "fixed offset supplied by caller"}
    D = dict(X=raw["X"], fs=fs, ch=raw["ch"], t=t, lsl=lsl, markers=raw["markers"], events=events,
             acc=raw["acc"], counter=raw["counter"], link_quality=raw["link_quality"], di=raw["di"],
             di_offset=float(off), di_report=rep, meta={**raw["meta"], "eeg_file": os.path.basename(eeg_path)})
    nan = np.full(len(t), np.nan)
    D.update(head_speed_unity=nan.copy(), head_yaw=nan.copy(), head_yaw_rate=nan.copy(), hand_speed=nan.copy(),
             move_onset=float("nan"), move_offset=float("nan"), touch_time=float("nan"), hand_speed_peak=float("nan"))
    t_target = next((v for k, v in events.items() if k.startswith(cfg["target_prefix"])), float("nan"))
    D["t_target"] = float(t_target)
    if vr_path is not None:
        vr = read_vr_json(vr_path)
        D["meta"]["vr_file"] = os.path.basename(vr_path)
        try:
            D["hand_speed"] = hand_speed_host(vr, lsl[0], t)
            on, offt, pk = reach_span(t, D["hand_speed"], t_target, fs)
            D.update(move_onset=on, move_offset=offt, hand_speed_peak=pk)
        except (KeyError, ValueError, IndexError) as e:
            warnings.warn(f"hand kinematics unavailable: {e}")
        ts_lab = cfg["trial_start_label"]
        if ts_lab in events and "head_coordinates" in vr:
            hu = head_unity(vr, events[ts_lab], t)
            D.update(head_speed_unity=hu["speed"], head_yaw=hu["yaw"], head_yaw_rate=hu["yaw_rate"],
                     head_elev=hu["elev"])
        else:
            warnings.warn("no trial_start marker or no Unity head_coordinates: head-speed mask disabled")
        if "touch_latency_s" in vr and np.isfinite(t_target):
            D["touch_time"] = float(t_target + float(vr["touch_latency_s"]))
        for k in ("trial_type", "target_id", "touch_success", "is_catch"):
            if k in vr:
                D["meta"]["vr_" + k] = vr[k]
    return _finish(D, cfg)


def trial_from_arrays(X, fs, ch, *, events=None, head_speed=None, hand_speed=None, move_onset=float("nan"),
                      move_offset=float("nan"), di_offset=0.0, t=None, cfg=None):
    """Build a trial dict from arrays (other recording formats, simulations, tests).  ``head_speed`` and
    ``hand_speed`` are on the host/marker axis (same convention as :func:`load_trial`)."""
    cfg = merge_config(None, cfg) if cfg is None or "band" not in cfg else cfg
    X = np.asarray(X, float)
    N = X.shape[1]
    t = np.arange(N) / fs if t is None else np.asarray(t, float)
    nan = np.full(N, np.nan)
    D = dict(X=X, fs=float(fs), ch=list(ch), t=t, lsl=t.copy(), markers=[], events=dict(events or {}),
             acc=None, counter=None, link_quality=None, di=None, di_offset=float(di_offset),
             di_report=dict(offset_s=float(di_offset), n=0, note="supplied by caller"), meta={},
             head_speed_unity=nan.copy() if head_speed is None else np.asarray(head_speed, float),
             head_yaw=nan.copy(), head_yaw_rate=nan.copy(),
             hand_speed=nan.copy() if hand_speed is None else np.asarray(hand_speed, float),
             move_onset=float(move_onset), move_offset=float(move_offset), touch_time=float("nan"),
             hand_speed_peak=float("nan"), t_target=float("nan"))
    return _finish(D, cfg)


def kin_on_eeg_axis(D, key):
    """Kinematic ``D[key]`` (host/marker axis) moved onto the physical EEG axis: value at t = src(t - di_offset)."""
    t = D["t"]
    src = np.asarray(D[key], float)
    ok = np.isfinite(src)
    if not ok.any():
        return np.full(len(t), np.nan)
    return np.interp(t - D["di_offset"], t[ok], src[ok], left=np.nan, right=np.nan)
