"""Synthetic 32-channel recordings with known artefacts (no participant data anywhere in the test suite).

Background: independent 1/f noise per channel + a 10 Hz posterior/central rhythm + a 20 Hz central beta source.
Artefacts: blinks (vertical ocular field, frontal -> posterior fall-off), saccades / sustained gaze offsets (lateral
field), optional gel bridge (duplicated channel + tiny noise), optional broadband 55-95 Hz burst.
Everything is in microvolts on a native (single-electrode) reference.
"""
from __future__ import annotations

import json

import numpy as np

CH = ["FP1", "FP2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz",
      "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "Oz"]
FS = 500.0

# horizontal (saccade) propagation: signed (right = +)
_H = {"F8": 1.0, "F7": -1.0, "AF4": 0.3, "AF3": -0.3, "FP2": 0.2, "FP1": -0.2, "F4": 0.15, "F3": -0.15,
      "FC6": 0.15, "FC5": -0.15, "T8": 0.3, "T7": -0.3, "C4": 0.05, "C3": -0.05, "CP6": 0.05, "CP5": -0.05}


def vgain(c):
    """Vertical (blink) propagation as a fraction of the FP amplitude: frontal -> posterior fall-off."""
    for prefix, g in (("FP", 1.0), ("AF", 0.5), ("F7", 0.25), ("F8", 0.25), ("FC", 0.15), ("F", 0.3), ("T", 0.08),
                      ("CP", 0.05), ("C", 0.1), ("PO", 0.02), ("P", 0.03)):
        if c.startswith(prefix):
            return g
    return 0.02


def hgain(c):
    return _H.get(c, 0.0)


def pink(rng, n_ch, N, fs, rms):
    W = rng.standard_normal((n_ch, N))
    F = np.fft.rfft(W, axis=1)
    f = np.fft.rfftfreq(N, 1 / fs)
    F[:, 1:] /= np.sqrt(f[1:])
    F[:, 0] = 0
    X = np.fft.irfft(F, n=N, axis=1)
    return X / X.std(1, keepdims=True) * rms


def band_noise(rng, N, fs, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(N))
    return x / x.std()


def make_eeg(seed=0, dur=12.0, fs=FS, rms=8.0):
    """Clean background EEG (n_ch, N) with alpha/mu (10 Hz) and beta (20 Hz) sources."""
    rng = np.random.default_rng(seed)
    N = int(dur * fs)
    t = np.arange(N) / fs
    X = pink(rng, len(CH), N, fs, rms)
    alpha = band_noise(rng, N, fs, 8.5, 11.5) * 6.0
    beta = band_noise(rng, N, fs, 17.0, 23.0) * 2.5
    for i, c in enumerate(CH):
        X[i] += alpha * (1.0 if c[0] in "PO" else 0.6 if c[0] == "C" else 0.2)
        X[i] += beta * (1.0 if c in ("C3", "Cz", "C4", "CP1", "CP2", "FC1", "FC2") else 0.3)
    return X, t


def blink_wave(t, times, amp=300.0, sigma=0.034):
    v = np.zeros_like(t)
    for b in times:
        v += amp * np.exp(-0.5 * ((t - b) / sigma) ** 2)
    return v


def gaze_wave(t, segments, amp=100.0, tau=0.012):
    """Sustained horizontal gaze offsets: list of (t_on, t_off, sign)."""
    from scipy.special import expit
    h = np.zeros_like(t)
    for a, b, s in segments:
        h += s * amp * (expit((t - a) / tau) - expit((t - b) / tau))
    return h


def artefacts(t, blinks, gaze):
    v = blink_wave(t, blinks)
    h = gaze_wave(t, gaze)
    A = np.vstack([vgain(c) * v + hgain(c) * h for c in CH])
    return A, v, h


DEFAULT_BLINKS = [0.6, 1.4, 2.3, 3.1, 4.0, 4.9, 5.8, 6.6, 7.5, 8.3, 9.2, 10.1, 10.9, 11.5]
DEFAULT_GAZE = [(1.8, 2.6, 1.0), (3.4, 3.9, -1.0), (6.9, 7.8, 1.0), (9.5, 10.4, -0.8)]


def make_recording(seed=0, dur=12.0, blinks=None, gaze=None, bridge=("F4", "FC6"), hf_burst=None):
    """Returns dict(X raw, X0 clean, A artefact, t, blinks, ...)."""
    X0, t = make_eeg(seed, dur)
    rng = np.random.default_rng(seed + 1000)
    blinks = DEFAULT_BLINKS if blinks is None else blinks
    gaze = DEFAULT_GAZE if gaze is None else gaze
    A, v, h = artefacts(t, [b for b in blinks if b < dur], [g for g in gaze if g[1] < dur])
    if hf_burst is not None:
        a, b = hf_burst
        m = ((t >= a) & (t < b)).astype(float)
        for i in range(len(CH)):
            A[i] += 25.0 * band_noise(rng, len(t), FS, 55.0, 95.0) * m
    X = X0 + A
    if bridge:
        keep, dup = bridge
        i, j = CH.index(keep), CH.index(dup)
        X[j] = X[i] + 0.3 * rng.standard_normal(len(t))
        X0 = X0.copy()
        X0[j] = X0[i]
    return dict(X=X, X0=X0, A=A, t=t, veog=v, heog=h, blinks=blinks, gaze=gaze)


# ------------------------------------------------------------------------------------------------ files
def yaw_quat(theta_rad):
    return np.c_[np.zeros_like(theta_rad), np.sin(theta_rad / 2), np.zeros_like(theta_rad), np.cos(theta_rad / 2)]


def write_files(tmpdir, seed=0, dur=12.4, di_lag=0.080, crop_s=12.0, turn=(5.0, 5.6, 20.0)):
    """Write a neuropype-style trial HDF5 and a Unity-style VR JSON.

    turn = (start_s, end_s, yaw_deg) on the host/marker axis: a smooth (raised-cosine velocity) head turn.
    Markers: trial_start (99) at 1.0 s, trial_type (70) 1 ms later (queued DI pulse), target_on_1 (1) at 3.0 s,
    trial_reset (0) at crop_s.  DI pulses (5 samples) follow each non-zero marker by ``di_lag``; a 1-sample heartbeat
    code 100 every 50 samples."""
    import h5py
    rec = make_recording(seed, dur)
    N = rec["X"].shape[1]
    lsl0 = 1000.0
    lsl = lsl0 + np.arange(N) / FS
    marks = [("trial_start", 99, 1.0), ("trial_type_executed", 70, 1.001), ("target_on_1", 1, 3.0),
             ("trial_reset", 0, crop_s)]
    di = np.zeros(N)
    di[::50] = 100
    last_end = -1
    for lab, code, ts in marks:
        if code == 0:
            continue
        i0 = int(round((ts + di_lag) * FS))
        i0 = max(i0, last_end + 12)          # serialised behind the previous pulse
        di[i0:i0 + 5] = code
        last_end = i0 + 5
    mdt = np.dtype([("sample_index", "<u8"), ("sample_index_valid", "u1"), ("seq", "<u4"), ("lsl_timestamp", "<f8"),
                    ("code", "<u2"), ("source", "S16"), ("label", "S64")])
    mk = np.array([(int(round(ts * FS)), 1, k, lsl0 + ts, code, b"software", lab.encode())
                   for k, (lab, code, ts) in enumerate(marks)], dtype=mdt)
    eeg_path = f"{tmpdir}/trial.hdf5"
    with h5py.File(eeg_path, "w") as f:
        f.create_dataset("eeg", data=rec["X"].T.astype(np.float32))
        f.create_dataset("lsl_timestamps", data=lsl)
        f.create_dataset("markers", data=mk)
        f.create_dataset("di_byte", data=di.astype(np.float32))
        f.create_dataset("acc", data=np.zeros((N, 3), np.float32))
        f.create_dataset("counter", data=np.arange(N, dtype=float))
        f.create_dataset("link_quality", data=np.ones(N, np.float32))
        f.attrs["fs_hz"] = FS
        f.attrs["channel_labels"] = np.array(CH, dtype=h5py.string_dtype())
        f.attrs["layout"] = "samples_x_channels"
        f.attrs["schema_version"] = "1.0"
    # VR: head on the Unity clock (60 Hz, relative to trial start), hand on the host clock (90 Hz)
    ts0 = 1.0
    tu = np.arange(0, dur - ts0, 1 / 60.0)
    a, b, deg = turn
    th = np.clip((tu + ts0 - a) / (b - a), 0, 1)
    yaw = np.radians(deg) * (th - np.sin(2 * np.pi * th) / (2 * np.pi))       # raised-cosine velocity profile
    q = yaw_quat(yaw)
    head = [dict(t_us_since_trial_start=int(round(x * 1e6)), x=0.0, y=1.3, z=0.0, qx=float(qq[0]), qy=float(qq[1]),
                 qz=float(qq[2]), qw=float(qq[3]), tracking_ok=True) for x, qq in zip(tu, q)]
    th_host = np.arange(0.05, dur, 1 / 90.0)
    reach = np.clip((th_host - 5.3) / 0.6, 0, 1)
    px = 0.3 + 0.4 * (reach - np.sin(2 * np.pi * reach) / (2 * np.pi))
    hand = [dict(host_ts_ns=int(round((lsl0 + x) * 1e9)), x=float(p), y=1.0, z=0.2, tracking_ok=1.0)
            for x, p in zip(th_host, px)]
    vr = dict(schema_version="test", trial_type="executed", target_id=1, touch_latency_s=2.6,
              head_coordinates=dict(fs=60.0, frame="world", samples=head), hand_position=hand)
    vr_path = f"{tmpdir}/trial_vr.json"
    with open(vr_path, "w") as fh:
        json.dump(vr, fh)
    peak_deg_s = 2 * deg / (b - a)                # peak of the raised-cosine velocity
    return dict(eeg=eeg_path, vr=vr_path, rec=rec, di_lag=di_lag, turn=turn, peak_deg_s=peak_deg_s, crop_s=crop_s)
