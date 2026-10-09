"""Small signal-processing helpers shared by every module (all filters are zero-phase SOS Butterworth)."""
from __future__ import annotations

import numpy as np


def bandpass(X, fs, lo=None, hi=None, order=4, pad_s=3.0):
    """Zero-phase Butterworth (SOS).  ``lo=None`` -> low-pass, ``hi=None`` -> high-pass.

    The signal is extended by odd reflection over ``pad_s`` seconds (or the whole record if shorter)."""
    from scipy.signal import butter, sosfiltfilt
    X = np.asarray(X, float)
    if lo and hi:
        sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    elif lo:
        sos = butter(order, lo, btype="high", fs=fs, output="sos")
    elif hi:
        sos = butter(order, hi, btype="low", fs=fs, output="sos")
    else:
        return X.copy()
    padlen = min(int(round(pad_s * fs)), X.shape[-1] - 1)
    return sosfiltfilt(sos, X, axis=-1, padtype="odd", padlen=padlen)


def lowpass(X, fs, hi, order=4, pad_s=3.0):
    return bandpass(X, fs, None, hi, order=order, pad_s=pad_s)


def span_mask(t, spans):
    """Boolean mask of samples with t in any [a, b) span."""
    m = np.zeros(len(t), bool)
    for a, b in spans or ():
        m |= (t >= a) & (t < b)
    return m


def runs(m, t, nmax=None):
    """Contiguous True runs of a boolean mask -> [(t_start, t_end_exclusive)] on the t axis."""
    m = np.asarray(m, bool)
    if nmax is not None:
        m = m.copy()
        m[nmax:] = False
    d = np.diff(np.r_[0, m.astype(int), 0])
    on, off = np.where(d == 1)[0], np.where(d == -1)[0]
    if len(t) < 2:
        return []
    fs = 1.0 / np.median(np.diff(t))
    return [(float(t[a]), float(t[b - 1] + 1.0 / fs)) for a, b in zip(on, off)]


def pad_mask(m, fs, before_s, after_s):
    """Dilate every True run by ``before_s`` (earlier) and ``after_s`` (later)."""
    m = np.asarray(m, bool)
    out = np.zeros_like(m)
    d = np.diff(np.r_[0, m.astype(int), 0])
    nb, na = int(round(before_s * fs)), int(round(after_s * fs))
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        out[max(0, a - nb):min(len(m), b + na)] = True
    return out


def boxcar(x, n):
    """Centred moving average (``np.convolve(..., mode='same')``) along the last axis."""
    k = np.ones(max(int(n), 1)) / max(int(n), 1)
    x = np.asarray(x, float)
    if x.ndim == 1:
        return np.convolve(x, k, mode="same")
    return np.vstack([np.convolve(r, k, mode="same") for r in x])


def corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def welch_runs(y, m, fs, nperseg=None, nfft=None):
    """Welch PSD averaged over the contiguous True runs of ``m`` (runs shorter than one segment are skipped);
    each run is weighted by its length.  Returns (f, P) or (None, None)."""
    from scipy.signal import welch
    nperseg = int(round(fs)) if nperseg is None else nperseg
    nfft = 4 * nperseg if nfft is None else nfft
    idx = np.where(np.asarray(m, bool))[0]
    if idx.size == 0:
        return None, None
    rr = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    Ps, W, f = [], 0, None
    for r in rr:
        if len(r) < nperseg:
            continue
        f, p = welch(y[..., r] - y[..., r].mean(-1, keepdims=True), fs, nperseg=nperseg,
                     noverlap=int(0.75 * nperseg), nfft=nfft, axis=-1)
        Ps.append(p * len(r))
        W += len(r)
    if not Ps:
        return None, None
    return f, np.sum(Ps, 0) / W


def peak_db(f, p, lo=7.0, hi=13.0):
    """Aperiodic-corrected peak: log-log line fit over 3-6.5 and 15-30 Hz; max residual (dB) in [lo, hi] Hz."""
    f, p = np.asarray(f, float), np.asarray(p, float)
    s = ((f >= 3) & (f <= 6.5)) | ((f >= 15) & (f <= 30))
    c = np.polyfit(np.log10(f[s]), np.log10(p[s]), 1)
    b = (f >= lo) & (f <= hi)
    resid = 10 * (np.log10(p[b]) - np.polyval(c, np.log10(f[b])))
    i = int(np.argmax(resid))
    return float(f[b][i]), float(resid[i])


def numerical_rank(X, rel_tol=1e-6):
    Z = np.asarray(X, float)
    Z = Z - Z.mean(1, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    return int((s > s[0] * rel_tol).sum()) if s.size and s[0] > 0 else 0
