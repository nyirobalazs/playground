"""Band-limited EOG regression (the delivered correction) and an optional band-limited ICA cross-check.

Regression
----------
VEOG = mean(FP1, FP2) - mean(PO7, PO3, PO4, PO8, Oz) and HEOG = F8 - F7 are built from the untouched recording
(bipolar, hence reference-free).  For every channel of the robust-CAR data, the < ``lp_hz`` part is regressed on the
< ``lp_hz`` proxies with a standardised ridge (lambda = 1 % of the trace), using only calibration samples.  Only the
< ``lp_hz`` part of ``gain x proxy`` is subtracted, from every channel except the proxy poles (FP1/FP2/F7/F8), which
stay uncorrected and are typed ``eog`` downstream.

Why < 7 Hz and never above: the proxies are electrode differences, so they also carry brain rhythms.  In the
development recording F8 - F7 at 8-13 Hz tracked the C4 mu rhythm (r 0.66-0.93) and VEOG carried posterior alpha
through its negative pole; subtracting the proxies up to 12 Hz removed 30-48 % of quiet-window frontal mu and
created an ERD-like C4 decrease during the reach.  Below 7 Hz the subtraction removed the ocular low band (decoder
1-7 Hz power to ~0.26x) while 13-40 Hz power stayed at 1.000x.  The cost: the > 7 Hz part of blinks is left on the
frontal channels (flagged via EYE_blink annotations and the usability table).

Gains are fitted on calibration spans (or loaded from a calibration file) and then frozen, so the method is
deterministic, runs online, and can be calibrated once per session (blink + saccade block) and applied per trial.

ICA cross-check (optional, QA only)
-----------------------------------
Extended-infomax ICA on the bridge-free channels (CAR, 1-40 Hz), ocular ICs selected by |r| with the proxies plus a
multi-dimensional blink-subspace rule, alpha veto, and subtraction of ONLY the < ``lp_hz`` back-projection.  Never
subtract an ocular IC broadband: in the development recording broadband removal of the two ocular ICs inflated
decoder 13-30 Hz power 1.3-2.2x and 30-40 Hz power 1.3-2.1x (the removed projection is anticorrelated with the
remainder in beta), in every one of nine fit variants.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import __version__, dsp
from .config import HEOG_NEG, HEOG_POS, VEOG_NEG, VEOG_POS
from .reference import robust_car


# ------------------------------------------------------------------------------------------------ proxies
def prep(D, band, cfg):
    """Median removal, linear detrend fitted on the trial crop (applied to the full record), zero-phase band-pass.
    Native reference."""
    Z = D["X"] - np.median(D["X"], axis=1, keepdims=True)
    nc = int(D["n_crop"])
    tt = np.arange(Z.shape[1], dtype=float)
    p = np.polyfit(tt[:nc], Z[:, :nc].T, 1)
    Zd = Z - (np.outer(p[0], tt) + p[1][:, None])
    return dsp.bandpass(Zd, D["fs"], band[0], band[1], order=cfg["filter_order"], pad_s=cfg["pad_s"]), p


def proxies(D, Xb, names=("VEOG", "HEOG")):
    """Proxy signals (same band as ``Xb``), built from the band-passed native channels."""
    ch = list(D["ch"])
    ix = {c: i for i, c in enumerate(ch)}
    m = lambda cs: Xb[[ix[c] for c in cs]].mean(0)  # noqa: E731
    base = {"VEOG": lambda: m(VEOG_POS) - m(VEOG_NEG), "HEOG": lambda: m(HEOG_POS) - m(HEOG_NEG),
            "REOG": lambda: m(["FP1", "FP2", "F7", "F8"]) - m(VEOG_NEG)}
    missing = [c for c in VEOG_POS + VEOG_NEG + HEOG_POS + HEOG_NEG if c not in ix]
    if missing:
        raise ValueError(f"EOG proxy channels missing from the recording: {missing}")
    return np.vstack([base[n]() for n in names])


# ------------------------------------------------------------------------------------------------ ridge
def ridge_samples(Ef, Yf, lam_frac=0.01, standardise=True):
    """Per-channel ridge of Yf (n_ch, n) on Ef (p, n), both already restricted to the fit samples."""
    Ef = Ef - Ef.mean(1, keepdims=True)
    Yf = Yf - Yf.mean(1, keepdims=True)
    n = Ef.shape[1]
    sd = Ef.std(1, keepdims=True) if standardise else np.ones((Ef.shape[0], 1))
    Es = Ef / sd
    S = Es @ Es.T / n
    lam = lam_frac * np.trace(S)
    Bs = np.linalg.solve(S + lam * np.eye(len(S)), Es @ Yf.T / n)
    B = Bs / sd
    ev = np.linalg.eigvalsh(S)
    R2 = 1 - ((Yf - B.T @ Ef) ** 2).sum(1) / np.maximum((Yf ** 2).sum(1), 1e-30)
    corr = S / np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S)))
    return B, dict(lambda_=float(lam), n_fit=int(n), cond=float(ev.max() / ev.min()), proxy_corr=corr.tolist(),
                   proxy_sd_uV=Ef.std(1).tolist(), shrink_eig=(ev / (ev + lam)).tolist(), R2=R2.tolist())


def ridge(E, Y, mask, lam_frac=0.01, standardise=True):
    return ridge_samples(E[:, mask], Y[:, mask], lam_frac, standardise)


# ------------------------------------------------------------------------------------------------ calibration spans
def auto_fit_spans(D, cfg):
    """Automatic calibration spans for a trial without a calibration block: the trial crop minus ``auto_edge_s`` at
    both ends, minus head motion (Unity head speed > threshold on the EEG axis, padded like the motion mask) and minus
    the reach (hand speed > ``auto_hand_frac`` x peak, padded 0.1 s).  Pre-movement saccades stay in, so saccade gains
    are not extrapolated."""
    from .io import kin_on_eeg_axis
    oc, mc = cfg["ocular"], cfg["motion"]
    t, fs, nc = D["t"], D["fs"], int(D["n_crop"])
    m = (np.arange(len(t)) < nc) & (t >= t[0] + oc["auto_edge_s"]) & (t < t[min(nc, len(t)) - 1] - oc["auto_edge_s"])
    hs = kin_on_eeg_axis(D, "head_speed_unity")
    m &= ~dsp.pad_mask(np.nan_to_num(hs) > oc["auto_head_thr_deg_s"], fs, mc["head_pad_before_s"], mc["head_pad_after_s"])
    hand = kin_on_eeg_axis(D, "hand_speed")
    if np.isfinite(hand).any():
        pk = np.nanmax(hand)
        m &= ~dsp.pad_mask(np.nan_to_num(hand) > oc["auto_hand_frac"] * pk, fs, 0.1, 0.1)
    spans = [list(s) for s in dsp.runs(m, t, nc) if s[1] - s[0] >= oc.get("auto_min_run_s", 0.0)]
    tot = sum(b - a for a, b in spans)
    if tot < oc["auto_min_s"]:
        warnings.warn(f"only {tot:.1f} s of automatic calibration data; prefer a calibration block")
    return spans


def fit_spans_for(D, cfg, spans=None):
    spans = spans if spans is not None else cfg["ocular"]["fit_spans"]
    return [list(map(float, s)) for s in spans] if spans is not None else auto_fit_spans(D, cfg)


# ------------------------------------------------------------------------------------------------ gain model
def fit_gains(trials, cfg, car_sel, drop, spans_per_trial=None):
    """Fit one gain model on the calibration samples of one or more trials (session mode = several trials, whose
    calibration samples are concatenated after per-trial filtering, so no filter runs across trial boundaries).

    Returns a model dict: ``B`` (n_regressors, n_channels) in robust-CAR space with zeros for the poles and the
    dropped duplicates, plus provenance.  Save it with :func:`save_calibration`."""
    oc = cfg["ocular"]
    ch = list(trials[0]["ch"])
    Ecat, Ycat, used = [], [], []
    for k, D in enumerate(trials):
        if list(D["ch"]) != ch:
            raise ValueError("all trials of a session must have the same channel list and order")
        Xb, _ = prep(D, cfg["band"], cfg)
        Xr = robust_car(Xb, ch, car_sel)
        E = proxies(D, Xb, oc["regressors"])
        E_lp = dsp.lowpass(E, D["fs"], oc["lp_hz"], order=oc["lp_order"], pad_s=cfg["pad_s"])
        Y_lp = dsp.lowpass(Xr, D["fs"], oc["lp_hz"], order=oc["lp_order"], pad_s=cfg["pad_s"])
        sp = fit_spans_for(D, cfg, None if spans_per_trial is None else spans_per_trial[k])
        m = dsp.span_mask(D["t"], sp)
        Ecat.append(E_lp[:, m])
        Ycat.append(Y_lp[:, m])
        used.append(dict(trial=k, spans_s=[[round(a, 4), round(b, 4)] for a, b in sp], n_samples=int(m.sum())))
    B, rd = ridge_samples(np.hstack(Ecat), np.hstack(Ycat), oc["ridge_lambda_frac"], oc["standardise"])
    corrected = [c for c in ch if c not in oc["poles"] and c not in drop]
    B_applied = np.zeros_like(B)
    ci = [ch.index(c) for c in corrected]
    B_applied[:, ci] = B[:, ci]
    return dict(B=B_applied, B_full=B, regressors=list(oc["regressors"]), ch=ch, corrected=corrected,
                poles=list(oc["poles"]), drop=list(drop), car_set=list(car_sel), lp_hz=oc["lp_hz"],
                lp_order=oc["lp_order"], band=list(cfg["band"]), ridge=rd, calibration=used,
                n_fit_s=float(sum(u["n_samples"] for u in used) / trials[0]["fs"]))


def save_calibration(model, path):
    """Write a gain model as JSON (gains per regressor and channel + the settings it is valid for)."""
    import json
    out = {"format": "eeg_vr_clean.ocular_gains/1", "package_version": __version__,
           "regressors": model["regressors"], "ch": model["ch"], "corrected": model["corrected"],
           "poles": model["poles"], "drop": model["drop"], "car_set": model["car_set"], "lp_hz": model["lp_hz"],
           "lp_order": model["lp_order"], "band": model["band"], "n_fit_s": model["n_fit_s"],
           "calibration": model["calibration"],
           "ridge": {k: v for k, v in model["ridge"].items() if k != "R2"},
           "gains": {r: {c: float(model["B"][k, i]) for i, c in enumerate(model["ch"])}
                     for k, r in enumerate(model["regressors"])}}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return path


def load_calibration(path, ch=None):
    import json
    with open(path) as fh:
        d = json.load(fh)
    if d.get("format") != "eeg_vr_clean.ocular_gains/1":
        raise ValueError(f"{path}: not an eeg_vr_clean gain file")
    chs = list(d["ch"]) if ch is None else list(ch)
    missing = [c for c in chs if c not in d["gains"][d["regressors"][0]]]
    if missing:
        raise ValueError(f"calibration file has no gains for {missing}")
    B = np.array([[d["gains"][r][c] for c in chs] for r in d["regressors"]], float)
    return dict(B=B, B_full=B, regressors=d["regressors"], ch=chs, corrected=d["corrected"], poles=d["poles"],
                drop=d["drop"], car_set=d["car_set"], lp_hz=d["lp_hz"], lp_order=d["lp_order"], band=d["band"],
                ridge=d.get("ridge", {}), calibration=d.get("calibration", []), n_fit_s=d.get("n_fit_s"),
                source=str(path))


# ------------------------------------------------------------------------------------------------ application
def apply_chain(D, band, model, cfg, M, car_sel):
    """Same correction on any band copy: prep(band) -> robust CAR -> subtract B' x proxy(< lp) on the corrected
    channels -> spline re-estimate of the dropped duplicates -> robust CAR.  Returns (X, removed, E_lp)."""
    from .bridging import apply_interp
    oc = cfg["ocular"]
    ch = list(D["ch"])
    if model["lp_hz"] != oc["lp_hz"] or list(model["regressors"]) != list(oc["regressors"]):
        raise ValueError("gain model and configuration disagree on lp_hz / regressors")
    Xb, _ = prep(D, band, cfg)
    Xr = robust_car(Xb, ch, car_sel)
    E = proxies(D, Xb, oc["regressors"])
    E_lp = dsp.lowpass(E, D["fs"], oc["lp_hz"], order=oc["lp_order"], pad_s=cfg["pad_s"])
    ci = [ch.index(c) for c in model["corrected"]]
    removed = np.zeros_like(Xr)
    removed[ci] = model["B"][:, ci].T @ E_lp
    X = robust_car(apply_interp(Xr - removed, ch, M), ch, car_sel)
    return X, removed, E_lp


# ------------------------------------------------------------------------------------------------ ICA cross-check
def _is_central_parietal(label):
    u = label.upper()
    return u.startswith(("C", "P")) and not u.startswith("FC")


def ica_crosscheck(D, cfg, drop, M, car_sel, X_reg=None):
    """Band-limited ICA ocular correction for comparison with the regression output (QA only, not delivered).

    Steps: ICA copy = bridge-free channels, CAR, ``ica_check.band``; extended infomax with a fixed seed.  Ocular ICs =
    |r| > ``abs_r_thr`` with the 1-7 Hz raw VEOG or VEOG-partialled HEOG, UNION blink-subspace ICs (map maximum on
    FP1/FP2/AF3/AF4 AND |r_VEOG| > ``blink_subspace_r``), minus any IC with an 8-13 Hz peak >= ``alpha_veto_db`` above
    1/f and a central/parietal map maximum; at most ``max_remove``.  Only the < ``lp_hz`` part of the ocular
    back-projection is subtracted.  Returns the ICA-cleaned array and agreement metrics with ``X_reg``."""
    import mne
    from .bridging import to_raw, apply_interp
    ic = cfg["ica_check"]
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    ix = {c: i for i, c in enumerate(ch)}
    keep = [c for c in ch if c not in drop]
    Z = D["X"] - np.median(D["X"], axis=1, keepdims=True)
    tt = np.arange(Z.shape[1], dtype=float)
    p = np.polyfit(tt[:nc], Z[:, :nc].T, 1)
    Zd = Z - (np.outer(p[0], tt) + p[1][:, None])
    Zf = dsp.bandpass(Zd, fs, *ic["band"], order=cfg["filter_order"], pad_s=cfg["pad_s"])[[ix[c] for c in keep]]
    Zf = Zf - Zf.mean(0, keepdims=True)
    raw_fit = to_raw(Zf[:, :nc], keep, fs)
    n_comp = int(min(ic["n_components"], len(keep) - 1))
    ica = mne.preprocessing.ICA(n_components=n_comp, method=ic["method"],
                                fit_params=dict(extended=bool(ic["extended"])) if ic["method"] == "infomax" else None,
                                random_state=ic["random_state"], max_iter=ic["max_iter"], verbose="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(raw_fit, verbose="error")
    S = ica.get_sources(raw_fit).get_data()
    A = ica.get_components()
    v = Z[[ix[c] for c in VEOG_POS]].mean(0) - Z[[ix[c] for c in VEOG_NEG]].mean(0)
    h = Z[ix[HEOG_POS[0]]] - Z[ix[HEOG_NEG[0]]]
    V, H = dsp.bandpass(np.vstack([v, h]), fs, 1.0, 7.0)[:, :nc]
    Hp = H - np.polyval(np.polyfit(V, H, 1), V)
    S17 = dsp.bandpass(S, fs, 1.0, 7.0)
    stats = []
    for k in range(S.shape[0]):
        f, P = dsp.welch_runs(S[k], np.ones(nc, bool), fs)
        pk_hz, pk_db = dsp.peak_db(f, P, 8.0, 13.0)
        imax = keep[int(np.argmax(np.abs(A[:, k])))]
        stats.append(dict(ic=k, r_veog=dsp.corr(S17[k], V), r_heog_partial=dsp.corr(S17[k], Hp), map_max=imax,
                          alpha_peak_db=pk_db, alpha_peak_hz=pk_hz))
    sel = [s["ic"] for s in stats if abs(s["r_veog"]) > ic["abs_r_thr"] or abs(s["r_heog_partial"]) > ic["abs_r_thr"]]
    blink_sub = [s["ic"] for s in stats if s["map_max"] in ("FP1", "FP2", "AF3", "AF4")
                 and abs(s["r_veog"]) > ic["blink_subspace_r"]]
    cand = sorted(set(sel) | set(blink_sub))
    vetoed = [k for k in cand if stats[k]["alpha_peak_db"] >= ic["alpha_veto_db"]
              and _is_central_parietal(stats[k]["map_max"])]
    cand = [k for k in cand if k not in vetoed]
    cand = sorted(cand, key=lambda k: -max(abs(stats[k]["r_veog"]), abs(stats[k]["r_heog_partial"])))[:ic["max_remove"]]
    # analysis copy in the delivered band, CAR over the ICA channels
    Xa = dsp.bandpass(Zd, fs, *cfg["band"], order=cfg["filter_order"], pad_s=cfg["pad_s"])
    Za = Xa[[ix[c] for c in keep]]
    Za = Za - Za.mean(0, keepdims=True)
    raw_an = to_raw(Za, keep, fs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl = ica.apply(raw_an.copy(), exclude=list(cand), verbose="error").get_data() * 1e6
    art = Za - cl
    art -= art.mean(1, keepdims=True)
    art_lp = dsp.lowpass(art, fs, ic["lp_hz"], order=4, pad_s=cfg["pad_s"])
    Y = np.zeros_like(Xa)
    Y[[ix[c] for c in keep]] = Za - art_lp
    Y = robust_car(apply_interp(Y, ch, M), ch, car_sel)
    out = dict(removed_ics=[stats[k] for k in cand], vetoed_ics=[stats[k] for k in vetoed], ic_stats=stats,
               n_iter=int(getattr(ica, "n_iter_", -1)), max_iter=int(ic["max_iter"]), X_ica=Y)
    if X_reg is not None:
        dec = [ix[c] for c in cfg["decoder_channels"] if c in ix and c not in drop]
        from scipy.signal import welch
        f, Pr = welch(X_reg[dec][:, :nc], fs, nperseg=int(fs), nfft=int(4 * fs))
        Pi = welch(Y[dec][:, :nc], fs, nperseg=int(fs), nfft=int(4 * fs))[1]
        bands = {"1-7": (1, 7), "8-13": (8, 13), "13-20": (13, 20), "20-30": (20, 30), "30-40": (30, 40)}
        out["ica_over_regression_power_dec_median"] = {
            b: float(np.median(Pi[:, (f >= lo) & (f < hi)].sum(1) / Pr[:, (f >= lo) & (f < hi)].sum(1)))
            for b, (lo, hi) in bands.items()}
        d = (Y - X_reg)[dec][:, :nc]
        out["rms_difference_uV_dec"] = {ch[i]: float(np.sqrt((r ** 2).mean())) for i, r in zip(dec, d)}
        if out["n_iter"] >= out["max_iter"]:
            out["note"] = "ICA stopped at max_iter (common on short single trials); compare, do not adopt"
    return out
