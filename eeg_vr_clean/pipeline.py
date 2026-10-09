"""End-to-end cleaning: single trial and session mode.

Single trial
    1. detrend (per-channel median, linear fit on the trial crop) and 0.5-45 Hz zero-phase band-pass;
    2. gel-bridge detection (electrical distance) - drop one member per bridged group;
    3. robust CAR; band-limited (< 7 Hz) EOG regression with gains fitted on calibration spans
       (or loaded from a calibration file);
    4. spherical-spline re-estimate of the dropped duplicates from the corrected channels; robust CAR again;
    5. the same chain on a 0.5-150 Hz copy (X_broadband);
    6. rule-based masks (BAD_motion, BAD_transient, BAD_next_trial), EYE_blink flags and the reach_onset_DI marker;
    7. QA (ocular leakage, band-power preservation, rank, channel QC), masked window features and usability flags;
       optional band-limited ICA cross-check.

Session
    Bridges are decided once on the session-median electrical distances, gains are fitted once on the concatenated
    calibration samples of the calibration trials (or a dedicated calibration block, or loaded from a calibration
    file), then frozen and applied per trial.  Every trial keeps its own masks and DI offset.
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np

from . import __version__, dsp
from . import bridging as BR
from . import features as FT
from . import masks as MK
from . import ocular as OC
from . import qc as QC
from .config import merge_config
from .reference import car_set, robust_car


def _band(b, fs):
    lo, hi = float(b[0]), float(b[1])
    return (lo, min(hi, 0.45 * fs))


def _cfg(cfg):
    return merge_config(None, cfg) if cfg is None or "band" not in cfg else cfg


def reference_raw(D, cfg, car_sel, band=None):
    """Reference-matched raw: same detrend / band / robust-CAR set as the cleaned output, no correction."""
    Xb, _ = OC.prep(D, _band(cfg["band"] if band is None else band, D["fs"]), cfg)
    return robust_car(Xb, D["ch"], car_sel)


def run_trial(D, cfg=None, *, model=None, bridges=None, fit_spans=None, ica_check=None, qa=True):
    """Clean one trial (dict from :func:`eeg_vr_clean.io.load_trial`).

    model     : gain model from :func:`fit_gains` / :func:`load_calibration`; None = fit on this trial.
    bridges   : bridge decision (e.g. session-level); None = detect on this trial.
    fit_spans : calibration spans [[t0, t1], ...] (EEG axis, s) when fitting on this trial; None = cfg / auto rule.
    """
    cfg = _cfg(cfg)
    t0 = time.time()
    fs, t, ch, nc = D["fs"], D["t"], list(D["ch"]), int(D["n_crop"])
    br = BR.detect_bridges(D, cfg, decoder=cfg["decoder_channels"], car_base=cfg["reference"]["car_set"],
                           poles=cfg["ocular"]["poles"]) if bridges is None else bridges
    drop = list(br["drop"])
    car_sel = car_set(ch, cfg["reference"]["car_set"], drop)
    if model is None:
        model = OC.fit_gains([D], cfg, car_sel, drop, None if fit_spans is None else [fit_spans])
    else:
        if list(model["ch"]) != ch:
            raise ValueError("gain model channel list differs from the trial's")
        if sorted(model["drop"]) != sorted(drop):
            warnings.warn(f"gain model was fitted with dropped {model['drop']} but this trial drops {drop}; "
                          "using the model's drop list")
            drop = list(model["drop"])
            car_sel = car_set(ch, cfg["reference"]["car_set"], drop)
    M = BR.interp_operator(ch, drop, exclude=cfg["ocular"]["poles"])
    X, removed, E_lp = OC.apply_chain(D, _band(cfg["band"], fs), model, cfg, M, car_sel)
    Xbb, _, _ = OC.apply_chain(D, _band(cfg["band_broadband"], fs), model, cfg, M, car_sel)
    mo = MK.motion_masks(D, X, Xbb, cfg, car_sel)
    tw, W = MK.transient_windows(X, D, cfg)
    bt, bamp, _ = MK.blink_events(D, cfg)
    on, du, de = MK.build_annotations(D, mo["spans"]["BAD_motion"], tw, bt, cfg)
    valid = ~MK.bad_mask(t, on, du, de)
    ch_types = np.array(["eog" if c in cfg["ocular"]["poles"] else "eeg" for c in ch], dtype="<U3")
    res = dict(cfg=cfg, X=X, X_broadband=Xbb, removed=removed, E_lp=E_lp, model=model, drop=drop, bridges=br,
               interp=M, car_set=car_sel, motion=mo, transient=tw, transient_W=W, blinks=bt, blink_amp=bamp,
               ann=(on, du, de), valid=valid, ch_types=ch_types, rank=dsp.numerical_rank(X[:, :nc]),
               version=__version__)
    res["runtime_core_s"] = round(time.time() - t0, 2)
    if qa:
        res["qa"] = trial_qa(D, res, cfg)
        res["features"] = FT.epoch_features(D, res, cfg)
        res["usability"] = FT.usability(D, res, cfg)
    ic = cfg["ica_check"]["enabled"] if ica_check is None else ica_check
    if ic:
        res["ica_check"] = OC.ica_crosscheck(D, cfg, drop, M, car_sel, X_reg=X)
    return res


def trial_qa(D, res, cfg):
    fs, t, nc = D["fs"], D["t"], int(D["n_crop"])
    dec = [c for c in cfg["decoder_channels"] if c in D["ch"]]
    R = reference_raw(D, cfg, res["car_set"])
    e0, e1 = D["decoder_epoch_di"]
    ep = (t >= e0) & (t < e1)
    v = res["valid"]
    return dict(
        ocular_leakage=QC.ocular_leakage({"reference_raw": R, "cleaned": res["X"]}, D, dec),
        power_ratio_vs_reference_raw=QC.band_power_ratios(res["X"], R, D, dec),
        channel_qc=QC.channel_qc(D, cfg, res["bridges"]),
        masked_s_in_trial=round(float((~v[:nc]).sum() / fs), 3),
        valid_s_in_trial=round(float(v[:nc].sum() / fs), 3),
        decoder_epoch_s=[round(float(e0), 4), round(float(e1), 4)] if np.isfinite(e0) else None,
        usable_epoch_s=round(float(v[ep].sum() / fs), 3) if ep.any() else None,
        usable_epoch_fraction=round(float(v[ep].mean()), 3) if ep.any() else None,
        usable_epoch_runs_s=[[round(a, 3), round(b, 3)] for a, b in dsp.runs(v & ep, t)] if ep.any() else [],
        output_rank=res["rank"],
        expected_rank=len(D["ch"]) - 1 - len(res["drop"]),
    )


# ------------------------------------------------------------------------------------------------ session
def run_session(trials, cfg=None, *, calibration=None, calib_trials=None, calib_spans=None, ica_check=None):
    """Clean several trials with ONE bridge decision and ONE frozen gain model.

    calibration  : path to a gain file (``save_calibration``) or a model dict; overrides fitting.
    calib_trials : list of trial dicts used only for fitting (e.g. a dedicated blink/saccade calibration block);
                   default = the session trials themselves (their automatic calibration spans, concatenated).
    calib_spans  : optional list (one entry per calibration trial) of span lists.
    """
    cfg = _cfg(cfg)
    ch = list(trials[0]["ch"])
    for D in trials:
        if list(D["ch"]) != ch:
            raise ValueError("all trials must share the channel list and order")
    if isinstance(calibration, (str, os.PathLike)):
        model = OC.load_calibration(calibration, ch)
    else:
        model = calibration
    if model is not None:
        drop = list(model["drop"])
        br = BR.session_bridges(trials, cfg)
        if sorted(br["drop"]) != sorted(drop):
            warnings.warn(f"session bridge check drops {br['drop']} but the calibration file was made with {drop}; "
                          "the calibration's channel set is used - re-calibrate if the cap was re-gelled")
        br = {**br, "drop": drop, "drop_source": "calibration file"}
    else:
        br = BR.session_bridges(trials, cfg)
        drop = list(br["drop"])
        car_sel = car_set(ch, cfg["reference"]["car_set"], drop)
        fit_set = trials if calib_trials is None else calib_trials
        model = OC.fit_gains(fit_set, cfg, car_sel, drop, calib_spans)
        model["fitted_on"] = "session trials" if calib_trials is None else "calibration recordings"
    results = [run_trial(D, cfg, model=model, bridges=br, ica_check=ica_check) for D in trials]
    di = [D["di_offset"] for D in trials]
    summary = dict(n_trials=len(trials), bridges=_jsonable_bridges(br), drop=drop,
                   gains_fitted_on=model.get("fitted_on", model.get("source", "supplied model")),
                   n_fit_s=model.get("n_fit_s"),
                   di_offset_ms=dict(per_trial=[round(x * 1e3, 3) for x in di], median=round(float(np.median(di)) * 1e3, 3),
                                     range=[round(min(di) * 1e3, 3), round(max(di) * 1e3, 3)]),
                   per_trial=[dict(trial=k, file=D["meta"].get("eeg_file"), usable_epoch_s=r["qa"]["usable_epoch_s"],
                                   masked_s=r["qa"]["masked_s_in_trial"],
                                   n_valid_epoch_windows=r["features"]["n_valid_windows"],
                                   n_epoch_windows=r["features"]["n_windows"])
                              for k, (D, r) in enumerate(zip(trials, results))])
    return dict(model=model, bridges=br, results=results, summary=summary)


def _jsonable_bridges(br):
    return {k: v for k, v in br.items() if k != "all_pairs"}


# ------------------------------------------------------------------------------------------------ toolkit handoff
def export_decoder_arrays(res, D, channels=None, fs_out=250.0):
    """Decoder-ready arrays for toolkits that assume a fixed sampling rate (e.g. a 250 Hz band-power decoder).

    Channels are selected BY NAME (positional indexing silently picks the wrong channels on caps whose order
    differs from the toolkit's).  X is resampled with a polyphase anti-alias filter; ``valid`` is down-sampled
    conservatively (a sample is valid only if every source sample it covers is valid); the reach onset is given
    as a sample index on the DI-corrected axis (target-onset codes are NOT reach onsets)."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    ch = list(D["ch"])
    channels = [c for c in (res["cfg"]["decoder_channels"] if channels is None else channels)]
    missing = [c for c in channels if c not in ch]
    if missing:
        raise ValueError(f"channels not in the recording: {missing}")
    idx = [ch.index(c) for c in channels]
    fr = Fraction(float(fs_out) / round(D["fs"])).limit_denominator(1000)
    Y = resample_poly(res["X"][idx], fr.numerator, fr.denominator, axis=1)
    n = Y.shape[1]
    tt = np.arange(n) / fs_out
    bad = ~res["valid"]
    t = D["t"]
    pos = np.searchsorted(t, tt, side="left")
    nxt = np.searchsorted(t, tt + 1.0 / fs_out, side="left")
    cs = np.r_[0, np.cumsum(bad)]
    valid = (cs[np.minimum(nxt, len(t))] - cs[np.minimum(pos, len(t))]) == 0
    on = D.get("move_onset_di", np.nan)
    excluded = [c for c in channels if c in res.get("drop", [])]
    return dict(X=Y, ch=np.array(channels), fs=float(fs_out), valid=valid,
                reach_onset_sample=int(round(on * fs_out)) if np.isfinite(on) else -1,
                reestimated_channels=np.array(excluded, dtype="<U8"))


# ------------------------------------------------------------------------------------------------ outputs
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def save_npz(res, D, path):
    on, du, de = res["ann"]
    np.savez(path, X=res["X"].astype(np.float64), fs=np.float64(D["fs"]), ch=np.array(list(D["ch"])),
             ann_onset=on, ann_duration=du, ann_desc=de, ch_types=res["ch_types"], valid=res["valid"].astype(bool),
             X_broadband=res["X_broadband"].astype(np.float64))
    return path


def save_fif(res, D, path):
    import mne
    ch = list(D["ch"])
    raw = BR.to_raw(res["X"], ch, D["fs"])
    names = raw.ch_names
    raw.set_channel_types({names[i]: "eog" for i, c in enumerate(ch) if res["ch_types"][i] == "eog"},
                          verbose="error")
    on, du, de = res["ann"]
    raw.set_annotations(mne.Annotations(onset=on, duration=du, description=list(de)), verbose="error")
    raw.info["bads"] = [names[ch.index(c)] for c in res["drop"]]
    raw.info["description"] = (
        f"eeg_vr_clean {__version__}: {res['cfg']['band'][0]:g}-{res['cfg']['band'][1]:g} Hz, robust CAR, "
        f"< {res['cfg']['ocular']['lp_hz']:g} Hz EOG regression. EOG-typed channels are uncorrected proxies. Channels "
        "marked bad are spline re-estimates of bridged sites (not independent). Honour BAD_* annotations before "
        "feature extraction; EYE_blink is informational.")
    raw.save(path, overwrite=True, verbose="error")
    return path


def _write_csv(rows, path, cols):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return path


def info_dict(res, D):
    cfg, mo, model = res["cfg"], res["motion"], res["model"]
    on, du, de = res["ann"]
    ch = list(D["ch"])
    info = dict(
        package="eeg_vr_clean", version=__version__, config=cfg, meta=D.get("meta", {}),
        timing=dict(fs=D["fs"], n_samples=int(D["X"].shape[1]), n_crop=int(D["n_crop"]),
                    di_offset_s=D["di_offset"], di_report=D.get("di_report"),
                    reach_onset_di_s=D.get("move_onset_di"), decoder_epoch_di_s=list(D["decoder_epoch_di"])),
        bridging=_jsonable_bridges(res["bridges"]), dropped_and_reestimated=res["drop"], robust_car_set=res["car_set"],
        interpolation_weights={b: {c: round(float(w), 4) for c, w in zip(m[2], m[1])} for b, m in res["interp"].items()},
        ocular_model=dict(regressors=model["regressors"], lp_hz=model["lp_hz"], n_fit_s=model.get("n_fit_s"),
                          calibration=model.get("calibration"), source=model.get("source", model.get("fitted_on")),
                          ridge={k: v for k, v in model.get("ridge", {}).items() if k != "R2"},
                          gains={r: {c: round(float(model["B"][k, i]), 5) for i, c in enumerate(ch)}
                                 for k, r in enumerate(model["regressors"])}),
        ch_types={c: str(tp) for c, tp in zip(ch, res["ch_types"])},
        masks=dict(spans=mo["spans"], validation_coverage=mo["validation_coverage"],
                   alpha_component_applied=mo["applied_alpha"], hf_quiet_level_uV=mo["hf_level"],
                   alpha_quiet_level_uV=mo["alpha_level"], transient_windows=res["transient"],
                   blinks_s=[round(float(x), 3) for x in res["blinks"]],
                   blink_peak_veog_uV=[round(float(x), 1) for x in res["blink_amp"]]),
        annotations=[dict(onset=round(float(a), 4), duration=round(float(b), 4), desc=str(c)) for a, b, c in zip(on, du, de)],
        output_rank=res["rank"], runtime_core_s=res["runtime_core_s"],
    )
    if "qa" in res:
        info["qa"] = res["qa"]
        info["features"] = {k: v for k, v in res["features"].items()
                            if k in ("n_windows", "n_valid_windows", "baseline", "channels")}
        info["usability_counts"] = {s: sum(1 for r in res["usability"] if r["status"] == s)
                                    for s in ("use", "caution", "exclude")}
    if "ica_check" in res:
        info["ica_check"] = {k: v for k, v in res["ica_check"].items() if k not in ("X_ica", "ic_stats")}
    return _jsonable(info)


def save_trial_outputs(res, D, out_dir, figures=True, decoder_fs=None):
    """cleaned.npz, cleaned_raw.fif, info.json, gains.json, features.csv, feature_usability.csv, figures/ and,
    with ``decoder_fs``, decoder_<fs>hz.npz (decoder channels by name, resampled)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = [save_npz(res, D, os.path.join(out_dir, "cleaned.npz")),
             save_fif(res, D, os.path.join(out_dir, "cleaned_raw.fif")),
             OC.save_calibration(res["model"], os.path.join(out_dir, "gains.json"))]
    if decoder_fs:
        ex = export_decoder_arrays(res, D, fs_out=float(decoder_fs))
        pth = os.path.join(out_dir, f"decoder_{float(decoder_fs):g}hz.npz")
        np.savez(pth, **ex)
        paths.append(pth)
    if "features" in res:
        paths.append(_write_csv(res["features"]["rows"], os.path.join(out_dir, "features.csv"),
                                ["window_start_s", "window_end_s", "window_valid", "channel", "band", "power_db",
                                 "rel_baseline_db"]))
        paths.append(_write_csv(res["usability"], os.path.join(out_dir, "feature_usability.csv"),
                                ["channel", "band", "status", "reason"]))
    if figures:
        from . import figures as FG
        res["figure_paths"] = FG.standard_set(D, res, os.path.join(out_dir, "figures"))
        paths += res["figure_paths"]
    info = info_dict(res, D)
    info["outputs"] = [os.path.basename(p) if os.path.dirname(p) == out_dir else os.path.relpath(p, out_dir)
                       for p in paths]
    with open(os.path.join(out_dir, "info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return paths + [os.path.join(out_dir, "info.json")]
