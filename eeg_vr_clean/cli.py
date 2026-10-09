"""Command line interface.

    python -m eeg_vr_clean run       --eeg T.hdf5 --vr T.json --out OUT [--fit-spans 0.2-4.4,8.5-12.4]
                                     [--calibration gains.json] [--di-offset auto|0.0775] [--config cfg.json]
                                     [--ica-check] [--no-figs] [--decoder-fs 250]
    python -m eeg_vr_clean session   --eeg A.hdf5 B.hdf5 ... --vr A.json B.json ... --out OUT
                                     [--calibration gains.json | --calib-eeg C.hdf5 --calib-vr C.json]
    python -m eeg_vr_clean calibrate --eeg C.hdf5 [...] --vr C.json [...] --out gains.json [--fit-spans ...]
    python -m eeg_vr_clean qc        --eeg T.hdf5 [--vr T.json] --out OUT   (quick bridge / channel check)

Times are seconds on the physical (DI-corrected) EEG axis, measured from the first EEG sample.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _spans(s):
    if not s:
        return None
    out = []
    for part in s.split(","):
        a, b = part.strip().split("-")
        out.append([float(a), float(b)])
    return out


def _di(s):
    return s if s in (None, "auto") else float(s)


def _common(p):
    p.add_argument("--config", help="JSON file overriding the default configuration")
    p.add_argument("--di-offset", default=None, help="'auto' (default: measured from the DI codes) or seconds")


def _load(eeg, vr, cfg, di):
    from .io import load_trial
    return load_trial(eeg, vr, cfg, di_offset=di)


def cmd_run(a):
    from .config import load_config
    from .ocular import load_calibration
    from .pipeline import run_trial, save_trial_outputs
    cfg = load_config(a.config)
    D = _load(a.eeg, a.vr, cfg, _di(a.di_offset))
    model = load_calibration(a.calibration, D["ch"]) if a.calibration else None
    res = run_trial(D, cfg, model=model, fit_spans=_spans(a.fit_spans), ica_check=a.ica_check or None)
    paths = save_trial_outputs(res, D, a.out, figures=not a.no_figs, decoder_fs=a.decoder_fs)
    q = res["qa"]
    print(f"cleaned: rank {res['rank']}, dropped {res['drop']}, DI offset {D['di_offset'] * 1e3:.1f} ms, "
          f"usable epoch {q['usable_epoch_s']} s, valid windows {res['features']['n_valid_windows']}/"
          f"{res['features']['n_windows']}")
    print(f"wrote {len(paths)} files to {a.out}")
    return 0


def cmd_session(a):
    from .config import load_config
    from .pipeline import run_session, save_trial_outputs
    if len(a.eeg) != len(a.vr):
        raise SystemExit("--eeg and --vr need the same number of files")
    cfg = load_config(a.config)
    trials = [_load(e, v, cfg, _di(a.di_offset)) for e, v in zip(a.eeg, a.vr)]
    calib = None
    if a.calib_eeg:
        cv = a.calib_vr or [None] * len(a.calib_eeg)
        calib = [_load(e, v, cfg, _di(a.di_offset)) for e, v in zip(a.calib_eeg, cv)]
    ses = run_session(trials, cfg, calibration=a.calibration, calib_trials=calib,
                      calib_spans=None if not a.fit_spans else [_spans(a.fit_spans)] * len(calib or trials),
                      ica_check=a.ica_check or None)
    os.makedirs(a.out, exist_ok=True)
    from .ocular import save_calibration
    save_calibration(ses["model"], os.path.join(a.out, "session_gains.json"))
    for k, (D, r) in enumerate(zip(trials, ses["results"])):
        name = os.path.splitext(os.path.basename(a.eeg[k]))[0]
        save_trial_outputs(r, D, os.path.join(a.out, f"{k:03d}_{name}"), figures=not a.no_figs,
                           decoder_fs=a.decoder_fs)
    from .pipeline import _jsonable
    with open(os.path.join(a.out, "session_summary.json"), "w") as fh:
        json.dump(_jsonable(ses["summary"]), fh, indent=1)
    print(json.dumps(_jsonable(ses["summary"]["per_trial"]), indent=1))
    return 0


def cmd_calibrate(a):
    from .bridging import session_bridges
    from .config import load_config
    from .ocular import fit_gains, save_calibration
    from .reference import car_set
    cfg = load_config(a.config)
    vr = a.vr or [None] * len(a.eeg)
    trials = [_load(e, v, cfg, _di(a.di_offset)) for e, v in zip(a.eeg, vr)]
    br = session_bridges(trials, cfg)
    sel = car_set(trials[0]["ch"], cfg["reference"]["car_set"], br["drop"])
    spans = None if not a.fit_spans else [_spans(a.fit_spans)] * len(trials)
    model = fit_gains(trials, cfg, sel, br["drop"], spans)
    model["fitted_on"] = "calibration recordings"
    save_calibration(model, a.out)
    print(f"gains fitted on {model['n_fit_s']:.1f} s ({len(trials)} recording(s)); dropped {br['drop']}; wrote {a.out}")
    return 0


def cmd_qc(a):
    from .bridging import detect_bridges
    from .config import load_config
    from .pipeline import _jsonable
    from .qc import channel_qc
    cfg = load_config(a.config)
    D = _load(a.eeg, a.vr, cfg, _di(a.di_offset))
    br = detect_bridges(D, cfg)
    q = channel_qc(D, cfg, br)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "qc.json"), "w") as fh:
        json.dump(_jsonable(dict(bridges=br, channel_qc=q, di_report=D["di_report"])), fh, indent=1)
    if br["bridged"]:
        print("BRIDGED pairs (re-gel before recording if possible):",
              ", ".join(f"{d['a']}-{d['b']} {d['ed_uV2']:.2f} uV^2" for d in br["bridged"]))
    else:
        print("no bridged neighbour pairs")
    flagged = {c: r["flags"] for c, r in q["channels"].items() if r["flags"]}
    for c, f in flagged.items():
        print(f"  {c}: {'; '.join(f)}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m eeg_vr_clean", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="clean one trial")
    r.add_argument("--eeg", required=True)
    r.add_argument("--vr", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--fit-spans", default=None, help="calibration spans 'a-b,c-d' (s); default: automatic rule")
    r.add_argument("--calibration", default=None, help="gain file from 'calibrate' or a previous run")
    r.add_argument("--ica-check", action="store_true", help="also run the band-limited ICA cross-check (QA)")
    r.add_argument("--no-figs", action="store_true")
    r.add_argument("--decoder-fs", type=float, default=None,
                   help="also write decoder channels (by name) resampled to this rate, e.g. 250")
    _common(r)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("session", help="clean several trials with one bridge decision and one frozen gain model")
    s.add_argument("--eeg", nargs="+", required=True)
    s.add_argument("--vr", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--calibration", default=None)
    s.add_argument("--calib-eeg", nargs="+", default=None, help="dedicated calibration recordings (blink/saccade block)")
    s.add_argument("--calib-vr", nargs="+", default=None)
    s.add_argument("--fit-spans", default=None, help="spans applied to every calibration recording")
    s.add_argument("--ica-check", action="store_true")
    s.add_argument("--no-figs", action="store_true")
    s.add_argument("--decoder-fs", type=float, default=None)
    _common(s)
    s.set_defaults(func=cmd_session)
    c = sub.add_parser("calibrate", help="fit ocular gains on calibration recordings and save them")
    c.add_argument("--eeg", nargs="+", required=True)
    c.add_argument("--vr", nargs="+", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--fit-spans", default=None)
    _common(c)
    c.set_defaults(func=cmd_calibrate)
    q = sub.add_parser("qc", help="bridge / channel check (e.g. right after HMD donning)")
    q.add_argument("--eeg", required=True)
    q.add_argument("--vr", default=None)
    q.add_argument("--out", required=True)
    _common(q)
    q.set_defaults(func=cmd_qc)
    return p


def main(argv=None):
    a = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
