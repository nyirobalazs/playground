"""Gel-bridge detection (electrical distance) and spherical-spline re-estimation of the dropped duplicates.

Electrical distance (ED) = variance of the difference of two neighbouring channels (uV^2) in a 1-30 Hz band.
Bridged pairs carry the same signal (ED well below 1 uV^2), while normal neighbours are tens of uV^2 apart (in
the development recording the four bridged pairs were at 0.2-0.3 uV^2 and the next pair at 16-21 uV^2).  One member
of each bridged group is dropped BEFORE any covariance-based step (regression, ICA, whitening) - otherwise the rank
deficit is silent (numerical rank stays full) - and re-estimated by spherical spline only at the very end.  The
re-estimate is not an independent measurement: flag it and exclude it from features.

Keeper rule (in order): decoder channel > robust-CAR member > member whose spline-predicted site correlates best with
the shared (pair-mean) signal.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import dsp
from .config import DECODER_17, EOG_POLES, MNE_NAMES, ROBUST_CAR20


# ------------------------------------------------------------------------------------------------ MNE helpers
def montage():
    """Standard 10-20 positions (MNE >= 1.13 calls this montage 'colin27_1020'; identical positions)."""
    import mne
    names = mne.channels.get_builtin_montages()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return mne.channels.make_standard_montage("colin27_1020" if "colin27_1020" in names else "standard_1020")


def to_raw(X, ch, fs, bads=()):
    """MNE RawArray (X in microvolts) with standard 10-20 positions."""
    import mne
    names = [MNE_NAMES.get(c, c) for c in ch]
    info = mne.create_info(names, fs, ch_types="eeg")
    raw = mne.io.RawArray(np.asarray(X, float) * 1e-6, info, verbose="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.set_montage(montage(), verbose="error")
    raw.info["bads"] = [MNE_NAMES.get(b, b) for b in bads]
    return raw


def neighbour_pairs(ch):
    """Delaunay neighbour pairs (MNE ``find_ch_adjacency``) as index tuples (i < j)."""
    import mne
    raw = to_raw(np.zeros((len(ch), 2)), ch, 500.0)
    with mne.utils.use_log_level("error"):
        adj, _ = mne.channels.find_ch_adjacency(raw.info, "eeg")
    adj = adj.toarray()
    return [(i, j) for i in range(len(ch)) for j in range(i + 1, len(ch)) if adj[i, j]]


def interp_weights(target, sources):
    """Spherical-spline weights (len(sources),) estimating ``target`` from ``sources`` (MNE, public API): a Raw
    whose data is the identity is interpolated, so the interpolated row is the weight vector."""
    names = list(sources) + [target]
    n = len(names)
    raw = to_raw(np.eye(n) * 1e6, names, 500.0, bads=[target])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.interpolate_bads(reset_bads=True, verbose="error")
    return raw.get_data()[n - 1, :n - 1]


def interp_operator(ch, drop, exclude=()):
    """{dropped channel: (source indices, weights, source names)}.  Sources = all channels except the dropped ones
    and ``exclude`` (the uncorrected EOG poles, so no raw ocular signal leaks into a re-estimate)."""
    ch = list(ch)
    excl = set(drop) | set(exclude)
    src = [c for c in ch if c not in excl]
    out = {}
    for b in drop:
        w = interp_weights(b, src)
        out[b] = ([ch.index(c) for c in src], w, src)
    return out


def apply_interp(Y, ch, M):
    Y = np.array(Y, float, copy=True)
    ch = list(ch)
    for b, (si, w, _) in M.items():
        Y[ch.index(b)] = w @ Y[si]
    return Y


# ------------------------------------------------------------------------------------------------ detection
def detrended(D):
    """Per-channel median removal + linear detrend fitted on the trial crop and applied to the full record."""
    Z = D["X"] - np.median(D["X"], axis=1, keepdims=True)
    nc = int(D["n_crop"])
    tt = np.arange(Z.shape[1], dtype=float)
    p = np.polyfit(tt[:nc], Z[:, :nc].T, 1)
    return Z - (np.outer(p[0], tt) + p[1][:, None]), p


def electrical_distance(D, band=(1.0, 30.0), order=4, pad_s=3.0):
    """ED (uV^2) for every neighbour pair over the trial crop -> sorted list of (ed, a, b)."""
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    Zd, _ = detrended(D)
    Bf = dsp.bandpass(Zd, fs, *band, order=order, pad_s=pad_s)[:, :nc]
    pairs = [(float(np.var(Bf[i] - Bf[j])), ch[i], ch[j]) for i, j in neighbour_pairs(ch)]
    return sorted(pairs)


def _components(bridged):
    comp = []
    for a, b, _ in bridged:
        hit = [g for g in comp if a in g or b in g]
        g = set([a, b]).union(*hit) if hit else {a, b}
        comp = [x for x in comp if x not in hit] + [g]
    return comp


def detect_bridges(D, cfg=None, *, pairs=None, decoder=None, car_base=None, poles=None):
    """Data-driven bridge detection and keeper decision.

    ``pairs`` may be supplied (e.g. session-median EDs from :func:`session_bridges`); otherwise they are computed
    on this trial.  Returns a dict with the bridged pairs, connected groups, the keep/drop decision with its evidence,
    and ``drop`` (channels to remove before the ocular step, in channel order)."""
    from .reference import robust_car
    bc = (cfg or {}).get("bridge", {"band": [1.0, 30.0], "thr_uV2": 5.0})
    order, pad_s = (cfg or {}).get("filter_order", 4), (cfg or {}).get("pad_s", 3.0)
    decoder = DECODER_17 if decoder is None else decoder
    car_base = ROBUST_CAR20 if car_base is None else car_base
    poles = EOG_POLES if poles is None else poles
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    ix = {c: i for i, c in enumerate(ch)}
    pairs = electrical_distance(D, bc["band"], order, pad_s) if pairs is None else sorted(pairs)
    thr = bc["thr_uV2"]
    bridged = [(a, b, ed) for ed, a, b in pairs if ed < thr]
    comp = _components(bridged)
    decisions, drop = [], []
    if comp:
        Zd, _ = detrended(D)
        sel = [c for c in car_base if c in ch]
        Xr = robust_car(dsp.bandpass(Zd, fs, *bc["band"], order=order, pad_s=pad_s), ch, sel)
        excl_src = set(poles) | set(c for g in comp for c in g)
        src = [c for c in ch if c not in excl_src]
        for g in comp:
            members = sorted(g, key=lambda c: ch.index(c))
            shared = Xr[[ix[c] for c in members]].mean(0)[:nc]
            rows = []
            for m in members:
                w = interp_weights(m, src)
                pred = (w @ Xr[[ix[c] for c in src]])[:nc]
                rows.append(dict(ch=m, decoder=m in decoder, car_member=m in car_base,
                                 r_site_vs_shared=round(dsp.corr(pred, shared), 4)))
            best = sorted(rows, key=lambda d: (d["decoder"], d["car_member"], d["r_site_vs_shared"]), reverse=True)[0]
            n_dec = sum(r["decoder"] for r in rows)
            n_car = sum(r["car_member"] for r in rows if r["decoder"] == best["decoder"])
            why = ("decoder channel" if best["decoder"] and n_dec == 1 else
                   "robust-CAR member" if best["car_member"] and n_car == 1 else "best r(site, shared)")
            decisions.append(dict(members=members, keep=best["ch"], drop=[m for m in members if m != best["ch"]],
                                  rule_applied=why, evidence=rows))
            drop += [m for m in members if m != best["ch"]]
    drop = [c for c in ch if c in drop]
    nb = [p for p in pairs if p[0] >= thr]
    gap = (nb[0][0] / max(b[2] for b in bridged)) if (bridged and nb) else None
    return dict(threshold_uV2=thr, band_hz=list(bc["band"]), n_neighbour_pairs=len(pairs),
                pairs_lowest=[dict(a=a, b=b, ed_uV2=round(ed, 3)) for ed, a, b in pairs[:10]],
                bridged=[dict(a=a, b=b, ed_uV2=round(ed, 3)) for a, b, ed in bridged],
                components=decisions, drop=drop,
                gap_ratio_next_pair=None if gap is None else round(float(gap), 1))


def session_bridges(trials, cfg=None):
    """Session-level bridge decision: per-pair median ED across trials, then the same threshold/keeper rule
    (evaluated on the first trial).  One drop list for the whole session keeps channel sets identical."""
    bc = (cfg or {}).get("bridge", {"band": [1.0, 30.0], "thr_uV2": 5.0})
    order, pad_s = (cfg or {}).get("filter_order", 4), (cfg or {}).get("pad_s", 3.0)
    per = [{(a, b): ed for ed, a, b in electrical_distance(D, bc["band"], order, pad_s)} for D in trials]
    keys = set(per[0]).intersection(*per[1:]) if len(per) > 1 else set(per[0])
    med = [(float(np.median([p[k] for p in per])), k[0], k[1]) for k in keys]
    res = detect_bridges(trials[0], cfg, pairs=med)
    res["per_trial_ed_of_bridged_uV2"] = {f"{d['a']}-{d['b']}": [round(p[(d['a'], d['b'])], 3) for p in per]
                                          for d in res["bridged"]}
    res["n_trials"] = len(trials)
    return res
