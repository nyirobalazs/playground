"""Fixed before/after figure set for one cleaned trial.

``standard_set(D, res, outdir)`` writes:

    01_overview_traces.png    decoder channels, reference-matched raw (grey) vs cleaned (blue), masks shaded
    02_motion_mask_rule.png   the three motion-rule components, thresholds and the resulting mask timeline
    03_spectra.png            decoder-median spectra (raw / cleaned / removed) and per-band power ratios
    04_ocular_correction.png  regression gain maps, variance removed per channel, blink-locked averages
    05_decoder_epoch.png      the decoder epoch: traces, masks and the fully valid feature windows
    06_feature_usability.png  channel x band usability matrix (status colour + text label)
    07_channel_qc.png         electrical distance of neighbour pairs (bridges) and per-channel HF / beta floors

Palette: reference/raw grey #9a9893, cleaned blue #2a78d6, removed / attention orange #eb6834, third series aqua
#1baf7a; status good #0ca30c / warning #fab219 / critical #d03b3b always with a text label; diverging maps run
blue - grey - red.  "Raw" in every panel is the reference-matched raw (same detrend, band and robust-CAR set as the
cleaned output, no correction), so differences are due to the correction alone.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import dsp  # noqa: E402

GREY, BLUE, ORANGE, AQUA = "#9a9893", "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, MUTED, GRID = "#222222", "#6b6a66", "#e6e5e1"
STATUS = {"use": GOOD, "caution": WARN, "exclude": CRIT}
STATUS_INK = {"use": "white", "caution": INK, "exclude": "white"}
CMAP_DIV = LinearSegmentedColormap.from_list("blue_grey_red", ["#104281", BLUE, "#9ec5f4", "#d9d8d4", "#f2a8a5",
                                                              "#e34948", "#9c1f1e"])
BAD_COL = {"BAD_motion": ORANGE, "BAD_transient": CRIT, "BAD_next_trial": MUTED}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})


# ------------------------------------------------------------------------------------------------ helpers
def _save(fig, path, dpi=120):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _ann_spans(res, prefix):
    on, du, de = res["ann"]
    return [(float(a), float(a + d), str(s)) for a, d, s in zip(on, du, de) if str(s).startswith(prefix)]


def _shade_bad(ax, res, xlim=None, alpha=0.16):
    for a, b, s in _ann_spans(res, "BAD"):
        if xlim is not None and (b < xlim[0] or a > xlim[1]):
            continue
        ax.axvspan(a, b, color=BAD_COL.get(s, ORANGE), alpha=alpha, lw=0, zorder=0)


def _events(ax, D):
    t_end = D["n_crop"] / D["fs"] + D.get("di_offset", 0.0) + 0.01
    for k, v in D.get("events_di", {}).items():
        if D["t"][0] <= v <= min(D["t"][-1], t_end):
            ax.axvline(v, color=MUTED, lw=0.7, ls="--", zorder=1)
    on = D.get("move_onset_di", np.nan)
    if np.isfinite(on):
        ax.axvline(on, color=INK, lw=1.1, zorder=3)


def _stack(ax, t, Y, spacing, labels, color, lw, clip=1.5, zorder=2):
    n = Y.shape[0]
    offs = (n - 1 - np.arange(n)) * spacing
    for i in range(n):
        ax.plot(t, np.clip(Y[i], -clip * spacing, clip * spacing) + offs[i], color=color, lw=lw, zorder=zorder)
    ax.set_yticks(offs)
    ax.set_yticklabels(labels)
    ax.set_ylim(-1.7 * spacing, offs[0] + 1.7 * spacing)
    return offs


def _scalebar(ax, x, y0, size, unit="uV"):
    ax.plot([x, x], [y0, y0 + size], color=INK, lw=1.6, clip_on=False, zorder=6)
    ax.text(x, y0 + size / 2, f" {size:g} {unit}", va="center", ha="left", fontsize=8, clip_on=False)


def _ref_raw(D, res):
    from .pipeline import reference_raw
    return reference_raw(D, res["cfg"], res["car_set"])


def _demean(X):
    return X - np.median(X, 1, keepdims=True)


def _dec(D, res):
    ch = list(D["ch"])
    return [c for c in res["cfg"]["decoder_channels"] if c in ch]


def _mask_legend():
    return [Patch(color=ORANGE, alpha=0.35, label="BAD_motion (masked)"),
            Patch(color=CRIT, alpha=0.35, label="BAD_transient"),
            Patch(color=MUTED, alpha=0.35, label="BAD_next_trial"),
            Line2D([], [], color=AQUA, marker="v", ls="none", label="EYE_blink (flag, not masked)"),
            Line2D([], [], color=INK, lw=1.2, label="reach onset (DI axis)")]


def _event_list(D, xlim, merge_s=0.25):
    """Markers on the EEG axis inside xlim, near-simultaneous ones merged, nothing after the crop marker."""
    ev = sorted(D.get("events_di", {}).items(), key=lambda kv: kv[1])
    t_end = D["n_crop"] / D["fs"] + D.get("di_offset", 0.0) + 0.01
    out = []
    for k, v in ev:
        if not (xlim[0] <= v <= xlim[1]) or v > t_end:
            continue
        k = k.replace("trial_type_", "type: ")
        if out and v - out[-1][0] < merge_s:
            out[-1] = (out[-1][0], out[-1][1] + " + " + k)
        else:
            out.append((v, k))
    return out


def _event_strip(ax, D, res, xlim):
    """Thin axis above a time-series panel: marker names (staggered), blinks and the reach onset."""
    evs = _event_list(D, xlim)
    span = xlim[1] - xlim[0]
    levels, last = [0.78, 0.48], [-np.inf, -np.inf]
    for v, k in evs:
        ax.axvline(v, color=MUTED, lw=0.7, ls="--")
        need = 0.012 * span * len(k) * 0.55
        lv = 0 if v - last[0] > need else 1
        last[lv] = v
        ax.text(v, levels[lv], " " + k, ha="left", va="center", fontsize=7.5, color=MUTED)
    on = D.get("move_onset_di", np.nan)
    if np.isfinite(on) and xlim[0] <= on <= xlim[1]:
        ax.axvline(on, color=INK, lw=1.1)
        ax.text(on, 0.18, " reach onset (DI)", ha="left", va="center", fontsize=7.5, color=INK)
    for b in res["blinks"]:
        if xlim[0] <= b <= xlim[1]:
            ax.plot([b], [0.06], marker="v", color=AQUA, ms=6, clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(*xlim)
    ax.axis("off")


# ------------------------------------------------------------------------------------------------ figures
def fig_overview(D, res, path, spacing=60.0):
    t, ch = D["t"], list(D["ch"])
    dec = _dec(D, res)
    idx = [ch.index(c) for c in dec]
    R = _demean(_ref_raw(D, res))[idx]
    X = _demean(res["X"])[idx]
    labels = [c + (" (re-estimated)" if c in res["drop"] else "") for c in dec]
    xlim = (t[0], t[-1])
    fig, (st, ax) = plt.subplots(2, 1, figsize=(15, 10.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[0.55, 10], hspace=0.02))
    _event_strip(st, D, res, xlim)
    _shade_bad(ax, res)
    _stack(ax, t, R, spacing, labels, GREY, 0.8, zorder=1)
    _stack(ax, t, X, spacing, labels, BLUE, 0.6, zorder=2)
    _events(ax, D)
    e0, e1 = D["decoder_epoch_di"]
    yl = ax.get_ylim()
    if np.isfinite(e0):
        ax.annotate("", xy=(e0, yl[0] + 0.3 * spacing), xytext=(e1, yl[0] + 0.3 * spacing),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
        ax.text((e0 + e1) / 2, yl[0] + 0.45 * spacing, "decoder epoch", ha="center", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.set_xlim(*xlim)
    _scalebar(ax, t[-1] + 0.06, yl[0] + 0.2 * spacing, spacing)
    ax.set_xlabel("time (s, physical EEG axis)")
    st.set_title(f"Decoder channels, {res['cfg']['band'][0]:g}-{res['cfg']['band'][1]:g} Hz, robust CAR: "
                 f"reference-matched raw (grey) vs cleaned (blue); offset {spacing:g} uV, clipped at "
                 f"+-{1.5 * spacing:g} uV", loc="left", pad=34)
    h = [Line2D([], [], color=GREY, lw=1.5, label="raw (same band / reference, no correction)"),
         Line2D([], [], color=BLUE, lw=1.5, label="cleaned")] + _mask_legend()
    st.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=7, frameon=False, fontsize=7.5)
    return _save(fig, path)


def fig_motion_rule(D, res, path):
    mo, mc, t = res["motion"], res["cfg"]["motion"], D["t"]
    nc = int(D["n_crop"])
    xlim = (t[0], t[-1])
    fig, axs = plt.subplots(5, 1, figsize=(14, 11), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.45, 1, 1, 1, 0.9], hspace=0.32))
    _event_strip(axs[0], D, res, xlim)
    axs[0].set_title("Rule-based motion mask on the physical EEG axis (shading = final BAD spans)", loc="left",
                     pad=6, fontsize=11)
    lg = dict(loc="upper left", frameon=True, framealpha=0.95, edgecolor="none", ncol=1)
    a = axs[1]
    _shade_bad(a, res)
    a.plot(t, mo["head_speed_eeg"], color=INK, lw=1.0, label="Unity head angular speed (+DI)")
    a.axhline(mc["head_thr_deg_s"], color=CRIT, lw=1, ls="--", label=f"threshold {mc['head_thr_deg_s']:g} deg/s")
    for s0, s1 in mo["spans"]["head_speed_padded"]:
        a.axvspan(s0, s1, ymin=0.93, ymax=1.0, color=INK, alpha=0.6, lw=0)
    a.set_ylabel("deg/s")
    a.set_title(f"(1) head speed > {mc['head_thr_deg_s']:g} deg/s, padded {mc['head_pad_before_s']:g} s before / "
                f"{mc['head_pad_after_s']:g} s after (dark bar = component span)", loc="left")
    a.legend(**lg)
    b = axs[2]
    _shade_bad(b, res)
    b.plot(t[:nc], mo["hf_db"][:nc], color=BLUE, lw=0.8, label="global 55-95 Hz envelope\n(cleaned, robust-CAR median)")
    b.axhline(mc["hf_thr_db"], color=CRIT, lw=1, ls="--", label=f"threshold +{mc['hf_thr_db']:g} dB")
    for s0, s1 in mo["spans"]["hf_padded"]:
        b.axvspan(s0, s1, ymin=0.93, ymax=1.0, color=BLUE, alpha=0.7, lw=0)
    b.set_ylabel(f"dB re P{mc['hf_quiet_pct']:g}")
    b.set_title(f"(2) broadband HF (EMG / cable) envelope > +{mc['hf_thr_db']:g} dB over its quiet level, padded "
                f"{mc['hf_pad_s']:g} s (blue bar = component span)", loc="left")
    b.set_ylim(top=max(b.get_ylim()[1], mc["hf_thr_db"]) * 1.25)
    b.legend(**lg)
    c = axs[3]
    _shade_bad(c, res)
    c.plot(t[:nc], mo["alpha_ratio"][:nc], color=AQUA, lw=1.0, label="7.5-10.5 Hz central in-phase envelope")
    c.axhline(mc["alpha_thr_x"], color=CRIT, lw=1, ls="--", label=f"threshold {mc['alpha_thr_x']:g}x")
    for s0, s1 in mo["spans"]["alpha_padded"]:
        c.axvspan(s0, s1, ymin=0.93, ymax=1.0, color=AQUA, alpha=0.8, lw=0)
    c.set_ylabel("x quiet level")
    c.set_title("(3) ~9 Hz burst envelope: " + ("APPLIED" if mo["applied_alpha"] else
                                                "computed for QA, NOT applied (it would also flag genuine rest mu)"),
                loc="left")
    c.set_ylim(top=c.get_ylim()[1] * 1.1)
    c.legend(**lg)
    d = axs[4]
    rows = [("BAD_motion", ORANGE), ("BAD_transient", CRIT), ("BAD_next_trial", MUTED), ("EYE_blink", AQUA)]
    for k, (name, col) in enumerate(rows):
        spans = [(s0, s1 - s0) for s0, s1, _ in _ann_spans(res, name)]
        if spans:
            d.broken_barh(spans, (len(rows) - 1 - k - 0.35, 0.7), color=col, alpha=0.85 if name != "EYE_blink" else 0.55)
        if name == "BAD_transient" and not spans:
            d.text(xlim[0] + 0.05, len(rows) - 1 - k, "none", va="center", fontsize=8, color=MUTED)
    d.set_yticks(range(len(rows)))
    d.set_yticklabels([r[0] for r in rows][::-1])
    d.set_ylim(-0.6, len(rows) - 0.4)
    d.set_title("(4) resulting annotations: BAD_* are excluded from features; EYE_blink only flags trust", loc="left")
    d.set_xlabel("time (s, physical EEG axis)")
    for ax in axs[1:]:
        _events(ax, D)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlim(*xlim)
    return _save(fig, path)


def fig_spectra(D, res, path):
    from scipy.signal import welch
    fs, ch, nc = D["fs"], list(D["ch"]), int(D["n_crop"])
    dec = [c for c in _dec(D, res) if c not in res["drop"]]
    idx = [ch.index(c) for c in dec]
    R = _ref_raw(D, res)[idx][:, :nc]
    X = res["X"][idx][:, :nc]
    f, Pr = welch(R, fs, nperseg=int(2 * fs))
    Px = welch(X, fs, nperseg=int(2 * fs))[1]
    Pm = welch(R - X, fs, nperseg=int(2 * fs))[1]
    m = (f >= 0.5) & (f <= 45)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.25))
    for lo, hi, lab in ((1, 7, "ocular band"), (8, 13, "mu"), (13, 30, "beta")):
        a.axvspan(lo, hi, color=GRID, alpha=0.6 if lab == "mu" else 0.35, lw=0)
        a.text(np.sqrt(lo * hi), 0.985, lab, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8,
               color=MUTED)
    a.loglog(f[m], np.median(Pr, 0)[m], color=GREY, lw=2.0, label="raw (reference-matched)")
    a.loglog(f[m], np.median(Px, 0)[m], color=BLUE, lw=1.4, label="cleaned")
    a.loglog(f[m], np.median(Pm, 0)[m], color=ORANGE, lw=1.4, label="removed (raw - cleaned)")
    top = float(np.median(Pr, 0)[m].max())
    a.set_ylim(top=top * 60)
    lp = res["cfg"]["ocular"]["lp_hz"]
    a.axvline(lp, color=INK, lw=0.8, ls=":")
    a.text(lp, 0.03, f"correction band edge {lp:g} Hz ", transform=a.get_xaxis_transform(), fontsize=8, rotation=90,
           va="bottom", ha="right")
    a.set_xticks([1, 2, 5, 10, 20, 40])
    a.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel("median PSD over decoder channels (uV$^2$/Hz)")
    a.set_title(f"Decoder-channel median spectra, trial crop ({len(dec)} channels, re-estimates excluded)", loc="left")
    a.legend(frameon=False, loc="lower left")
    a.grid(axis="y", color=GRID, lw=0.5)
    pr = res["qa"]["power_ratio_vs_reference_raw"] if "qa" in res else None
    if pr:
        names = list(pr)
        vals = [pr[k]["median"] for k in names]
        y = np.arange(len(names))[::-1]
        b.axvspan(0.95, 1.05, color=GRID, alpha=0.9, lw=0, zorder=0)
        b.barh(y, vals, color=[ORANGE if k == "1-7" else BLUE for k in names], height=0.6, zorder=2)
        b.axvline(1.0, color=INK, lw=1, zorder=3)
        for yy, v in zip(y, vals):
            b.text(max(v, 1.05) + 0.02, yy, f"{v:.3f}", va="center", fontsize=9)
        b.set_yticks(y)
        b.set_yticklabels([f"{k} Hz" for k in names])
        b.set_xlim(0, 1.3)
        b.set_xlabel("cleaned / raw power, decoder median (trial crop)")
        b.set_title("Band-power ratio (grey band = within 5 % of raw)", loc="left")
        b.text(0.0, -0.2, "orange = ocular band (intended reduction); blue = must stay at 1", transform=b.transAxes,
               fontsize=8, color=MUTED)
    return _save(fig, path)


def fig_ocular(D, res, path):
    import mne
    from .bridging import to_raw
    ch, fs = list(D["ch"]), D["fs"]
    model = res["model"]
    corr = [c for c in ch if c in model["corrected"]]
    ci = [ch.index(c) for c in corr]
    info = to_raw(np.zeros((len(corr), 2)), corr, fs).info
    fig = plt.figure(figsize=(15, 9.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42)
    top = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 0.05, 0.42, 2.3], wspace=0.12)
    bot = outer[1].subgridspec(1, 4, wspace=0.3)
    vmax = float(np.abs(model["B"][:, ci]).max()) or 1.0
    for k, r in enumerate(model["regressors"][:2]):
        ax = fig.add_subplot(top[0, k])
        im, _ = mne.viz.plot_topomap(model["B"][k, ci], info, axes=ax, show=False, cmap=CMAP_DIV,
                                     vlim=(-vmax, vmax), contours=0, sensors=True)
        ax.set_title(f"{r} gain", fontsize=10)
    cax = fig.add_subplot(top[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("uV per uV of proxy (blue < 0 < red)")
    ax = fig.add_subplot(top[0, 4])
    R = _ref_raw(D, res)
    nc = int(D["n_crop"])
    frac = ((R - res["X"])[:, :nc]).var(1) / R[:, :nc].var(1)
    cols = [GREY if (c in res["drop"] or c in model["poles"]) else ORANGE for c in ch]
    ax.bar(range(len(ch)), [frac[ch.index(c)] for c in ch], color=cols, width=0.75)
    for k, c in enumerate(ch):
        if c in model["poles"]:
            ax.text(k, 0.02, "EOG proxy", rotation=90, ha="center", va="bottom", fontsize=6.5, color=MUTED)
        elif c in res["drop"]:
            ax.text(k, frac[k] + 0.02, "re-est.", rotation=90, ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(len(ch)))
    ax.set_xticklabels(ch, rotation=90, fontsize=7.5)
    for tl, c in zip(ax.get_xticklabels(), ch):
        if c in res["cfg"]["decoder_channels"]:
            tl.set_fontweight("bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("var(removed) / var(raw)")
    ax.set_title("Variance removed per channel, 0.5-45 Hz (bold = decoder channel)", loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.5)
    show = [c for c in ("F3", "Fz", "Cz", "Pz") if c in ch]
    hw = int(round(0.5 * fs))
    tt = np.arange(-hw, hw + 1) / fs
    Rd, Xd = _demean(R), _demean(res["X"])
    peaks = [int(round(b * fs)) for b in res["blinks"] if hw <= b * fs < D["X"].shape[1] - hw - 1]
    for k, c in enumerate(show):
        ax = fig.add_subplot(bot[0, k])
        i = ch.index(c)
        if peaks:
            ra = np.mean([Rd[i, p - hw:p + hw + 1] for p in peaks], 0)
            xa = np.mean([Xd[i, p - hw:p + hw + 1] for p in peaks], 0)
            ax.plot(tt, ra, color=GREY, lw=2, label="raw")
            ax.plot(tt, xa, color=BLUE, lw=1.4, label="cleaned")
            ax.plot(tt, ra - xa, color=ORANGE, lw=1.2, label="removed")
        ax.axvline(0, color=MUTED, lw=0.6, ls="--")
        ax.set_title(f"{c}: blink-locked mean (n={len(peaks)})", fontsize=10)
        ax.set_xlabel("time from blink peak (s)")
        if k == 0:
            ax.set_ylabel("uV")
            ax.legend(frameon=False, loc="upper left", fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.suptitle(f"Ocular regression: gains fitted on {model.get('n_fit_s') or 0:.1f} s of calibration data, "
                 f"subtracted below {model['lp_hz']:g} Hz only (residual > {model['lp_hz']:g} Hz blink energy stays on "
                 "frontal channels)", x=0.01, ha="left", fontsize=11, y=0.97)
    return _save(fig, path)


def fig_decoder_epoch(D, res, path, spacing=40.0):
    t, ch = D["t"], list(D["ch"])
    e0, e1 = D["decoder_epoch_di"]
    if not np.isfinite(e0):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "no reach onset (no VR file or no hand kinematics): decoder epoch undefined",
                ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    pad = 0.4
    w = (t >= e0 - pad) & (t <= e1 + pad)
    xlim = (float(t[w][0]), float(t[w][-1]))
    dec = _dec(D, res)
    idx = [ch.index(c) for c in dec]
    R = _demean(_ref_raw(D, res))[idx][:, w]
    X = _demean(res["X"])[idx][:, w]
    labels = [c + (" (re-est.)" if c in res["drop"] else "") for c in dec]
    fig, axs = plt.subplots(3, 1, figsize=(14, 11.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.55, 8, 1.6], hspace=0.05))
    _event_strip(axs[0], D, res, xlim)
    ax = axs[1]
    _shade_bad(ax, res, xlim)
    _stack(ax, t[w], R, spacing, labels, GREY, 0.9, zorder=1)
    _stack(ax, t[w], X, spacing, labels, BLUE, 0.8, zorder=2)
    _events(ax, D)
    yl = ax.get_ylim()
    ax.annotate("", xy=(e0, yl[0] + 0.3 * spacing), xytext=(e1, yl[0] + 0.3 * spacing),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.text((e0 + e1) / 2, yl[0] + 0.45 * spacing, "decoder epoch", ha="center", va="bottom", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    _scalebar(ax, xlim[1] + 0.02, yl[0] + 0.2 * spacing, spacing)
    axs[0].set_title(f"Decoder epoch {e0:.3f}-{e1:.3f} s around the DI-corrected reach onset "
                     f"({D['move_onset_di']:.3f} s): raw grey, cleaned blue; offset {spacing:g} uV", loc="left", pad=34)
    axs[0].legend(handles=[Line2D([], [], color=GREY, lw=1.5, label="raw"),
                           Line2D([], [], color=BLUE, lw=1.5, label="cleaned")] + _mask_legend(),
                  loc="lower left", bbox_to_anchor=(0, 1.0), ncol=7, frameon=False, fontsize=7.5)
    b = axs[2]
    _shade_bad(b, res, xlim)
    ft = res.get("features")
    if ft is not None and ft["n_windows"]:
        ws = res["cfg"]["features"]["win_s"]
        for k, (s0, ok) in enumerate(zip(ft["starts"], ft["window_valid"])):
            b.plot([s0, s0 + ws], [k, k], color=GOOD if ok else CRIT, lw=2.4 if ok else 0.9, alpha=1 if ok else 0.45,
                   solid_capstyle="butt")
        b.set_ylim(-1, len(ft["starts"]) * 1.45)
        b.set_yticks([])
        n_ok = int(ft["window_valid"].sum())
        b.set_ylabel(f"{ws:g}-s\nwindows")
        b.text(0.005, 0.97, f"{n_ok}/{len(ft['starts'])} feature windows fully valid: green = usable; red = overlaps a "
                            "BAD span (features set to NaN)", transform=b.transAxes, ha="left", va="top", fontsize=8.5,
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    _events(b, D)
    b.set_xlim(*xlim)
    b.set_xlabel("time (s, physical EEG axis)")
    return _save(fig, path)


def fig_usability(D, res, path):
    rows = res.get("usability") or []
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "usability not computed", ha="center")
        return _save(fig, path)
    chans = list(dict.fromkeys(r["channel"] for r in rows))
    bands = list(dict.fromkeys(r["band"] for r in rows))
    bhz = {r["band"]: r["band_hz"] for r in rows}
    st = {(r["channel"], r["band"]): r["status"] for r in rows}
    fig, ax = plt.subplots(figsize=(1.25 * len(bands) + 3.2, 0.36 * len(chans) + 1.8))
    for i, c in enumerate(chans):
        for j, b in enumerate(bands):
            s = st[(c, b)]
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=STATUS[s]))
            ax.text(j + 0.48, i + 0.46, s, ha="center", va="center", fontsize=8, color=STATUS_INK[s],
                    fontweight="bold" if s == "use" else "normal")
    ax.set_xlim(0, len(bands))
    ax.set_ylim(len(chans), 0)
    ax.set_xticks(np.arange(len(bands)) + 0.48)
    ax.set_xticklabels([f"{b}\n{bhz[b][0]:g}-{bhz[b][1]:g} Hz" for b in bands])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(chans)) + 0.46)
    ax.set_yticklabels([c + (" (re-est.)" if c in res["drop"] else "") for c in chans])
    for s in ("left", "bottom", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    n = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS}
    ax.set_xlabel(f"use {n['use']}  |  caution {n['caution']}  |  exclude {n['exclude']}   (reasons per cell in "
                  "feature_usability.csv)", fontsize=8)
    ax.set_title("Feature usability per decoder channel and band", loc="left", pad=34)
    return _save(fig, path)


def fig_channel_qc(D, res, path):
    from matplotlib.ticker import FuncFormatter
    from .bridging import electrical_distance
    br = res["bridges"]
    qc = res.get("qa", {}).get("channel_qc")
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.55, wspace=0.2)
    a = fig.add_subplot(gs[:, 0])
    pairs = electrical_distance(D, br["band_hz"])
    eds = np.array([p[0] for p in pairs])
    x = np.arange(len(eds))
    bad = eds < br["threshold_uV2"]
    a.scatter(x[~bad], eds[~bad], s=12, color=GREY, label="neighbour pairs")
    a.scatter(x[bad], eds[bad], s=28, color=CRIT, label="bridged (critical)")
    a.axhline(br["threshold_uV2"], color=CRIT, lw=1, ls="--")
    a.text(len(eds) * 0.98, br["threshold_uV2"] * 1.15, f"threshold {br['threshold_uV2']:g} uV$^2$", ha="right",
           va="bottom", fontsize=8, color=CRIT)
    a.set_yscale("log")
    lo = max(eds.min(), 1e-3)
    ys = np.geomspace(lo * 1.1, br["threshold_uV2"] * 0.55, max(int(bad.sum()), 1))
    for n, k in enumerate(np.where(bad)[0]):
        a.annotate(f"critical: {pairs[k][1]}-{pairs[k][2]} {eds[k]:.2f} uV$^2$", (x[k], eds[k]),
                   xytext=(len(eds) * 0.18, ys[n]), textcoords="data", fontsize=8, color=CRIT, va="center",
                   arrowprops=dict(arrowstyle="-", color=CRIT, lw=0.6))
    a.set_ylim(lo * 0.6, eds.max() * 2)
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_xlabel("neighbour pairs, sorted by electrical distance")
    a.set_ylabel(f"electrical distance (uV$^2$, {br['band_hz'][0]:g}-{br['band_hz'][1]:g} Hz)")
    gap = br.get("gap_ratio_next_pair")
    a.set_title("Gel bridges: var(xi - xj) of neighbours" + (f"\n(next pair {gap:g}x above the bridged maximum)"
                                                                if gap else ""), loc="left", fontsize=10)
    a.legend(frameon=False, loc="upper left")
    a.grid(axis="y", color=GRID, lw=0.5)
    if qc:
        chs = list(qc["channels"])
        xx = np.arange(len(chs))
        hf = np.array([qc["channels"][c]["hf_30_150_rel_median"] for c in chs])
        rb = np.array([qc["channels"][c]["ratio_20_30_over_60_90"] for c in chs])
        for row, (vals, bad_if, col_ok, col_bad, lab_ok, lab_bad, ylab, title) in enumerate((
                (hf, hf > 2, BLUE, CRIT, "<= 2x median", "critical: > 2x median (EMG / poor contact)",
                 "30-150 Hz power / median", "High-frequency floor per channel (robust CAR, trial crop)"),
                (rb, rb < 2, AQUA, WARN, ">= 2 (beta above the EMG floor)", "warning: < 2 (beta at the EMG floor)",
                 "20-30 / 60-90 Hz power", "Beta-to-HF ratio per channel"))):
            ax = fig.add_subplot(gs[row, 1])
            ax.bar(xx, vals, width=0.7, color=[col_bad if f else col_ok for f in bad_if])
            ax.axhline(2.0, color=INK, lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xticks(xx)
            ax.set_xticklabels(chs, rotation=90, fontsize=7.5)
            for tl, c in zip(ax.get_xticklabels(), chs):
                if c in res["cfg"]["decoder_channels"]:
                    tl.set_fontweight("bold")
            ax.set_ylabel(ylab)
            ax.set_title(title + "; bold = decoder; dashed = 2", loc="left", fontsize=10)
            ax.legend(handles=[Patch(color=col_ok, label=lab_ok), Patch(color=col_bad, label=lab_bad)],
                      frameon=True, framealpha=0.95, edgecolor="none", fontsize=7.5, loc="upper right", ncol=2)
            ax.set_ylim(top=vals.max() * 3)
            ax.grid(axis="y", color=GRID, lw=0.5)
    return _save(fig, path)


def standard_set(D, res, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = lambda s: os.path.join(outdir, s)  # noqa: E731
    out = [fig_overview(D, res, p("01_overview_traces.png")),
           fig_motion_rule(D, res, p("02_motion_mask_rule.png"))]
    if "qa" in res:
        out.append(fig_spectra(D, res, p("03_spectra.png")))
    out += [fig_ocular(D, res, p("04_ocular_correction.png")),
            fig_decoder_epoch(D, res, p("05_decoder_epoch.png")),
            fig_usability(D, res, p("06_feature_usability.png")),
            fig_channel_qc(D, res, p("07_channel_qc.png"))]
    return out
