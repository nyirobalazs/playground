"""Robust common-average reference and small Laplacians.

Why a *robust* CAR: a full 32-channel average includes the frontal poles, so every blink and saccade is spread with
inverted sign onto every channel (in the development recording the Cz blink gain flipped from +0.16 to -0.10 under
a full CAR).  Averaging only over channels without large ocular, temporal-EMG or reference-adjacent content, and
without bridged duplicates (which would be counted twice), lowers absolute ocular leakage instead (median |b_VEOG|
over the decoder channels fell from 0.17 to 0.06 in the same recording) and removes most of the common-mode,
reference-borne high-frequency bursts.
"""
from __future__ import annotations

import numpy as np

from .config import LAPLACIANS, ROBUST_CAR20


def car_set(ch, base=None, drop=()):
    """Robust-CAR channel set: ``base`` (default ROBUST_CAR20) restricted to present channels, minus ``drop``."""
    base = ROBUST_CAR20 if base is None else base
    return [c for c in base if c in ch and c not in set(drop)]


def robust_car(Y, ch, sel=None):
    """Re-reference every row of ``Y`` (n_ch, N) to the mean of the channels in ``sel`` (default ROBUST_CAR20)."""
    ch = list(ch)
    sel = ROBUST_CAR20 if sel is None else sel
    ix = [ch.index(c) for c in sel]
    return Y - Y[ix].mean(0, keepdims=True)


def laplacian(Y, ch, centre, neighbours=None):
    """Small surface Laplacian: centre minus the mean of its neighbours (default :data:`LAPLACIANS`)."""
    ch = list(ch)
    nb = LAPLACIANS[centre] if neighbours is None else neighbours
    return Y[ch.index(centre)] - Y[[ch.index(c) for c in nb]].mean(0)
