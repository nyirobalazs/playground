# eeg_vr_clean

Artefact handling for EEG recorded while a participant reaches for targets in virtual reality (head-mounted
display + 32-channel wireless cap). The package turns one trial, or a whole session, into a cleaned array with
explicit masks, decoder-ready band-power features and a per-channel, per-band usability table, plus a fixed set of
before/after figures.

It is deliberately conservative. It removes what can be removed without touching the decoder bands, and masks
what cannot be removed. Everything else is flagged, not "cleaned".

> **Evidence base.** The design choices below were made on one 12.9-s executed-reach development trial
> (32 channels, 500 Hz) and checked by an independent scorer and three adversarial reviews. Every number quoted
> here is a descriptive single-trial point estimate. The thresholds (bridge distance, motion rule, usability
> ratios) need to be confirmed at session level before they are treated as validated.

No recordings, no derived arrays and no figures of real data are part of this repository. The tests use
synthetic data only.

---

## What it does

| step | module | what | why (measured on the development trial) |
|---|---|---|---|
| 1 | `io` | Loads the trial HDF5 (EEG, markers, amplifier DI byte) and the Unity VR JSON. Measures the DI-trigger latency and computes head kinematics on the Unity clock. | The amplifier's DI pulses lagged the software markers by 77.5 ± 0.6 ms (n = 5), so host-clock events and kinematics must be shifted onto the physical EEG axis. Differentiating the host-timestamped head pose gave spurious 150 deg/s spikes (timestamp bunching); the Unity clock gives the true 44 deg/s peak. |
| 2 | `ocular.prep` | Per-channel median, linear detrend (fitted on the trial crop), zero-phase Butterworth 0.5-45 Hz. Crop at the trial-reset marker. | Common-mode drift was about 9 µV/s. The next trial's markers start within 0.4 s of the reset. Line noise was low and the data were not notched (50 Hz bin 1.5x its flanks), so no notch: the 45 Hz low-pass suffices. |
| 3 | `bridging` | Electrical distance (variance of the 1-30 Hz difference) for every neighbour pair. Pairs below 5 µV² are bridged. One member is dropped before any covariance step and re-estimated by spherical spline at the end. | Four bridged pairs measured 0.17-0.23 µV², against 21 µV² for the next pair (a 90x gap). The effective rank was 28 while the numerical rank reported 32, so the deficit has to be imposed explicitly. |
| 4 | `reference` | Robust common average over 20 channels: no frontal poles, no T7/T8, no bridged duplicates. | A full 32-channel CAR spread blinks with inverted sign (Cz blink gain +0.16 → −0.10). The robust set lowered the decoder-median blink gain from 0.17 to 0.06 and removed most of the common-mode HF bursts. |
| 5 | `ocular` | Ridge regression of every channel's **< 7 Hz** signal on the < 7 Hz VEOG/HEOG proxies (from the untouched recording). Only the < 7 Hz part of `gain × proxy` is subtracted. Gains come from calibration spans or a calibration file and are then frozen. FP1/FP2/F7/F8 stay uncorrected and are typed `eog`. | Decoder 1-7 Hz power fell to 0.26x, while 13-20/20-30/30-40 Hz stayed at 1.000x. Leakage on proxies the method never saw (AF-based) fell from 0.126 to 0.010. Why the band limit, and why not ICA: see below. |
| 6 | `bridging` | Spline re-estimate of the dropped duplicates from the corrected channels only (never from the raw EOG poles), then the robust CAR again. | The re-estimate is not a measurement. FC6 rebuilt from F4/T8/C4 carried about 20-28 % more beta than the bridged original. It is excluded from features. |
| 7 | `masks` | Rule-based `BAD_motion`, `BAD_transient` and `BAD_next_trial` on the DI-corrected axis. `EYE_blink` flags (not bad). `reach_onset_DI` marker. | See "Masks" below. |
| 8 | `features`, `qc` | Masked log band-power windows. Usability flags per channel × band. Leakage, band-power preservation, rank and channel QC. | Ignoring the masks inflated the epoch mu feature by 3-4 dB relative to fixation. |
| 9 | `figures` | Fixed figure set (7 PNGs). | So that every run can be checked by eye on identical axes. |

Output rank is `n_channels − 1 (CAR) − n_dropped`. Band-limited subtraction does not reduce rank.

### Why regression below 7 Hz, and not ICA

* **Never subtract ocular components broadband.** Removing two ocular ICs with a plain `ica.apply` inflated
  decoder 13-30 Hz power 1.3-2.2x and 30-40 Hz power 1.3-2.1x, in nine of nine fit variants. The removed projection
  is anticorrelated with the remaining signal in the beta band.
* **Never subtract EOG proxies (or ocular IC back-projections) above 7 Hz.**
  * The proxies are electrode differences, so they also carry brain rhythms. F8 − F7 at 8-13 Hz tracked the C4 mu
    rhythm (r 0.66-0.93), and VEOG carried posterior alpha through its negative pole.
  * Subtracting the proxies up to 12 Hz removed 30-48 % of quiet-window frontal mu.
  * Subtracting ocular-IC back-projections up to 12 Hz produced an ERD-like −1.9 dB C4 decrease during the reach,
    with no eye event present.
  * Below 7 Hz, the same correction moved none of 48 mu contrasts by more than 1 dB.
* **ICA is fragile on a single short trial.**
  * The ratio k = N/n² was 27 at 15 components.
  * Extended infomax never converged.
  * Selecting the second blink component depended on a z-score of 3.02 against a threshold of 3.0.
  * The output changed by 8 % with a change in filter padding alone.
* **Regression suits session use.** It is deterministic, has no seed, runs online, and its gains can be calibrated
  once per session.
* **The cost is known.** The part of each blink above 7 Hz stays on F3/Fz/F4. Blinks were not task-locked, so this
  adds noise rather than label leakage. The `EYE_blink` annotations and the usability table flag it.

An optional **band-limited ICA cross-check** (`--ica-check`) is kept as QA, never as the output. It runs extended
infomax on the bridge-free channels and selects ocular components by |r| > 0.6 with the proxies, plus a
blink-subspace rule with margin: FP/AF map maximum and |r_VEOG| > 0.4. It applies an alpha veto and subtracts only the
< 7 Hz back-projection. It then reports how its band power and rms differ from the regression output.

### Masks (rule-based, nothing trial-specific hard-coded)

* `BAD_motion` is the union of two rules:
  * Unity head angular speed (shifted by the DI offset) above 10 deg/s, padded 0.1 s before and 0.4 s after;
  * the global 55-95 Hz envelope (median over the robust-CAR channels of the cleaned 0.5-150 Hz copy, 100-ms
    boxcar) more than +6 dB above its quiet 25th percentile, padded 0.1 s.

  **Why mask instead of clean.** High-amplitude ~9 Hz bursts were locked to the head turn (peak about 25 ms after
  peak head speed, once DI-corrected) and to the arm landing. They formed an in-phase fronto-central field at
  20-25 µV, 4-8x the fixation level. They survived CAR, Laplacian, EOG regression and ICA. The burst component was
  unstable across ICA fits (map r 0.36-0.70) and carried genuine alpha. Whether their origin is mechanical (HMD,
  strap, cable) or cortical cannot be decided from one trial. Removing them would either leave them in or fabricate
  an ERD, so they are masked.

  **Why the 0.4 s pad.** The burst tail ran about 0.25 s past the end of the turn.

  **Why the cleaned robust-CAR HF source.** On native-reference data the HF rule also fired on a 97 % common-mode
  event that the robust CAR removes.

  A third component, a 7.5-10.5 Hz central envelope above 3x its quiet level, is computed and plotted but not
  applied by default: it also fires on genuine resting mu.
* `BAD_transient`: 250-ms windows where any decoder channel exceeds |x| > 100 µV (0.5 Hz high-pass, cleaned array).
* `BAD_next_trial`: from the trial-reset marker to the end of the record.
* `EYE_blink`: ±150 ms around blinks detected on the raw VEOG. These are informational, not bad.

### Usability rules (`feature_usability.csv`)

Each decoder channel × band (`eog_low` 0.5-7, `mu` 8-13, `beta_low` 13-20, `beta_high` 20-30, `gamma` 30-40 Hz) gets
`use`, `caution` or `exclude`, with a reason. The rules:

* **Re-estimated (bridged) channels:** `exclude` in every band.
* **`eog_low` and `gamma`:** `exclude` everywhere by default.
  * Eye-head-coupled activity in the movement epoch cannot be certified as ocular-free: a hold-out proxy test sat
    inside its own null distribution.
  * The head-posture slow shift is not ocular.
  * Saccadic spike potentials and EMG bursts sit above 30 Hz.
* **`mu`:** `caution` where the VEOG gain is ≥ 0.10 (blink energy above 7 Hz remains), where the local blink
  contrast is ≥ +3 dB, or where a blink falls inside the valid decoder epoch and the channel's local blink contrast is
  ≥ +1 dB.
* **Beta ratios are measured where and how the features are computed:** on the fully valid 0.5-s Hann windows of the
  decoder epoch, falling back to all valid samples only when no such window exists. Trial-wide ratios are dominated
  by rest and fixation and overstated beta quality in the moving posture 1.1-9.4x on the development trial. On
  sub-second spans the ratio also varied 2-6x between estimators, so the feature estimator itself is used.
* **Too little data cannot be certified:** if the valid decoder-epoch time is shorter than two independent windows
  (`min_independent_windows`), no cell is better than `caution`.
* **`beta_low`:** `exclude` where the 13-20/60-90 Hz ratio is < 1.5 (white HF floor). `caution` where it is < 3, or
  with a blink or posture effect.
* **`beta_high`:** `exclude` where the 20-30/60-90 Hz ratio is < 2 (EMG or contact floor).
  `caution` where it is < 3.
* **Values within ±10 % of a threshold are indeterminate:**
  * at an exclusion threshold they become `exclude`;
  * at a caution threshold they become `caution`.
* **Overrides:** `config["usability"]["policy"]` can override any cell.

An adversarial review of the development trial showed that ratios computed over all valid samples (mostly rest)
overstated beta usability in the movement epoch, and that a blink inside the only valid epoch span inflated mu at
central and parietal sites. The rules above encode both corrections.

---

## Install

```bash
pip install -r requirements.txt        # pinned, tested versions (Python 3.13)
# or: pip install -e .                 # uses pyproject.toml
```

Run it as a module (`python -m eeg_vr_clean ...`). The package has a submodule named `io`, so do not run its files
as scripts.

## Quick start

```bash
# setup check right after donning the HMD (bridges, HF floor, line noise)
python -m eeg_vr_clean qc --eeg trial.hdf5 --out qc/

# one trial (gains from automatic calibration spans of this trial)
python -m eeg_vr_clean run --eeg trial.hdf5 --vr trial_vr.json --out out/trial --decoder-fs 250

# session: fit gains once on a dedicated calibration block, then apply them to every trial
python -m eeg_vr_clean calibrate --eeg calib.hdf5 --vr calib_vr.json --out gains.json
python -m eeg_vr_clean session --eeg t01.hdf5 t02.hdf5 --vr t01.json t02.json --calibration gains.json --out out/session
```

Other options:

* `--fit-spans 0.2-4.4,8.5-12.4`: explicit calibration spans, in seconds on the EEG axis.
* `--di-offset 0.0775`: a fixed DI latency instead of the per-trial measurement.
* `--config cfg.json`: a deep-merged override of `eeg_vr_clean.config.DEFAULT_CONFIG`.
* `--ica-check`: run the ICA cross-check.
* `--no-figs`: skip the figures.

```python
from eeg_vr_clean import load_trial, run_trial, run_session, save_trial_outputs
D = load_trial("trial.hdf5", "trial_vr.json")          # di_offset='auto' (measured from the DI codes)
res = run_trial(D)                                      # or run_trial(D, model=load_calibration("gains.json"))
save_trial_outputs(res, D, "out/trial")
```

## Inputs

* **Trial HDF5:**
  * datasets `eeg` (samples × channels, µV), `lsl_timestamps`, `markers` (fields `sample_index`, `lsl_timestamp`,
    `code`, `label`), and optionally `di_byte`, `acc`, `counter`, `link_quality`;
  * attributes `fs_hz`, `channel_labels` and `layout`.
* **VR JSON:**
  * `head_coordinates.samples[*]`: `t_us_since_trial_start`, `qx`, `qy`, `qz`, `qw` (Unity clock);
  * `hand_position[*]`: `host_ts_ns`, `x`, `y`, `z`;
  * optional `touch_latency_s`.
* **Marker labels used:** `trial_start` (anchors the Unity clock), the first label starting with `target_on`, and
  `trial_reset` (crop). All three are configurable.
* **Other formats:** build the trial dict with `eeg_vr_clean.io.trial_from_arrays`.

## Outputs (per trial)

| file | content |
|---|---|
| `cleaned.npz` | `X` (n_ch × N, µV, 0.5-45 Hz, robust CAR), `X_broadband` (0.5-150 Hz), `fs`, `ch`, `ch_types` (`eog` for the uncorrected proxies), `valid` (False inside any `BAD_*`), `ann_onset/ann_duration/ann_desc` |
| `cleaned_raw.fif` | Same as `X`, with annotations. EOG proxies are typed `eog`; re-estimated channels are marked bad. |
| `gains.json` | The frozen gain model (re-usable with `--calibration`). |
| `features.csv` | 0.5-s Hann-window log band power per decoder channel and band over the decoder epoch. NaN when the window touches a `BAD_*` span. Also given relative to the valid fixation baseline. |
| `feature_usability.csv` | Status and reason per channel × band. |
| `decoder_250hz.npz` | (`--decoder-fs 250`) Decoder channels selected by name, polyphase-resampled. Conservative validity. Reach onset as a sample index. |
| `info.json` | Config, DI report, bridges and keeper decisions, gains, interpolation weights, mask components, annotations, QA (leakage, band-power ratios, channel QC, usable epoch) and optional ICA cross-check. |
| `figures/` | `01_overview_traces`, `02_motion_mask_rule`, `03_spectra`, `04_ocular_correction`, `05_decoder_epoch`, `06_feature_usability`, `07_channel_qc` |

Session mode also writes `session_gains.json` and `session_summary.json`. These hold the per-trial DI offset,
usable epoch, masked time and valid windows.

## Acquisition and protocol recommendations

1. **Check for bridges after donning the HMD.** Run `python -m eeg_vr_clean qc` once the headset is on.
   * The strap sits over the cap. Neighbouring electrodes with electrical distance below 5 µV² share one signal.
   * Re-gel before recording. Each bridge costs one independent channel, and on the development cap one of them
     was a decoder channel.
2. **Give blinks their own windows.** The blink rate was about 5x normal (about 91/min; short partial blinks, possibly
   HMD-related).
   * Schedule explicit blink breaks between trials.
   * Instruct the participant to hold fixation without blinking during the cue-to-reach interval.
3. **Use a delayed reach.** Target-directed saccades and the head turn began at target onset and fell inside the
   pre-movement decoder window: after masking, only 0.15 s of pre-movement data remained.
   * Present the target, then give a separate go cue after ≥ 1 s.
   * Orienting then happens before the pre-movement window.
4. **Keep targets within ±15° of straight ahead,** so they can be foveated with the eyes alone, and instruct the
   participant to keep the head still.
   * A ~12° yaw turn (peak 44 deg/s) produced the motion bursts that removed most of the epoch.
   * Log head yaw to verify compliance.
5. **Add a calibration block** at session start and after breaks: ≥ 60 s of blinks plus horizontal and vertical
   saccades to fixed targets at ±15°.
   * Fit the EOG gains there (`calibrate`) and apply them to every trial.
   * Gains refitted on different single-trial spans replicated for VEOG (map |r| ≥ 0.999) but less well for HEOG
     (0.94), because a single trial has few horizontal saccades.
   * Also record ≥ 60 s of quiet rest if any ASR-type method is to be calibrated. 3 s of clean data is not enough.
6. **Epoch on the DI trigger codes,** not on the software markers.
   * The latency is constant within a setup but non-zero.
   * Avoid sending codes less than about 25 ms apart: the trigger line serialises them, so the later code is delayed.
   * Log a kinematic reach onset. A target-onset code is not a reach onset.
7. **Never remove ocular ICs broadband.** If ICA is used, subtract only the < 7 Hz back-projection.
8. **Never subtract EOG proxies above 7 Hz.**
9. **Enforce the masks in the feature extractor.** Use `valid`, or the NaN features. The motion bursts are still
   present in `X` and the downstream decoder has no annotation handling.

### Pitfalls of the eegbci decoder toolkit

* **Sampling rate.** `eegbci/eog.py` hard-codes 250 Hz, both for its epoch indices and for its Welch `fs`.
  * A 500 Hz file fed as-is gives −0.3..+0.7 s epochs.
  * Its "8-13 Hz" bin is then really 16-26 Hz.
  * Fix: resample to 250 Hz first (`--decoder-fs 250`), or pass the true rate.
* **Channel selection.** Select the 17 decoder channels **by name**, never by index. Positional indexing picked 13
  of 17 channels wrongly on this cap.
* **Reach onset.** Supply an explicit DI-corrected kinematic reach-onset marker (`reach_onset_DI`, or
  `reach_onset_sample` in the 250 Hz export). Code 2 in this protocol is the target cue, about 0.9 s before the hand
  moves.

## Known limits

* **The motion-burst origin is undetermined.**
  * Masks remove most of the movement-locked epoch.
  * Pre-movement features are not recoverable from trials with a head turn at target onset.
* **The > 7 Hz blink residual stays on frontal channels.** F3/Fz/F4 mu is therefore `caution`.
* **Posture-locked HF at contact- or EMG-prone sites is not removed.**
  * At CP5 and FC5, 30-40 Hz changed by up to −10 dB with the head turned.
  * Those bands are excluded instead.
* **No EMG removal step.** A windowed CCA was tried and was not reproducible across seeds or window grids.
* **The DI offset is measured per trial** (median over the matched codes). Run with `--di-offset` fixed when
  bit-exact masks across runs matter. Re-scoring under ±40 ms shifts changed only one criterion, and that one is
  decided by stop-band noise.

## Tests

```bash
python -m pytest tests -q
```

The suite uses synthetic data only. It checks that:

* injected blinks and saccades are removed below 7 Hz, with 13-40 Hz untouched to within 0.1 %;
* the true propagation gains are recovered;
* bridged pairs are detected, the keeper rule is applied, and the dropped channel is re-estimated;
* head-speed masks land on the DI-corrected axis, and the HF, transient and next-trial masks work;
* the feature extractor enforces the masks;
* session mode works: concatenated calibration, a frozen model, a calibration-file round trip, and a dedicated
  calibration recording;
* HDF5/JSON loading recovers the DI latency (with queued codes excluded) and the Unity-clock head speed;
* the CLI commands `run`, `calibrate`, `session` and `qc` work, as do the 250 Hz export and the ICA cross-check.
