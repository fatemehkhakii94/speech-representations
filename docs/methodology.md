# Methodology

This document specifies the experiment represented by the committed artifacts. The canonical
machine-readable settings are in [`configs/project.yaml`](../configs/project.yaml).

## Research questions

The study asks two related questions:

1. At which representation layer can a low-capacity supervised probe recover speaker identity,
   pitch, acoustic condition, or emotion?
2. How similar are the representation spaces within and across WavLM-Large, HuBERT-Large, and
   Whisper-Large-v3, and can probing results guide a stronger downstream representation choice?

The project measures accessibility to a probe. It does not establish that an attribute is absent
from layers where a particular probe performs poorly, nor does it claim causal localization.

## Frozen encoders

| Name | Checkpoint | Revision | Transformer layers | Representations | Width |
|---|---|---|---:|---:|---:|
| WavLM-Large | `microsoft/wavlm-large` | `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` | 24 | 25 | 1,024 |
| HuBERT-Large | `facebook/hubert-large-ll60k` | `ff022d095678a2995f3c49bab18a96a9e553f782` | 24 | 25 | 1,024 |
| Whisper-Large-v3 | `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | 32 | 33 | 1,280 |

Representation 0 is the model's pre-transformer hidden state. All backbone parameters are frozen.
Audio is decoded as mono, resampled to 16 kHz, and passed through the model-specific frontend.
WavLM and HuBERT batches with unequal lengths are forwarded sample-by-sample to prevent
half-precision padding from changing an utterance's representation. Whisper uses batched log-Mel
features.

Each hidden state is converted to FP32 and mean-pooled over valid encoder frames only. The output
shape of a store is `(utterances, representation_layers, hidden_size)`.

## Cohorts and split contracts

All cohort selection and split assignment use seed 42.

### Speaker identity

- Source: LibriSpeech `train-clean-100`.
- Cohort: 24 eligible speakers, 30 utterances each, and at least three chapters per speaker.
- Target: one of 24 known-speaker classes.
- Split unit: `(speaker, chapter)`; a chapter never crosses splits.
- Interpretation: closed-set identification among known speakers using unseen chapters, not
  recognition of unseen speakers.
- Counts: 265 train, 221 validation, 234 test.
- Primary metric: unweighted average recall (UAR).

### Continuous pitch

- Source: the same 720 LibriSpeech utterances.
- Target: median voiced F0, measured with Praat through Parselmouth using a 75–500 Hz search range,
  then converted to semitones relative to 55 Hz: `12 × log2(F0 / 55)`.
- Split unit: speaker; all utterances from one speaker remain in one split.
- Counts: 480 train, 90 validation, 150 test.
- Primary metric: mean absolute error (MAE) in semitones. RMSE and R² are also recorded.

Speaker-disjoint pitch evaluation deliberately tests cross-speaker generalization. Differences in
speaker distributions can produce a larger validation/test gap and near-zero or negative R² even
when MAE improves over a naive predictor.

### Acoustic condition

- Sources: the 720-utterance LibriSpeech cohort and MUSAN.
- Conditions: clean, noise, music, and competing speech.
- Construction: every source utterance produces one clean example and three matched mixtures.
  Noise segments are deterministically sampled, DC-centered, length-matched, and mixed at an exact
  10 dB RMS SNR. Peak scaling prevents clipping, and generated mixtures are stored as lossless
  FLAC.
- Split unit: source utterance; all four variants remain together.
- Counts: 2,012 train, 432 validation, 436 test (2,880 total).
- Primary metric: accuracy. The global cohort is exactly balanced across the four conditions.

Grouping variants by source prevents a probe from seeing the clean form of a test utterance during
training.

### Emotion

- Source: CREMA-D.
- Cohort: 91 actors, capped at 24 utterances per actor with round-robin emotion sampling.
- Classes: anger, disgust, fear, happiness, neutral, and sadness.
- Split unit: actor; actors are disjoint across train, validation, and test.
- Counts: 1,296 train, 432 validation, 456 test (2,184 total).
- Primary metric: UAR. Each split contains all six classes; the validation and test sets are exactly
  class-balanced.

## Probe protocol

The default probe is one affine layer. A fresh probe is trained for every representation layer and
seed (`13`, `42`, `77`).

1. Feature mean and standard deviation are fitted on the training split only.
2. For regression, target mean and standard deviation are also fitted on training targets only.
3. AdamW minimizes cross-entropy for classification or MSE for normalized regression targets.
4. Training uses learning rate `1e-3`, batch size 256, at most 50 epochs, and validation-loss early
   stopping with patience 5.
5. The best checkpoint for a run is chosen by validation loss.
6. The representation layer is chosen by mean validation primary metric across the three seeds.
7. Test metrics are reported for that validation-selected layer without using test performance in
   selection.

Regression predictions and labels are inverse-transformed before MAE, RMSE, and R² are computed.
Classification result files also contain accuracy, macro F1, UAR, class counts, and a majority
baseline.

## Linear CKA

CKA uses 500 deterministically sampled IDs shared by all three LibriSpeech feature stores. For each
layer, sample-centered features form a linear Gram matrix, which is normalized by its Frobenius
norm. Pairwise dot products between normalized Gram vectors produce the within-model and
cross-model matrices.

The implementation explicitly aligns IDs rather than assuming that two stores have the same row
order. Within-model matrices are checked for symmetry and unit diagonals.

## Downstream validation

The WavLM emotion study reuses the actor-disjoint CREMA-D split and compares five strategies:

- a fixed early layer (L2),
- the final layer (L24),
- the single validation-selected probing layer,
- concatenation of the three highest-ranked validation layers, and
- a trainable softmax-weighted sum of all 25 representations.

The probing rank comes only from validation UAR. Each strategy uses the same seeds and reports
validation/test accuracy, macro F1, and UAR. The study tests whether probe-informed representation
selection is useful under this fixed benchmark; it is not a comparison against externally tuned
emotion-recognition systems.

## Artifact interpretation

Uncertainty bands in `results/figures/probing_summary.png` show one standard deviation across the
three seeds. Markers indicate the layer selected by validation performance, while plotted curves
show test metrics to describe the held-out layer profile.

The compact committed artifacts are the intended audit surface:

- `results/probing/<task>/<model>/result.json`
- `results/cka/<model_a>__<model_b>.npz`
- `results/downstream/emotion/wavlm_large/result.json`
- `results/figures/*.png`

Raw audio, model weights, and HDF5 feature stores are excluded because of size and licensing.
