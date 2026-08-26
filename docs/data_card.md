# Data card: Google Speech Commands v0.01

## Intended use

This project uses Speech Commands for **speaker-independent, limited-vocabulary keyword
spotting**. It does not perform continuous speech recognition, speaker identification,
transcription, or language understanding.

## Source and license

- Creator: Google Speech and TensorFlow teams.
- Release: version 0.01, 3 August 2017.
- Original archive: `speech_commands_v0.01.tar.gz` from TensorFlow storage.
- License: Creative Commons Attribution 4.0, distributed inside the source archive.
- Primary reference: Pete Warden, *Speech Commands: A Dataset for Limited-Vocabulary
  Speech Recognition*, 2018, [arXiv:1804.03209](https://arxiv.org/abs/1804.03209).
- Release announcement: [Google Research Blog](https://research.google/blog/launching-the-speech-commands-dataset/).

The raw archive is not redistributed. `data/processed/dataset_source.json` records its
byte size and SHA-256 digest after download.

## Raw structure

The release contains approximately 65,000 mono 16 kHz WAV clips, each at most one second,
organized into 30 word folders. Filenames include a stable anonymized speaker identifier;
multiple recordings from the same speaker share the part before `_nohash_`. Separate
background-noise recordings support augmentation and silence synthesis.

Exact counts, durations, speakers, and per-word distributions are generated from the local
archive in `artifacts/tables/eda_summary.json` and `raw_word_counts.csv`; no headline count
is hard-coded into the evaluation.

## Modeled task

The fixed label order is:

1. `_silence_`
2. `_unknown_`
3. `yes`
4. `no`
5. `up`
6. `down`
7. `left`
8. `right`
9. `on`
10. `off`
11. `stop`
12. `go`

Non-target words are candidate recordings for `unknown`. Following the original TensorFlow
protocol, a deterministic sample equal to 10% of the target-word count is used in each split.
The same proportion of `silence` examples is synthesized from background recordings.

## Splits and leakage controls

The archive-provided `validation_list.txt` and `testing_list.txt` are authoritative. All
remaining speech files form the training split. These official lists keep every recording
from a speaker in one split. Manifest creation fails if an anonymized speaker identifier
appears in more than one split.

Silence clips use split-specific, temporally disjoint regions (80%/10%/10%) of every
background file. Overlap across train, validation, and test is checked before writing the
manifest. Feature normalization is learned on training data only; augmentation is never
applied to validation or test.

## Limitations and risks

- The labels cover isolated English words, not natural conversational speech.
- Volunteer recordings cannot represent every accent, microphone, room, or speaking style.
- Demographic attributes are unavailable, so demographic fairness metrics cannot be computed.
- The `unknown` class is heterogeneous and its sampled composition affects difficulty.
- Synthetic silence is a proxy for deployment background conditions.
- Performance on this dataset should not be interpreted as production ASR readiness.

The final evaluation therefore includes unseen speakers, per-class metrics, confidence
calibration, background-noise stress tests, temporal shifts, and direct inspection/listening
of selected failures.

