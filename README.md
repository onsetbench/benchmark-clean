# OnsetBench scorer

Validate and score physical-event predictions from video. Results: https://onsetbench.com

## Epoch-clock scorer example

Requires Python 3.11 or later.

```sh
python -m pip install .
python -m onsetbench verify --dataset examples/dataset.json
python -m onsetbench score --dataset examples/dataset.json --predictions examples/predictions.json --out score.json
python -m unittest discover -s tests
```

The examples demonstrate the file format and scorer.

See [Prediction format](docs/submission.md). To request an evaluation package, email hello@haptera.ai.

## Licence

Evaluation code: Apache License 2.0. See LICENSE and NOTICE.

## Relative-clock evaluation submissions

The current lab package uses `recording_relative_ms`, with each video's first frame at zero. Use its supplied frame timestamps and preserve repeated grid entries; grid starts may be negative.

```sh
python -m onsetbench.submission template --dataset /path/to/data/reassemble_eval83.json --name "model revision" --out predictions.json
python -m onsetbench.submission validate --dataset /path/to/data/reassemble_eval83.json --predictions predictions.json
```

Fill onset times in each recording's relative clock. Optional state decisions correspond to each window midpoint (`starts_ms + span_ms/2`), not its start. Optional source midpoints correspond one-for-one to emitted events. 

The epoch-clock scorer above is unchanged. Do not pass the relative lab package to it or relabel relative times as epoch times. Send the validated file to hello@haptera.ai.
