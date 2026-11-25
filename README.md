# OnsetBench scorer

Validate and score physical-event predictions from video. Results: https://onsetbench.com

## Run the scorer locally on 40 development recordings

Use Python 3.11 or later. From this checkout, install the scorer:

```sh
python -m pip install .
```

Download `onsetbench-public40-labels.zip` from the [OnsetBench data repo](https://huggingface.co/datasets/OnsetBench/benchmark-data). Download `onsetbench-public40-videos.zip` too if you need the wrist videos and frame timestamps to generate predictions. The scorer scores predictions you supply; it does not run your model.

Extract the labels archive and change to the extracted directory that contains `data/reassemble_public40.json`. If you downloaded the videos archive, you can extract it in that same directory. Then run:

```sh
python -m onsetbench verify --dataset data/reassemble_public40.json
python -m onsetbench template --dataset data/reassemble_public40.json --out predictions.json
```

Fill the template with your predicted `onsets_ms` using the supplied **epoch-millisecond** clock. Do not subtract the first frame timestamp. Then score:

```sh
python -m onsetbench score --dataset data/reassemble_public40.json --predictions predictions.json --out score.json
```

The 40 recordings are for local development. Official REASSEMBLE results use a [separate 83-recording evaluation package](https://onsetbench.com/downloads.html) with a recording-relative clock. The starter kit does not reproduce the published results.

## Relative-clock evaluation submissions

The current lab package uses `recording_relative_ms`, with each video's first frame at zero. Use its supplied frame timestamps and preserve repeated grid entries; grid starts may be negative.

```sh
python -m onsetbench.submission template --dataset /path/to/data/reassemble_eval83.json --name "model revision" --out predictions.json
python -m onsetbench.submission validate --dataset /path/to/data/reassemble_eval83.json --predictions predictions.json
```

Fill onset times in each recording's relative clock. Optional state decisions correspond to each window midpoint (`starts_ms + span_ms/2`), not its start. Optional source midpoints correspond one-for-one to emitted events.

Do not pass the relative lab package to the epoch-clock scorer or relabel relative times as epoch times. Send the validated file to hello@haptera.ai.

## Toy example

```sh
python -m pip install .
python -m onsetbench verify --dataset examples/dataset.json
python -m onsetbench score --dataset examples/dataset.json --predictions examples/predictions.json --out score.json
python -m unittest discover -s tests
```

See [Prediction format](docs/submission.md) for the epoch-clock file format.

## Licence

Evaluation code: Apache License 2.0. See LICENSE and NOTICE.
