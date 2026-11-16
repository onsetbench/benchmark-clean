# Prediction format

This page describes the epoch-clock scorer. For the current relative-clock lab package, use `python -m onsetbench.submission` as described in README. Its manifest hash binds the recording-relative clock; never reinterpret it as epoch time.

Submit JSON data. The evaluator runs no submitted code.

```json
{
  "schema": "onsetbench.predictions/1",
  "dataset_sha256": "checksum of the supplied dataset manifest",
  "name": "model name and revision",
  "inputs": "standard_video",
  "target": "SENSOR",
  "recordings": {
    "recording-id": {
      "onsets_ms": [1736429330000.0],
      "midpoints_ms": [1736429330200.0],
      "contact": [1, 0, -1]
    }
  }
}
```

The template command writes the correct dataset checksum and complete recording
list. All times use the supplied epoch-millisecond clock. `onsets_ms` is required;
an empty list represents no emitted events. Duplicates are preserved and penalized.

`midpoints_ms` is optional. When supplied, it contains one original source-window
midpoint per emitted event, in corresponding order. Supply it for all recordings,
including empty lists for silent recordings, or omit it throughout.

`contact` is optional for SENSOR. It contains one decision per `starts_ms` entry
in the dataset manifest: 1 for Contact, -1 for No-contact, 0 for Unknown. Supply
all recordings or omit contact throughout. A window's reference class is defined
by whether its center lies within a sensor-defined loading span, with endpoints
included. Unknown decisions reduce coverage. They are not treated as correct.

SENSOR is force-bearing loading onset.

Files with missing recordings, extra recordings, nonfinite numbers, repeated JSON
fields or altered reference checksums are rejected. The runner checks frame-clock
identity. For manually written JSON, use the supplied epoch-millisecond clock;
the evaluator counts and reports finite events outside the input time range.

## Output

`onsetbench.score/1` includes input hashes, target, split, recording count and
support hours. `timing["200"]` and `timing["100"]` each include native and shifted
counts and metrics, optional midpoint metrics, and paired control contrasts.
`contact_state` is null if omitted. Undefined ratios and matched-event timing
errors are null. `raw` retains per-recording counts and signed matched lags.

