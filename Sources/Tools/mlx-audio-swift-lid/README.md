# mlx-audio-swift-lid

CLI demo for `MLXAudioLID`.

## Usage

```bash
swift run mlx-audio-swift-lid --audio Tests/media/intention.wav
```

Use MMS-LID-256 instead of the default ECAPA model:

```bash
swift run mlx-audio-swift-lid \
  --audio Tests/media/intention.wav \
  --model facebook/mms-lid-256 \
  --top-k 3
```

Save the result as JSON:

```bash
swift run mlx-audio-swift-lid \
  --audio Tests/media/intention.wav \
  --output-path lid-output.json
```
