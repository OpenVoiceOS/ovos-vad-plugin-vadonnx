# vadonnx VAD

A voice activity detection (VAD) plugin for [OpenVoiceOS](https://github.com/OpenVoiceOS)
that runs the models of [vadonnx](https://github.com/TigreGotico/vadonnx). VAD tells the
listener when a chunk of audio contains speech, so the assistant knows when the user
stops talking.

One plugin gives access to all vadonnx models. The `model` config key selects the model.
All models run on ONNX Runtime. PyTorch is not necessary.

## Install

```bash
uv pip install ovos-vad-plugin-vadonnx
```

The `fsmn` models also need the fbank frontend:

```bash
uv pip install "ovos-vad-plugin-vadonnx[fsmn]"
```

## Usage

Set `ovos-vad-plugin-vadonnx` as the VAD module in your listener config:

```json
{
    "listener": {
        "VAD": {
            "module": "ovos-vad-plugin-vadonnx",
            "ovos-vad-plugin-vadonnx": {
                "model": "silero",
                "threshold": 0.5
            }
        }
    }
}
```

| key | default | description |
|-----|---------|-------------|
| `model` | `silero` | vadonnx model name, local `.onnx` path or URL |
| `threshold` | model default | minimum speech probability (0 to 1) for a chunk to count as speech |
| `revision` | registry value | HuggingFace revision of the model file |
| `cache_dir` | `$XDG_DATA_HOME/vadonnx` | download cache directory |
| `providers` | CPU | ONNX Runtime execution providers |
| `signature` | none | IO signature for a custom `.onnx` file |

The model default threshold is 0.5 for most models and 0.7 for `speechbrain`.

## Models

| model | frame | licence | notes |
|-------|-------|---------|-------|
| `silero`, `silero-8k` | 32 ms | MIT | in the vadonnx wheel, works offline |
| `silero-op15` | 32 ms | MIT | opset 15, for older ONNX Runtime versions |
| `ten` | 16 ms | Apache-2.0 with additional conditions | read the [TEN VAD licence](https://github.com/TEN-framework/ten-vad/blob/main/LICENSE) before use |
| `fsmn`, `fsmn-quant` | 10 ms | FunASR Model Open Source License | needs the `fsmn` extra. Read the [FunASR model licence](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) before use |
| `marblenet`, `marblenet-int8` | 20 ms | NVIDIA Open Model License | multilingual |
| `speechbrain` | 10 ms | Apache-2.0 | |
| `pyannote`, `pyannote-int8` | 17 ms | MIT | 10 s analysis window |

All models except `silero` and `silero-8k` download on first load. When you use one of
them, the device must have internet access the first time the listener starts.

Some models analyze audio in blocks that are longer than one microphone chunk
(`fsmn` 0.16 s, `marblenet` 0.5 s, `speechbrain` 0.5 s, `pyannote` 10 s). With these models the plugin reports the result of
the last complete block, so the speech decision comes later. Do not use `pyannote` for
live listening. A 10 s delay stops the listener from finding the end of speech in time.

Run `python -c "import vadonnx; print(vadonnx.list_models())"` to see the models that
your installed vadonnx version has.

## Related projects

- [vadonnx](https://github.com/TigreGotico/vadonnx), the VAD library that this plugin runs.
- [ovos-dinkum-listener](https://github.com/OpenVoiceOS/ovos-dinkum-listener), the OVOS listener that loads VAD plugins.
- [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager), defines the `VADEngine` template that this package implements.
- [ovos-vad-plugin-silero](https://github.com/OpenVoiceOS/ovos-vad-plugin-silero), a VAD plugin for the Silero model only.

## License

Apache-2.0. The model weights keep their upstream licences. See the
[vadonnx licensing notes](https://github.com/TigreGotico/vadonnx/blob/dev/docs/licensing.md).
