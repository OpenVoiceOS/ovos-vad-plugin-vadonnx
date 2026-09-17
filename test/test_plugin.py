"""Offline tests for VadonnxVAD with the Silero models bundled in the vadonnx wheel."""
import wave
from pathlib import Path

import pytest

from ovos_vad_plugin_vadonnx import VadonnxVAD

FIXTURE = Path(__file__).parent / "fixtures" / "command.wav"
CHUNK = 2048 * 2  # dinkum-listener default: 2048 int16 samples


def _speech() -> bytes:
    with wave.open(str(FIXTURE)) as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1
        return wf.readframes(wf.getnframes())


def _chunks(audio: bytes):
    return [audio[i:i + CHUNK] for i in range(0, len(audio) - CHUNK + 1, CHUNK)]


@pytest.fixture
def vad():
    return VadonnxVAD({}, sample_rate=16000)


def test_default_model_is_silero(vad):
    assert vad.model_name == "silero"
    assert vad.threshold == 0.5


def test_digital_silence_is_silence(vad):
    assert all(vad.is_silence(c) for c in _chunks(b"\x00" * CHUNK * 10))


def test_speech_is_detected(vad):
    results = [vad.is_silence(c) for c in _chunks(_speech())]
    assert results.count(False) >= 3


def test_threshold_from_config():
    vad = VadonnxVAD({"threshold": 1.01}, sample_rate=16000)
    assert vad.threshold == 1.01
    # no probability reaches 1.01, so every chunk is silence
    assert all(vad.is_silence(c) for c in _chunks(_speech()))


@pytest.mark.parametrize("model", ["silero", "silero-8k"])
def test_model_from_config(model):
    vad = VadonnxVAD({"model": model}, sample_rate=16000)
    assert vad.model_name == model
    assert [vad.is_silence(c) for c in _chunks(_speech())].count(False) >= 3


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        VadonnxVAD({"model": "no-such-vad"}, sample_rate=16000)


def test_reset_clears_stream_state(vad):
    for c in _chunks(_speech()):
        vad.is_silence(c)
    vad.reset()
    assert vad.vad._buf.size == 0
    assert vad.is_silence(b"\x00" * CHUNK)


def test_extract_speech_keeps_speech_and_resets(vad):
    audio = b"\x00" * 16000 + _speech() + b"\x00" * 32000
    # leave partial streaming state behind; extract_speech must not read it
    vad.is_silence(_speech()[:1000])
    out = vad.extract_speech(audio)
    assert out is not None
    assert 0 < len(out) < len(audio)
    assert vad.vad._buf.size == 0


def test_available_models_lists_vadonnx_models():
    assert "silero" in VadonnxVAD.available_models()


def test_opm_entry_point_loads():
    from ovos_plugin_manager.vad import OVOSVADFactory, load_vad_plugin

    assert load_vad_plugin("ovos-vad-plugin-vadonnx") is VadonnxVAD
    vad = OVOSVADFactory.create({"module": "ovos-vad-plugin-vadonnx",
                                 "ovos-vad-plugin-vadonnx": {"model": "silero-8k"}})
    assert isinstance(vad, VadonnxVAD)
    assert vad.model_name == "silero-8k"


@pytest.mark.parametrize("model", ["silero", "silero-8k"])
def test_speech_then_silence_chunk_is_speech(model):
    """A chunk whose last frame is silence is speech when an earlier frame is speech."""
    vad = VadonnxVAD({"model": model}, sample_rate=16000)
    quarter = CHUNK // 4
    detected = 0
    for c in _chunks(_speech()):
        # oracle: the speech part alone, read by vadonnx at its last frame
        vad.reset()
        head = c[:3 * quarter]
        if vad.vad.process_chunk(head, sample_rate=16000) < vad.threshold:
            continue
        vad.reset()
        mixed = c[:3 * quarter] + b"\x00" * quarter
        assert not vad.is_silence(mixed)
        detected += 1
    assert detected >= 3


def test_runtime_requirements_need_network_at_load():
    req = VadonnxVAD.runtime_requirements
    assert req.internet_before_load and req.network_before_load
