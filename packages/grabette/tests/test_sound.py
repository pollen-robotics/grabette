"""Speaker cue tests — no audio hardware involved.

The contract that matters on-device is that the cue is a valid 48 kHz stereo
WAV (the codec's fixed 12 MHz MCLK / mclk-fs=250 leaves no other rate) and that
every failure path stays silent instead of raising into start_capture.
"""

from __future__ import annotations

import wave

import pytest

from grabette.hardware import sound


def test_rendered_cue_is_48k_stereo_16bit(tmp_path):
    path = tmp_path / "cue.wav"
    sound._render_wav(path, sound.START_TONES, volume=0.6)
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == sound.SAMPLE_RATE == 48000
        assert w.getnchannels() == 2
        assert w.getsampwidth() == 2


@pytest.mark.parametrize("name", list(sound.CUES))
def test_every_cue_is_padded_past_the_alsa_buffer(tmp_path, name):
    """A clip shorter than the device's ring buffer can leave the DMA transfer
    never started, with aplay blocked in drain — which is what a 60 ms cue did,
    on every recording. The padding is a correctness requirement, so hold every
    cue to it, not just the short one that exposed it."""
    path = tmp_path / f"{name}.wav"
    sound._render_wav(path, sound.CUES[name], volume=0.6)
    with wave.open(str(path), "rb") as w:
        assert w.getnframes() >= int(sound.SAMPLE_RATE * sound.MIN_CUE_S)


def test_padding_is_silence_appended_after_the_tones(tmp_path):
    """The pad must not alter how a cue sounds: the audible part is unchanged
    and everything after it is exactly zero."""
    tones = ((1000.0, 0.05),)
    path = tmp_path / "short.wav"
    sound._render_wav(path, tones, volume=1.0)
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    voiced_frames = int(sound.SAMPLE_RATE * 0.05)
    assert set(frames[voiced_frames * 4:]) == {0}          # pad is silent
    assert any(frames[:voiced_frames * 4])                 # tone survived


def test_long_cue_is_not_truncated_by_the_minimum(tmp_path):
    tones = ((440.0, sound.MIN_CUE_S + 0.3),)
    path = tmp_path / "long.wav"
    sound._render_wav(path, tones, volume=0.6)
    with wave.open(str(path), "rb") as w:
        assert w.getnframes() == int(sound.SAMPLE_RATE * (sound.MIN_CUE_S + 0.3))


def test_cue_starts_and_ends_near_silence(tmp_path):
    """The fade envelope is what keeps the speaker from clicking on a DC step."""
    path = tmp_path / "cue.wav"
    sound._render_wav(path, sound.START_TONES, volume=1.0)
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    first = int.from_bytes(frames[0:2], "little", signed=True)
    last = int.from_bytes(frames[-2:], "little", signed=True)
    assert abs(first) < 500
    assert abs(last) < 500


def test_volume_scales_peak_amplitude(tmp_path):
    peaks = []
    for volume in (0.2, 1.0):
        path = tmp_path / f"cue-{volume}.wav"
        sound._render_wav(path, ((1000.0, 0.05),), volume=volume)
        with wave.open(str(path), "rb") as w:
            frames = w.readframes(w.getnframes())
        peaks.append(max(
            abs(int.from_bytes(frames[i:i + 2], "little", signed=True))
            for i in range(0, len(frames), 2)
        ))
    assert peaks[0] < peaks[1]
    assert peaks[1] <= 32767


def test_start_and_stop_cues_are_distinguishable(tmp_path):
    """Rising vs falling — the operator has to tell the two apart by ear."""
    assert sound.START_TONES != sound.STOP_TONES
    start_freqs = [f for f, _ in sound.START_TONES]
    stop_freqs = [f for f, _ in sound.STOP_TONES]
    assert start_freqs == sorted(start_freqs)              # ascending
    assert stop_freqs == sorted(stop_freqs, reverse=True)  # descending


def test_error_cue_is_low_and_repeated():
    """It has to read as "something went wrong" to someone looking at the
    workspace rather than the device: well below both other cues, and repeated
    instead of a single glide."""
    voiced = [f for f, _ in sound.ERROR_TONES if f > 0]
    assert len(voiced) >= 3                                   # repeated
    assert max(voiced) < min(f for f, _ in sound.START_TONES)  # lower than both
    assert max(voiced) < min(f for f, _ in sound.STOP_TONES)
    # And longer overall, so it isn't mistaken for a clipped start/stop.
    total = sum(d for _, d in sound.ERROR_TONES)
    assert total > sum(d for _, d in sound.START_TONES)


def test_saved_cue_is_the_slightest(tmp_path):
    """It fires on every single take, a second after the stop cue, so it must
    be the shortest and simplest of the four — a confirmation, not an event."""
    assert len(sound.SAVED_TONES) == 1
    saved_len = sum(d for _, d in sound.SAVED_TONES)
    for other in (sound.START_TONES, sound.STOP_TONES, sound.ERROR_TONES):
        assert saved_len < sum(d for _, d in other)
    # ...and above the others in pitch, so it can't be mistaken for one of them
    # arriving clipped.
    assert min(f for f, _ in sound.SAVED_TONES) > max(
        f for f, _ in sound.START_TONES + sound.STOP_TONES
    )


def test_silent_gaps_render_as_silence(tmp_path):
    """The error cue's gaps are 0 Hz "tones"; they must be actual silence."""
    path = tmp_path / "gap.wav"
    sound._render_wav(path, ((0.0, 0.02),), volume=1.0)
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(int(sound.SAMPLE_RATE * 0.02))
    assert set(frames) == {0}


def test_same_cue_is_debounced_but_a_different_one_is_not(monkeypatch):
    """Several layers report the same failure; only one buzz should come out.
    A different cue must still get through — a stop right after a start, say."""
    calls = []
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    monkeypatch.setattr(sound.subprocess, "Popen", _fake_popen(calls, FakeProc()))
    speaker = sound.Speaker(device="plughw:CARD=aic3104,DEV=0")
    speaker.prepare()

    now = [1000.0]
    monkeypatch.setattr(sound.time, "monotonic", lambda: now[0])

    speaker.play_error()
    speaker.play_error()          # same cue, same instant → suppressed
    speaker.play_start()          # different cue → allowed
    assert len(calls) == 2

    now[0] += sound.CUE_DEBOUNCE_S + 0.01
    speaker.play_error()          # window elapsed → allowed again
    assert len(calls) == 3
    speaker.close()


def test_prepare_renders_every_cue(monkeypatch):
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    speaker = sound.Speaker(device="plughw:CARD=aic3104,DEV=0")
    speaker.prepare()
    assert set(speaker._cues) == set(sound.CUES)
    assert all(p.exists() for p in speaker._cues.values())
    speaker.close()
    assert speaker._cues == {}
    assert not speaker.is_available


def test_play_stop_uses_the_stop_cue(monkeypatch):
    calls = []
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    monkeypatch.setattr(sound.subprocess, "Popen", _fake_popen(calls, FakeProc()))
    speaker = sound.Speaker(device="plughw:CARD=aic3104,DEV=0")
    speaker.prepare()
    assert speaker._spawn(speaker._cues[sound.CUE_STOP]) is True
    speaker.close()
    assert calls[0][-1].endswith(f"{sound.CUE_STOP}.wav")


def test_disabled_speaker_is_inert():
    speaker = sound.Speaker(enabled=False)
    speaker.prepare()
    assert not speaker.is_available
    speaker.play_start()  # must not raise, must not spawn anything
    speaker.play_stop()
    speaker.close()


def test_missing_aplay_disables_instead_of_raising(monkeypatch):
    monkeypatch.setattr(sound.shutil, "which", lambda _: None)
    speaker = sound.Speaker()
    speaker.prepare()
    assert not speaker.is_available
    speaker.play_start()
    speaker.close()


def test_missing_card_disables_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    # No /proc/asound/aic3104 → autodetect finds nothing.
    monkeypatch.setattr(sound, "autodetect_device", lambda: None)
    speaker = sound.Speaker()
    speaker.prepare()
    assert not speaker.is_available


def test_speakerless_device_never_runs_aplay(monkeypatch):
    """The speaker is optional hardware: with no card, BOTH cues must be inert
    — no subprocess, no thread, no exception — since the backend calls them
    unconditionally on every capture start/stop."""
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    monkeypatch.setattr(sound, "autodetect_device", lambda: None)

    def must_not_run(*a, **kw):
        raise AssertionError("aplay must not be spawned without a sound card")

    monkeypatch.setattr(sound.subprocess, "Popen", must_not_run)
    speaker = sound.Speaker()
    speaker.prepare()
    speaker.play_start()
    speaker.play_stop()
    speaker.play_saved()
    speaker.play_error()
    sound.cue_error()  # the module-level helper used by the non-backend callers
    speaker.close()  # also fine to close a speaker that never opened anything


class FakeProc:
    """Minimal Popen stand-in: exits with `returncode`, says `stderr`."""

    def __init__(self, returncode=0, stderr=b""):
        self.returncode = returncode
        self._stderr = stderr

    def communicate(self, timeout=None):
        return (b"", self._stderr)

    def poll(self):
        return self.returncode

    def terminate(self):
        pass


def _fake_popen(calls, proc):
    def popen(cmd, **kw):
        calls.append(cmd)
        return proc
    return popen


def test_play_start_spawns_aplay_with_the_named_card(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    monkeypatch.setattr(sound.subprocess, "Popen", _fake_popen(calls, FakeProc()))
    speaker = sound.Speaker(device="plughw:CARD=aic3104,DEV=0")
    speaker.prepare()
    assert speaker.is_available
    # play_start dispatches to a thread; call the spawn directly so the test
    # isn't timing-dependent.
    assert speaker._spawn(speaker._cues[sound.CUE_START]) is True
    speaker.close()

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "/usr/bin/aplay"
    assert "plughw:CARD=aic3104,DEV=0" in cmd
    assert cmd[-1].endswith(".wav")


def test_aplay_failure_is_logged_not_swallowed(monkeypatch, caplog):
    """A silent speaker must leave a trace in the journal — a muted mixer, a
    busy card and a /dev/snd permission error are otherwise indistinguishable."""
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    monkeypatch.setattr(sound.subprocess, "Popen", _fake_popen(
        [], FakeProc(returncode=1, stderr=b"aplay: main:831: audio open error: Permission denied"),
    ))
    speaker = sound.Speaker(device="plughw:CARD=aic3104,DEV=0")
    speaker.prepare()
    with caplog.at_level("WARNING"):
        assert speaker._spawn(speaker._cues[sound.CUE_START]) is False
    speaker.close()
    assert "Permission denied" in caplog.text
    assert "exit 1" in caplog.text


def test_hung_aplay_is_killed(monkeypatch, caplog):
    class HungProc(FakeProc):
        def __init__(self):
            super().__init__()
            self.killed = False

        def communicate(self, timeout=None):
            raise sound.subprocess.TimeoutExpired(cmd="aplay", timeout=timeout)

        def kill(self):
            self.killed = True

    proc = HungProc()
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    monkeypatch.setattr(sound.subprocess, "Popen", _fake_popen([], proc))
    speaker = sound.Speaker(device="plughw:CARD=aic3104,DEV=0")
    speaker.prepare()
    with caplog.at_level("WARNING"):
        assert speaker._spawn(speaker._cues[sound.CUE_START]) is False
    speaker.close()
    assert proc.killed
    assert "killed" in caplog.text


def test_play_never_raises_when_spawn_fails(monkeypatch):
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    speaker = sound.Speaker(device="plughw:CARD=aic3104,DEV=0")
    speaker.prepare()

    def boom(*a, **kw):
        raise OSError("no such device")

    monkeypatch.setattr(sound.subprocess, "Popen", boom)
    speaker._spawn(speaker._cues[sound.CUE_START])  # swallowed + logged, never raised
    speaker.close()
