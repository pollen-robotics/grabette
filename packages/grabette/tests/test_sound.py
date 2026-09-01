"""Speaker cue tests — no audio hardware involved.

Two contracts. On-device, the cue has to be a valid 48 kHz stereo WAV (the
codec's fixed 12 MHz MCLK / mclk-fs=250 leaves no other rate) and every failure
path has to stay silent rather than raise into start_capture. From the
dashboard, the volume has to actually reach every cue, survive a restart, and
keep working on a grabette with no speaker fitted at all.
"""

from __future__ import annotations

import threading
import wave

from grabette.hardware import sound


def test_rendered_cue_is_48k_stereo_16bit(tmp_path):
    path = tmp_path / "cue.wav"
    sound._render_wav(path, sound.START_TONES, volume=0.6)
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == sound.SAMPLE_RATE == 48000
        assert w.getnchannels() == 2
        assert w.getsampwidth() == 2
        expected = sum(int(sound.SAMPLE_RATE * d) for _, d in sound.START_TONES)
        assert w.getnframes() == expected


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


def _peak(path) -> int:
    """Loudest sample in a rendered cue."""
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return max(
        abs(int.from_bytes(frames[i:i + 2], "little", signed=True))
        for i in range(0, len(frames), 2)
    )


def test_volume_scales_peak_amplitude(tmp_path):
    peaks = []
    for volume in (0.2, 1.0):
        path = tmp_path / f"cue-{volume}.wav"
        sound._render_wav(path, ((1000.0, 0.05),), volume=volume)
        peaks.append(_peak(path))
    assert peaks[0] < peaks[1]
    assert peaks[1] <= 32767


def test_error_cue_is_rendered_with_more_headroom(monkeypatch):
    """The speaker rolls off where the error cue sits, so one shared amplitude
    leaves the buzz — the cue that must never be missed — the only inaudible one
    of the four. It is compensated in the render, not by raising the mixer for
    everything (see CUE_GAINS)."""
    monkeypatch.setattr(sound.shutil, "which", lambda _: "/usr/bin/aplay")
    speaker = sound.Speaker(device="plughw:CARD=aic3104,DEV=0", volume=0.5)
    speaker.prepare()
    peaks = {name: _peak(path) for name, path in speaker._cues.items()}
    assert peaks[sound.CUE_ERROR] > max(
        peak for name, peak in peaks.items() if name != sound.CUE_ERROR
    )
    speaker.close()


def test_start_and_stop_cues_are_distinguishable(tmp_path):
    """Rising vs falling — the operator has to tell the two apart by ear."""
    assert sound.START_TONES != sound.STOP_TONES
    start_freqs = [f for f, _ in sound.START_TONES]
    stop_freqs = [f for f, _ in sound.STOP_TONES]
    assert start_freqs == sorted(start_freqs)              # ascending
    assert stop_freqs == sorted(stop_freqs, reverse=True)  # descending


def test_error_cue_is_repeated_and_long():
    """It has to read as "something went wrong" to someone looking at the
    workspace rather than the device. It used to do that by being the lowest of
    the four — it can't any more, the HAT speaker doesn't reproduce the bottom
    end (see ERROR_TONES). What carries the distinction now is the SHAPE: three
    buzzes at one pitch, separated by silence, far longer than any other cue,
    where the others are a two-tone glide or a single blip."""
    voiced = [f for f, _ in sound.ERROR_TONES if f > 0]
    assert len(voiced) >= 3                              # repeated...
    assert len(set(voiced)) == 1                         # ...at one pitch, not a glide
    assert any(f == 0.0 for f, _ in sound.ERROR_TONES)   # ...with gaps between
    others = (sound.START_TONES, sound.STOP_TONES, sound.SAVED_TONES)
    total = sum(d for _, d in sound.ERROR_TONES)
    for other in others:
        assert total > sum(d for _, d in other)
    # And on a pitch no other cue uses, so a buzz heard through a door can't be
    # taken for one of them.
    assert not set(voiced) & {f for cue in others for f, _ in cue}


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
        frames = w.readframes(w.getnframes())
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


# ── Runtime volume ────────────────────────────────────────────────────────
#
# The volume is baked into the rendered samples (aplay has no volume of its
# own), so "set the volume" means "render the cues again" — and that is the
# part that can silently half-work.

def _speaker(tmp_path, monkeypatch, **kw):
    """A Speaker with its persistence pointed at tmp_path."""
    monkeypatch.setattr(sound, "_volume_path", lambda: tmp_path / "speaker_volume.json")
    return sound.Speaker(device="plughw:test", **kw)


def test_set_volume_rerenders_every_cue(tmp_path, monkeypatch):
    sp = _speaker(tmp_path, monkeypatch, volume=1.0)
    sp._render_cues()
    loud = {n: _peak(p) for n, p in sp._cues.items()}
    assert loud and all(v > 0 for v in loud.values())

    sp.set_volume(0.2)
    quiet = {n: _peak(p) for n, p in sp._cues.items()}
    # Every cue, not just the first: a partial re-render would leave some loud.
    for name in loud:
        assert quiet[name] < loud[name], f"{name} was not re-rendered"


def test_volume_is_clamped_to_a_usable_gain():
    assert sound.clamp_volume(2.5) == 1.0
    assert sound.clamp_volume(-1) == 0.0
    assert sound.clamp_volume(0.42) == 0.42


def test_volume_survives_a_restart(tmp_path, monkeypatch):
    sp = _speaker(tmp_path, monkeypatch, volume=0.6)
    sp.set_volume(0.25)
    monkeypatch.setattr(sound, "_volume_path", lambda: tmp_path / "speaker_volume.json")
    assert sound.load_saved_volume() == 0.25


def test_a_corrupt_volume_file_falls_back_instead_of_failing_boot(tmp_path, monkeypatch):
    path = tmp_path / "speaker_volume.json"
    path.write_text("{ not json")
    monkeypatch.setattr(sound, "_volume_path", lambda: path)
    assert sound.load_saved_volume() is None


def test_volume_is_stored_even_with_no_speaker_fitted(tmp_path, monkeypatch):
    """Fitting a speaker later must bring up the level the operator chose."""
    sp = _speaker(tmp_path, monkeypatch, volume=0.6)
    sp._enabled = False  # what prepare() does when no codec is present
    assert sp.set_volume(0.15) == 0.15
    assert sp.is_available is False
    monkeypatch.setattr(sound, "_volume_path", lambda: tmp_path / "speaker_volume.json")
    assert sound.load_saved_volume() == 0.15


def _join_cue_threads():
    for t in threading.enumerate():
        if t.name in ("speaker-cue", "speaker-test"):
            t.join(timeout=5)


def test_test_plays_the_start_and_stop_pair_in_order(tmp_path, monkeypatch):
    """The two cues an operator has to recognise — and in the order they occur."""
    sp = _speaker(tmp_path, monkeypatch, volume=0.6)
    sp._render_cues()
    played = []
    monkeypatch.setattr(sp, "_spawn", lambda wav: played.append(wav.stem) or True)
    monkeypatch.setattr(sound, "TEST_CUE_GAP_S", 0.0)

    assert sp.play_test() is True
    _join_cue_threads()
    assert played == [sound.CUE_START, sound.CUE_STOP]


def test_test_cue_ignores_the_debounce(tmp_path, monkeypatch):
    """A person pressing Test twice means it twice — unlike a burst of events."""
    sp = _speaker(tmp_path, monkeypatch, volume=0.6)
    sp._render_cues()
    spawned = []
    monkeypatch.setattr(sp, "_spawn", lambda wav: spawned.append(wav) or True)
    monkeypatch.setattr(sound, "TEST_CUE_GAP_S", 0.0)

    sp.play_start()
    sp.play_start()   # inside CUE_DEBOUNCE_S — swallowed
    assert sp.play_test() is True
    assert sp.play_test() is True

    _join_cue_threads()
    assert len(spawned) == 5  # one debounced cue + two per test


def test_test_cue_survives_a_failing_aplay(tmp_path, monkeypatch):
    """A dead first cue must not leave the thread raising into nothing."""
    sp = _speaker(tmp_path, monkeypatch, volume=0.6)
    sp._render_cues()
    monkeypatch.setattr(sound, "TEST_CUE_GAP_S", 0.0)
    monkeypatch.setattr(sp, "_spawn", lambda wav: (_ for _ in ()).throw(OSError("boom")))

    assert sp.play_test() is True
    _join_cue_threads()  # no exception escapes the worker


def test_test_cue_reports_false_with_no_cues(tmp_path, monkeypatch):
    sp = _speaker(tmp_path, monkeypatch, volume=0.6)
    sp._enabled = False
    assert sp.play_test() is False
