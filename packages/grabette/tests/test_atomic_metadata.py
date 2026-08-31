"""metadata.json must never be observed empty or half-written.

The episode writer puts metadata.json LAST so that its presence means "this
episode is fully saved". A plain write_text() truncates the target on open, so
an interrupted write — power cut on the Pi, SD card full — left a file that
existed but was empty. Reading it back raised, which 500'd GET /api/tasks and
blanked the device dashboard of every task while the fleet still showed them.
"""
import json

import pytest

from grabette.output import write_json_atomic


def test_it_writes_readable_json_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "metadata.json"

    write_json_atomic(target, {"duration_seconds": 4.0})

    assert json.loads(target.read_text()) == {"duration_seconds": 4.0}
    assert list(tmp_path.iterdir()) == [target], "a .tmp file was left behind"


def test_a_failed_write_leaves_the_previous_file_intact(monkeypatch, tmp_path):
    # The case that matters: the reader must see the OLD complete file or none —
    # never the 0-byte truncation that a plain write_text() produces here.
    target = tmp_path / "metadata.json"
    write_json_atomic(target, {"duration_seconds": 4.0})

    def boom(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(json, "dump", boom)

    with pytest.raises(OSError):
        write_json_atomic(target, {"duration_seconds": 9.0})

    assert json.loads(target.read_text()) == {"duration_seconds": 4.0}


def test_the_backends_write_metadata_through_it(tmp_path):
    # Both backends must go through the atomic writer — a direct write_text on
    # metadata.json anywhere reintroduces the truncation window.
    from pathlib import Path
    import grabette.backend.mock as mock
    import grabette.backend.rpi as rpi

    for mod in (mock, rpi):
        src = Path(mod.__file__).read_text()
        assert 'write_json_atomic(episode_dir / "metadata.json"' in src
        assert '"metadata.json").write_text' not in src, f"{mod.__name__} truncates"
