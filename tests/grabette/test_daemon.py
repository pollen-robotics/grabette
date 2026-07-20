"""Unit tests for grabette.daemon.SampleRing — the cursor-based sample buffer.

Every telemetry consumer (UI, replay, upload) reads the live IMU/angle stream
through SampleRing's cursor protocol. Correct cursor advancement and maxlen
eviction are what keep consumers from missing or re-reading samples.
"""

from types import SimpleNamespace

from grabette.daemon import SampleRing


def test_get_since_returns_all_then_advances_cursor():
    """get_since(0) returns everything; reading from the new cursor sees only fresh samples."""
    ring = SampleRing(maxlen=10)
    ring.push_raw(imu={"t": 1})
    ring.push_raw(imu={"t": 2})

    first = ring.get_since(0)
    assert [s["t"] for s in first["imu"]] == [1, 2]
    assert first["cursor"] == 2

    # Reading again from the returned cursor yields nothing new...
    assert ring.get_since(first["cursor"])["imu"] == []
    # ...until another sample arrives.
    ring.push_raw(imu={"t": 3})
    nxt = ring.get_since(first["cursor"])
    assert [s["t"] for s in nxt["imu"]] == [3]
    assert nxt["cursor"] == 3


def test_imu_and_angle_share_a_monotonic_seq():
    """IMU and angle streams advance one shared sequence, so a cursor spans both."""
    ring = SampleRing(maxlen=10)
    ring.push_raw(imu={"t": 1})            # seq 1
    ring.push_raw(angle={"t": 2})          # seq 2
    ring.push_raw(imu={"t": 3}, angle={"t": 3})  # seq 3 (both)

    out = ring.get_since(2)  # only seq 3 survives the cursor
    assert [s["t"] for s in out["imu"]] == [3]
    assert [s["t"] for s in out["angle"]] == [3]


def test_maxlen_evicts_oldest():
    """The ring keeps only maxlen samples while the cursor still counts all pushes."""
    ring = SampleRing(maxlen=3)
    for i in range(5):
        ring.push_raw(imu={"t": i})
    out = ring.get_since(0)
    # Only the last 3 remain; cursor still reflects total pushes.
    assert [s["t"] for s in out["imu"]] == [2, 3, 4]
    assert out["cursor"] == 5


def test_push_state_extracts_imu_and_angle_fields():
    """push_state flattens a SensorState into the ring's imu/angle dict schema."""
    ring = SampleRing(maxlen=10)
    state = SimpleNamespace(
        imu=SimpleNamespace(timestamp_ms=10.0, accel=(1, 2, 3), gyro=(4, 5, 6)),
        angle=SimpleNamespace(timestamp_ms=11.0, proximal=0.5, distal=0.7),
    )
    ring.push_state(state)
    out = ring.get_since(0)
    assert out["imu"] == [{"t": 10.0, "a": [1, 2, 3], "g": [4, 5, 6]}]
    assert out["angle"] == [{"t": 11.0, "p": 0.5, "d": 0.7}]
