"""The depth camera powers down after the keep-alive, and reports when it is warm."""
import asyncio
import time
import types

from grabette.backend import rpi


def _warm_backend(keepalive_s=0.05):
    b = rpi.RpiBackend()
    b._oakd_keepalive_s = keepalive_s
    b._oakd = types.SimpleNamespace(is_initialized=True)
    b._oakd_auto_enabled = True
    powered_down = []

    async def _set_enabled(on):
        powered_down.append(not on)
        b._oakd = None

    b.set_oakd_enabled = _set_enabled
    return b, powered_down


def test_the_keepalive_powers_the_camera_down():
    b, powered_down = _warm_backend()

    asyncio.run(b._oakd_keepalive_powerdown())

    assert powered_down == [True]


def test_a_warmup_with_no_capture_after_it_rearms_the_powerdown():
    # The fleet warms the camera when a session opens. If no take follows, the
    # camera must not be left on forever.
    b, _ = _warm_backend()
    b._oakd = types.SimpleNamespace(is_initialized=True, wait_until_ready=lambda t: True)
    b.raise_if_capture_blocked = lambda: None

    async def scenario():
        await b.prepare_capture()
        armed = b._oakd_keepalive_task is not None
        b._cancel_oakd_keepalive()
        return armed

    assert asyncio.run(scenario()) is True


# --- "warm" means converged, not merely powered ------------------------------

def _camera(ready):
    return types.SimpleNamespace(is_initialized=True, wait_until_ready=lambda t: ready)


def test_an_initialized_camera_is_not_warm_until_it_has_converged():
    b = rpi.RpiBackend()
    b._oakd = _camera(ready=True)
    assert b.is_depth_camera_warm is False

    asyncio.run(b._wait_depth_ready())
    assert b.is_depth_camera_warm is True


def test_a_camera_that_never_converged_is_not_warm():
    b = rpi.RpiBackend()
    b._oakd = _camera(ready=False)

    asyncio.run(b._wait_depth_ready())

    assert b.is_depth_camera_warm is False


def test_a_reinitialized_camera_has_to_converge_again():
    b = rpi.RpiBackend()
    b._oakd = _camera(ready=True)
    asyncio.run(b._wait_depth_ready())

    b._oakd = _camera(ready=True)

    assert b.is_depth_camera_warm is False


# --- two warm-ups at once bring the camera up only once ----------------------

def test_concurrent_enables_initialize_the_camera_once():
    import threading
    import time as _time

    b = rpi.RpiBackend()
    inits = []

    def _init():
        inits.append(threading.get_ident())
        _time.sleep(0.05)
        b._oakd = _camera(ready=True)

    b._init_oakd = _init

    async def scenario():
        await asyncio.gather(b.set_oakd_enabled(True), b.set_oakd_enabled(True))

    asyncio.run(scenario())
    assert len(inits) == 1


# --- not "warm" right before the keep-alive powers it down --------------------

def test_a_camera_about_to_power_down_is_not_warm():
    # The fleet would pick the short lead and the camera would be off at T0.
    b = rpi.RpiBackend()
    b._oakd = _camera(ready=True)
    asyncio.run(b._wait_depth_ready())

    b._oakd_powerdown_at = time.monotonic() + rpi.OAKD_WARM_MARGIN_S + 10
    assert b.is_depth_camera_warm is True
    b._oakd_powerdown_at = time.monotonic() + rpi.OAKD_WARM_MARGIN_S - 1
    assert b.is_depth_camera_warm is False
