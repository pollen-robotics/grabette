"""Shared dbus/gi stand-ins for the BLE service tests.

Lives in conftest so it runs before any test module imports
bluetooth_service: dbus and gi are Pi-only system packages, absent in CI.
"""
import sys
import types



def _install_dbus_stubs():
    """Put minimal dbus/gi stand-ins in sys.modules, before the import below.

    Only what bluetooth_service touches at import time; the rest is
    monkeypatched per-test.
    """
    if "dbus" in sys.modules:  # real dbus present (e.g. running on a Pi)
        return

    def _passthrough_decorator(*_args, **_kwargs):
        return lambda func: func

    dbus_service = types.ModuleType("dbus.service")
    dbus_service.Object = type("Object", (), {"__init__": lambda self, *a, **k: None})
    dbus_service.method = _passthrough_decorator
    dbus_service.signal = _passthrough_decorator

    class _DBusException(Exception):
        pass

    dbus_exceptions = types.ModuleType("dbus.exceptions")
    dbus_exceptions.DBusException = _DBusException

    dbus_glib = types.ModuleType("dbus.mainloop.glib")
    dbus_glib.DBusGMainLoop = lambda **_kwargs: None
    dbus_mainloop = types.ModuleType("dbus.mainloop")
    dbus_mainloop.glib = dbus_glib

    dbus = types.ModuleType("dbus")
    dbus.service = dbus_service
    dbus.exceptions = dbus_exceptions
    dbus.mainloop = dbus_mainloop
    dbus.DBusException = _DBusException
    dbus.Interface = lambda *a, **k: None
    for name in ("Boolean", "String", "UInt32", "Byte", "Array", "Dictionary",
                 "ObjectPath"):
        setattr(dbus, name, lambda value=None, *a, **k: value)

    glib = types.ModuleType("gi.repository.GLib")
    glib.timeout_add_seconds = lambda *a, **k: 0
    glib.idle_add = lambda *a, **k: 0
    glib.MainLoop = lambda: None
    gi_repository = types.ModuleType("gi.repository")
    gi_repository.GLib = glib
    gi = types.ModuleType("gi")
    gi.repository = gi_repository

    sys.modules.update({
        "dbus": dbus,
        "dbus.service": dbus_service,
        "dbus.exceptions": dbus_exceptions,
        "dbus.mainloop": dbus_mainloop,
        "dbus.mainloop.glib": dbus_glib,
        "gi": gi,
        "gi.repository": gi_repository,
        "gi.repository.GLib": glib,
    })


_install_dbus_stubs()
