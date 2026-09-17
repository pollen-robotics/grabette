"""Regression tests for the LE advertising interval.

The MGMT "Add Advertising" command carries no interval, so the kernel applied
its own default of 1.28s. At that rate a central discovers the robot
erratically AND cannot connect at all: a connection can only be opened in
answer to an advertisement, so the attempt times out and is cancelled locally
(le-connection-abort-by-local) without the robot ever seeing it — which looks
exactly like a dead robot. Measured on grabette-simsim (Pi 4 / CYW4345):
4 advertisement reports per 45s and no usable connection at 1.28s, ~19 reports
and a first-try connect at 100-150ms. These tests pin the interval to the wire.
"""
import pytest

from grabette.bluetooth import bluetooth_service as bts


class _BtmgmtStub:
    """Records btmgmt argv and answers with real btmgmt output per subcommand."""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = fail

    def __call__(self, *args):
        self.calls.append(args)
        sub = args[0]
        if sub in self.fail:
            return "Add Advertising failed with status 0x0d (Invalid Parameters)"
        if sub == "add-ext-adv-params":
            # This subcommand reports capabilities, not "Instance added".
            return "Tx Power: 127\nMax adv data len: 28\nMax scan resp len: 31"
        return "Instance added: 1"

    def argv(self, sub):
        """The first call to `sub`, or None if it was never invoked."""
        return next((call for call in self.calls if call[0] == sub), None)


@pytest.fixture
def service(monkeypatch):
    """Service whose btmgmt calls are recorded on `service.btmgmt`."""
    svc = bts.BluetoothWifiService(device_name="Grabette", pin_code="12345")
    svc._hci_index = 0
    stub = _BtmgmtStub()
    monkeypatch.setattr(svc, "_btmgmt", stub)
    svc.btmgmt = stub
    return svc


# ---- the interval reaches the controller ----

def test_advertises_with_an_explicit_interval(service):
    service._mgmt_advertise()
    argv = service.btmgmt.argv("add-ext-adv-params")
    assert argv is not None, "fell back to add-adv, which carries no interval"
    assert argv[argv.index("-r") + 1] == str(bts.ADV_MIN_INTERVAL)
    assert argv[argv.index("-x") + 1] == str(bts.ADV_MAX_INTERVAL)


def test_interval_is_fast_enough_to_be_connectable(service):
    # Units of 0.625ms. The kernel default that broke provisioning is
    # 0x0800 (1.28s); 32-480 is the normal fast-discoverable range (20-300ms).
    assert 32 <= bts.ADV_MIN_INTERVAL <= bts.ADV_MAX_INTERVAL <= 480


def test_name_still_goes_out_in_the_scan_response(service):
    # The web client filters by name prefix, so losing the scan response makes
    # the robot unfindable even while it advertises perfectly.
    service._mgmt_advertise()
    argv = service.btmgmt.argv("add-ext-adv-data")
    assert argv[argv.index("-s") + 1] == service._advertised_scan_rsp_hex()


def test_clears_the_previous_instance_first(service):
    service._mgmt_advertise()
    assert service.btmgmt.calls[0] == ("rm-adv", "1")


# ---- the fallback, for controllers/BlueZ that reject the ext commands ----

def test_falls_back_to_add_adv_when_ext_params_is_rejected(service, monkeypatch):
    # BlueZ <5.65 has no add-ext-adv-params at all. Advertising slowly beats
    # not advertising: this service is a robot's only route onto WiFi.
    service.btmgmt.fail = ("add-ext-adv-params",)
    tuned = []
    monkeypatch.setattr(service, "_set_kernel_adv_interval", lambda: tuned.append(True))
    service._mgmt_advertise()
    assert service.btmgmt.argv("add-adv") is not None
    # add-adv brings no interval of its own, so the kernel default is the only
    # remaining place to set one.
    assert tuned == [True]


def test_fallback_clears_the_half_built_instance(service, monkeypatch):
    # add-ext-adv-params may be accepted and add-ext-adv-data then rejected,
    # leaving instance 1 registered with no name in it.
    service.btmgmt.fail = ("add-ext-adv-data",)
    monkeypatch.setattr(service, "_set_kernel_adv_interval", lambda: True)
    service._mgmt_advertise()
    assert [c for c in service.btmgmt.calls if c[0] == "rm-adv"] == [
        ("rm-adv", "1"),
        ("rm-adv", "1"),
    ]
    assert service.btmgmt.argv("add-adv") is not None


def test_kernel_interval_write_targets_this_adapter(service, tmp_path, monkeypatch):
    monkeypatch.setattr(
        bts, "ADV_INTERVAL_DEBUGFS", str(tmp_path / "hci{index}_adv_{bound}_interval")
    )
    for bound in ("min", "max"):
        (tmp_path / f"hci0_adv_{bound}_interval").write_text("2048")
    assert service._set_kernel_adv_interval() is True
    assert (tmp_path / "hci0_adv_min_interval").read_text() == str(bts.ADV_MIN_INTERVAL)
    assert (tmp_path / "hci0_adv_max_interval").read_text() == str(bts.ADV_MAX_INTERVAL)


def test_kernel_interval_write_survives_a_missing_debugfs(service, monkeypatch):
    # debugfs isn't guaranteed mounted; this runs on the mainloop thread, so
    # raising here would stop re-advertising entirely and the Pi would go dark.
    monkeypatch.setattr(bts, "ADV_INTERVAL_DEBUGFS", "/nonexistent/hci{index}_{bound}")
    assert service._set_kernel_adv_interval() is False
