#!/usr/bin/env python3
"""Control one or two Grabette recorders over a trusted LAN (Python 3.11+).

Concurrent requests are software triggers, not synchronized camera exposures.
Only fresh, increasing counters confirm data receipt; saved files need a
separate integrity check. No SSH, Hugging Face login, or extra packages needed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


COUNTERS = ("frame_count", "imu_sample_count", "angle_sample_count", "head_frame_count")
BUSY = ("is_capturing", "is_starting", "is_saving", "pending_saves", "saving_episode_ids", "buffer_bytes")
EPISODE = re.compile(r"\d{8}_\d{6}\Z")


class ControlError(RuntimeError):
    def __init__(self, message, *, uncertain=False, status=None):
        super().__init__(message)
        self.uncertain, self.status = uncertain, status


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None  # A control request must stay on the configured recorder.


def load_devices(path):
    devices = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(devices, dict) or not devices or set(devices) - {"left", "right"}:
        raise ValueError("Configure left, right, or both recorder objects")
    origins = set()
    for side, cfg in devices.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"{side}: expected a recorder object")
        if not isinstance(cfg.get("api"), str):
            raise ValueError(f"{side}: api must be a string")
        url = urllib.parse.urlsplit(cfg["api"])
        try:
            url.port
        except ValueError as exc:
            raise ValueError(f"{side}: invalid API port") from exc
        if (url.scheme not in ("http", "https") or not url.hostname or
                url.username is not None or url.password is not None or url.query or url.fragment or
                url.path not in ("", "/")):
            raise ValueError(f"{side}: api must be an HTTP(S) origin without credentials or a path")
        cfg["api"] = cfg["api"].rstrip("/")
        if cfg["api"] in origins:
            raise ValueError("Both hands cannot use the same API origin")
        origins.add(cfg["api"])
        if cfg.get("expected_hand", side) != side:
            raise ValueError(f"{side}: expected_hand must match the configured side")
        if "expected_device_id" in cfg and (not isinstance(cfg["expected_device_id"], str) or not cfg["expected_device_id"]):
            raise ValueError(f"{side}: expected_device_id must be a nonempty string")
        counters = cfg.get("required_counters", [])
        if not isinstance(counters, list) or any(x not in COUNTERS for x in counters):
            raise ValueError(f"{side}: invalid required_counters")
    return devices


class HttpApi:
    def __init__(self, devices):
        self.devices = devices

    def request(self, side, path, body=None, *, method=None):
        method = method or ("GET" if body is None else "POST")
        request = urllib.request.Request(self.devices[side]["api"] + path,
            data=None if body is None else json.dumps(body).encode(), method=method,
            headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=120 if method != "GET" else 3) as response:
                result = json.load(response)
            if not isinstance(result, dict) and not (method == "GET" and path == "/api/tasks" and isinstance(result, list)):
                raise ValueError("Expected an endpoint-specific JSON object or task list")
            return result
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read(4096)).get("detail", exc.reason)
            except (ValueError, AttributeError):
                detail = exc.reason
            raise ControlError(f"HTTP {exc.code}: {str(detail)[:300]}",
                uncertain=method != "GET" and exc.code >= 500, status=exc.code) from exc
        except (OSError, ValueError, urllib.error.URLError) as exc:
            # A lost mutation reply does not prove that the device refused it.
            raise ControlError(f"Connection/reply error: {type(exc).__name__}",
                uncertain=method != "GET") from exc


def capture(status):
    sensor = status.get("sensor") or {}
    value = sensor.get("capture", status.get("capture", {}))
    if not isinstance(value, dict):
        raise ControlError("Malformed capture state")
    return value


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def parallel(sides, call):
    def run(side):
        try:
            return call(side), None
        except Exception as exc:
            return None, exc
    with ThreadPoolExecutor(max_workers=len(sides)) as executor:
        return dict(zip(sides, executor.map(run, sides)))


class EventLog:
    def __init__(self, directory):
        self.path = Path(directory) / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + ".jsonl")
        self.lock, self.error = threading.Lock(), ""

    def write(self, event, **fields):
        try:
            with self.lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(),
                        monotonic=time.monotonic(), **fields), ensure_ascii=False) + "\n")
        except OSError as exc:
            self.error = f"Local log unavailable ({type(exc).__name__}); control remains available"


class Recorder:
    """Observation state uses workstation monotonic time, never device wall time."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.identity = None
        self.stale_seconds, self.system_stale_seconds = 5, 10
        self.required = {"frame_count", *cfg.get("required_counters", [])}
        self.status, self.system, self.session, self.saved = {}, {}, {}, {}
        self.at = self.system_at = None
        self.request_started = -1.0
        self.error = self.system_error = ""
        self.last_episode = None
        self.previous = {}
        self.advanced = {}
        self.first = None
        self.pending = ""
        self.pending_token = None
        self.start_inflight = False
        self.start_unknown = False
        self.awaiting_start_observation = False
        self.control_error = ""
        self.save_error = ""
        self.deleted_episodes = set()
        self.save_idle_seconds = 0.0
        self.was_saving_idle = False
        self.angle_timestamp = None
        self.angle_changed_at = None

    def observe(self, status, now, requested=None):
        requested = now if requested is None else requested
        if requested < self.request_started:
            return False
        if not isinstance(status, dict):
            raise ControlError("Malformed daemon status")
        cap = capture(status)
        old = capture(self.status) if self.status else {}
        continuous = self.at is not None and not self.error and 0 <= now - self.at < self.stale_seconds
        elapsed = now - self.at if continuous else 0
        if not continuous or cap.get("episode_id") != old.get("episode_id") or not cap.get("is_capturing"):
            self.previous, self.advanced, self.first = {}, {}, now
        values = {key: cap.get(key) for key in self.required}
        if any(number(value) is None or value < 0 for value in values.values()):
            raise ControlError("Missing/invalid required capture counters")
        if any(value < self.previous.get(key, value) for key, value in values.items()):
            self.previous, self.advanced, self.first = {}, {}, now
        if self.previous:
            for key, value in values.items():
                if value > self.previous.get(key, value):
                    self.advanced[key] = now
        self.previous = values
        if cap.get("episode_id") and cap["episode_id"] not in self.deleted_episodes:
            if cap["episode_id"] != self.last_episode:
                self.saved, self.save_error = {}, ""
            self.last_episode = cap["episode_id"]
        if cap.get("is_capturing"):
            self.start_unknown = False
            self.awaiting_start_observation = False
        saving_idle = bool(not cap.get("is_capturing") and not cap.get("is_starting") and
                           any(cap.get(k) for k in BUSY[2:]))
        self.save_idle_seconds = (self.save_idle_seconds + elapsed if saving_idle and self.was_saving_idle else 0)
        self.was_saving_idle = saving_idle
        angle = (status.get("sensor") or {}).get("angle") or {}
        timestamp = angle.get("timestamp_ms")
        if not continuous or timestamp != self.angle_timestamp:
            self.angle_changed_at = now
        self.angle_timestamp = timestamp
        self.status, self.at, self.error = status, now, ""
        self.request_started = requested
        return True

    def phase(self, now):
        if self.control_error:
            return "UNVERIFIED: control disabled; check recorder identity/protocol"
        if self.at is None or self.error or now - self.at >= self.stale_seconds:
            return "UNKNOWN / disconnected" + ("; start result unknown" if self.start_unknown else "")
        cap = capture(self.status)
        saved = self.saved if self.last_episode and self.saved.get("episode_id") == self.last_episode else {}
        if self.status.get("state") != "running":
            return "Recorder service is not ready"
        if self.status.get("hardware_error") or cap.get("capture_fault"):
            return "FAULT: inspect recorder"
        if self.pending:
            return self.pending
        if cap.get("is_starting"):
            return "PREPARING: data receipt not confirmed"
        if cap.get("is_capturing"):
            if any(now - self.advanced.get(k, self.first if self.first is not None else now) >= self.stale_seconds for k in self.required):
                return "CAPTURING: selected counters have stalled"
            if (isinstance(cap.get("episode_id"), str) and EPISODE.fullmatch(cap["episode_id"]) and
                    all((number(cap.get(k)) or 0) > 0 and k in self.advanced for k in self.required)):
                return "CAPTURING: selected data counters growing"
            return "CAPTURING: waiting for selected data counters to grow"
        if self.start_unknown:
            return "Start result unknown; no automatic retry"
        if self.awaiting_start_observation:
            return "Start acknowledged; waiting for fresh recorder state"
        if any(cap.get(k) for k in BUSY[2:]):
            return "STOPPED: background saving"
        if saved.get("save_state") in ("recording", "saving"):
            return "STOPPED: waiting for saved-file publication"
        if cap.get("saving_error") or saved.get("save_state") in ("failed", "unknown") or saved.get("metadata_ok") is False:
            return "STOPPED: saved-file error; inspect episode"
        if saved.get("save_state") == "complete" and saved.get("metadata_ok") is True:
            return "SAVED: recorder validation completed"
        if saved.get("has_video"):
            return "STOPPED: files present (legacy API; integrity not independently checked)"
        return "IDLE: ready" if not cap.get("blocked_reason") else "BLOCKED: " + str(cap["blocked_reason"])

    def warnings(self, now):
        result = []
        if self.control_error:
            result.append("Control disabled: " + self.control_error)
        if self.save_error:
            result.append("Saved-file status unavailable: " + self.save_error)
        cap = capture(self.status)
        if self.error:
            result.append(self.error)
        for key in ("error", "hardware_error"):
            if self.status.get(key): result.append(str(self.status[key]))
        for key in ("capture_fault", "saving_error", "blocked_reason"):
            if cap.get(key): result.append(str(cap[key]))
        if self.save_idle_seconds >= 300:
            result.append(f"Saving has used {self.save_idle_seconds:.0f}s of idle time; inspect progress")
        used, limit = number(cap.get("buffer_bytes")), number(cap.get("buffer_limit_bytes"))
        if used is not None and limit and used / limit >= .85:
            result.append(f"Buffer {used / limit:.0%} full")
        if self.system_at is None or self.system_error or now - self.system_at >= self.system_stale_seconds:
            result.append("System telemetry unavailable/stale")
        else:
            sysinfo = self.system
            for field, low, label in (("disk_free_gb", 10, "Disk space low"),
                                     ("mem_available_mb", 1024, "Available memory low")):
                value = number(sysinfo.get(field))
                if value is not None and value < low: result.append(f"{label}: {value:.0f}")
            temperature, battery = number(sysinfo.get("cpu_temp_c")), number(sysinfo.get("battery_pct"))
            if temperature is not None and temperature >= 75: result.append(f"CPU temperature {temperature:.1f} C")
            if battery is not None and battery <= 25 and sysinfo.get("battery_charging") is not True:
                result.append(f"Battery {battery:.0f}% (external power not confirmed)")
        if (self.at is not None and not self.error and now - self.at < self.stale_seconds and
                self.angle_timestamp is not None and self.angle_changed_at is not None and now - self.angle_changed_at >= self.stale_seconds):
            result.append(f"Angle sensor timestamp has not advanced for {self.stale_seconds:.0f}s")
        return result


class Controller:
    def __init__(self, devices, log, api=None, clock=time.monotonic):
        self.api, self.log, self.clock = api or HttpApi(devices), log, clock
        self.views = {side: Recorder(cfg) for side, cfg in devices.items()}
        self.lock, self.shutdown = threading.RLock(), threading.Event()
        self.delete_plan = None
        self.message = "1/2/3 start left/right/both; wait for increasing counters before acting"

    def check_identity(self, side, reconnect=False):
        try:
            return self._check_identity(side, reconnect=reconnect)
        except Exception as exc:
            with self.lock: self.views[side].control_error = str(exc)
            raise

    def _check_identity(self, side, reconnect=False):
        info = self.api.request(side, "/api/capture/control-info")
        view = self.views[side]
        if (not isinstance(info, dict) or type(info.get("local_only_control_version")) is not int or
                info["local_only_control_version"] != 1 or type(info.get("expected_episode_stop_version")) is not int or
                info["expected_episode_stop_version"] != 1):
            raise ControlError("Recorder needs the local-control API from this change; control refused")
        if info.get("hand") != side or not isinstance(info.get("device_id"), str) or not info["device_id"]:
            raise ControlError("Recorder identity/hand does not match configuration")
        expected = view.cfg.get("expected_device_id")
        if expected and expected != info["device_id"]:
            raise ControlError("Recorder device ID does not match configuration")
        with self.lock:
            if view.identity and view.identity["device_id"] != info["device_id"] and not reconnect:
                raise ControlError("Recorder identity changed; reconnect and verify configuration")
            required = info.get("required_capture_counters", ["frame_count"])
            if not isinstance(required, list) or any(k not in COUNTERS for k in required):
                raise ControlError("Invalid recorder counter capabilities")
            view.required = {"frame_count", *required, *view.cfg.get("required_counters", [])}
            view.identity = info
            view.control_error = ""
        return info

    def refresh(self, side):
        started = self.clock()
        try:
            status = self.api.request(side, "/api/daemon/status")
            with self.lock:
                self.views[side].observe(status, self.clock(), requested=started)
        except Exception as exc:
            with self.lock:
                if started >= self.views[side].request_started:
                    self.views[side].error = str(exc)
            raise
        return status

    def poll(self, side):
        try:
            status = self.refresh(side)
        except Exception:
            return
        with self.lock:
            view = self.views[side]
            episode = view.last_episode
        if episode and not capture(status).get("is_capturing") and not capture(status).get("is_starting"):
            try:
                info = self.api.request(side, "/api/episodes/" + urllib.parse.quote(episode, safe=""))
                if not isinstance(info, dict) or info.get("episode_id") != episode:
                    raise ControlError("Saved-file reply does not match the selected episode")
                with self.lock:
                    if view.last_episode == episode:
                        view.saved, view.save_error = info, ""
            except Exception as exc:
                with self.lock: view.save_error = str(exc)

    def telemetry(self, side):
        try:
            info = self.api.request(side, "/api/system/info")
            if not isinstance(info, dict): raise ControlError("Malformed system telemetry")
            if "mem_available_mb" not in info and "mem_available_mib" in info:
                info["mem_available_mb"] = info["mem_available_mib"]
            with self.lock:
                view = self.views[side]
                view.system, view.system_at, view.system_error = info, self.clock(), ""
        except Exception as exc:
            with self.lock: self.views[side].system_error = str(exc)
        try:
            session = self.api.request(side, "/api/session/status")
            if not isinstance(session, dict): raise ControlError("Malformed task-session status")
            with self.lock: self.views[side].session = session
        except Exception:
            with self.lock: self.views[side].session = {}

    def run_pollers(self, interval):
        for view in self.views.values():
            view.stale_seconds = max(5, interval * 2.5)
            view.system_stale_seconds = max(10, interval * 2.5)
        def loop(side, telemetry=False):
            while not self.shutdown.is_set():
                (self.telemetry if telemetry else self.poll)(side)
                self.shutdown.wait(max(3, interval) if telemetry else interval)
        for side in self.views:
            for auxiliary in (False, True):
                threading.Thread(target=loop, args=(side, auxiliary), daemon=True).start()

    def ready_start(self, side):
        self.check_identity(side)
        status = self.refresh(side)
        if status.get("state") != "running" or any(capture(status).get(k) for k in ("is_capturing", "is_starting")):
            raise ControlError(f"{side}: recorder is not ready for a new capture")
        if status.get("error") or status.get("hardware_error") or capture(status).get("blocked_reason") or capture(status).get("capture_fault"):
            raise ControlError(f"{side}: capture is blocked")
        return status

    def idle(self, side, episode=None):
        status = self.ready_start(side)
        if any(capture(status).get(k) for k in BUSY):
            raise ControlError(f"{side}: recorder is not idle; wait for capture/save to finish")
        if episode is not None:
            info = self.api.request(side, "/api/episodes/" + urllib.parse.quote(episode, safe=""))
            if info.get("episode_id") != episode or info.get("save_state") not in (None, "complete", "failed", "legacy"):
                raise ControlError("Selected episode is still being written or has an unknown state")
        return status

    def start(self, sides):
        preflight = parallel(sides, self.ready_start)
        if any(error for _, error in preflight.values()):
            raise ControlError("; ".join(f"{s}: {e}" for s, (_, e) in preflight.items() if e))
        def send(side):
            self.check_identity(side)
            try:
                reply = self.api.request(side, "/api/episodes/start", {"local_only": True})
                if not isinstance(reply, dict) or not isinstance(reply.get("episode_id"), str) or not EPISODE.fullmatch(reply["episode_id"]):
                    raise ControlError("Start reply lacks a valid episode ID", uncertain=True)
                with self.lock:
                    self.views[side].last_episode, self.views[side].saved = reply["episode_id"], {}
                    self.views[side].deleted_episodes.discard(reply["episode_id"])
                    self.views[side].awaiting_start_observation = True
                    self.views[side].request_started = max(self.views[side].request_started, self.clock())
                return reply
            except ControlError as exc:
                if exc.uncertain:
                    with self.lock: self.views[side].start_unknown = True
                raise
        results = parallel(sides, send)
        failures = {s: str(e) for s, (_, e) in results.items() if e}
        self.log.write("start_replies", results={s: reply for s, (reply, _) in results.items()}, errors=failures)
        if failures:
            # Roll back only positively acknowledged starts. Unknown results
            # are never retried or treated as proof of failure.
            for side, (reply, _) in results.items():
                if reply:
                    try: self.stop(side, owned_episode=reply["episode_id"])
                    except Exception as exc: self.log.write("rollback_uncertain", side=side, error=str(exc))
            raise ControlError(f"Partial/unknown start: {failures}. Rollback attempted only for acknowledged, still-matching episodes")
        return "Start acknowledged; wait for fresh, increasing counters on each selected recorder"

    def stop(self, side, owned_episode=None):
        deadline = self.clock() + 95
        while True:
            self.check_identity(side)
            status = self.refresh(side)
            cap = capture(status)
            episode = cap.get("episode_id")
            if owned_episode and episode != owned_episode:
                return "Rollback skipped: current episode changed"
            if cap.get("is_capturing") and not cap.get("is_starting"):
                break
            if not cap.get("is_starting") and not self.views[side].start_unknown and not self.views[side].start_inflight:
                return f"{side}: no active recording"
            if self.clock() >= deadline:
                raise ControlError("Start is still unresolved; stop was not sent blindly", uncertain=True)
            if self.shutdown.wait(.3): return "Monitor exited before stop was sent"
        self.check_identity(side)
        reply = self.api.request(side, "/api/episodes/stop", {"local_only": True, "expected_episode_id": episode})
        with self.lock:
            self.views[side].last_episode, self.views[side].saved = episode, {}
            self.views[side].start_unknown = False
            self.views[side].awaiting_start_observation = False
            self.views[side].request_started = max(self.views[side].request_started, self.clock())
        self.log.write("stop_reply", side=side, episode_id=episode, reply=reply)
        return f"{side}: stop acknowledged; wait for saved-file status"

    def prepare_delete(self, sides):
        targets = {}
        for side in sides:
            self.idle(side)
            tasks = self.api.request(side, "/api/tasks")
            if not isinstance(tasks, list) or any(not isinstance(t, dict) for t in tasks):
                raise ControlError("Malformed task list")
            ids = [eid for task in tasks for eid in task.get("episode_ids", [])
                   if isinstance(eid, str) and EPISODE.fullmatch(eid)]
            if not ids: raise ControlError(f"{side}: no saved episodes")
            targets[side] = max(ids)
            self.idle(side, targets[side])
        with self.lock:
            self.delete_plan = {"targets": targets, "created": self.clock()}
        return "Review the fixed episode IDs; y permanently deletes them, n cancels (expires in 30s)"

    def delete(self):
        with self.lock:
            plan, self.delete_plan = self.delete_plan, None
        if not plan or self.clock() - plan["created"] >= 30:
            raise ControlError("No current deletion confirmation; select episodes again")
        # Check every hand before deleting any; never reselect 'the latest'.
        for side, episode in plan["targets"].items(): self.idle(side, episode)
        if self.clock() - plan["created"] >= 30:
            raise ControlError("Deletion confirmation expired during preflight; nothing deleted")
        def send(side):
            episode = plan["targets"][side]
            self.check_identity(side)
            if self.clock() - plan["created"] >= 30:
                raise ControlError("Deletion confirmation expired; nothing deleted on this hand")
            reply = self.api.request(side, "/api/episodes/" + urllib.parse.quote(episode, safe=""), method="DELETE")
            if reply.get("deleted") != episode: raise ControlError("Deletion reply not confirmed", uncertain=True)
            with self.lock:
                self.views[side].deleted_episodes.add(episode)
                if self.views[side].last_episode == episode:
                    self.views[side].last_episode, self.views[side].saved = None, {}
            return episode
        result = parallel(tuple(plan["targets"]), send)
        self.log.write("delete_results", targets=plan["targets"], results={s: (r, str(e) if e else None) for s, (r, e) in result.items()})
        return "; ".join(f"{s}: {r + ' deleted' if not e else 'deletion UNKNOWN/failed: ' + str(e)}" for s, (r, e) in result.items())

    def dispatch(self, action, sides):
        sides = tuple(s for s in sides if s in self.views)
        if not sides: self.message = "That hand is not configured"; return
        with self.lock:
            active = [v.pending for v in self.views.values() if v.pending]
            if active and not (action == "stop" and all(value.startswith("START") for value in active)):
                self.message = "A request is pending; wait for its result"; return
            if action == "stop":
                self.delete_plan = None
            if self.delete_plan and action != "delete":
                self.message = "Cancel or confirm the displayed deletion first"; return
            if action == "start" and any(self.views[s].start_unknown or self.views[s].awaiting_start_observation for s in sides):
                self.message = "Start result remains unknown; inspect status before sending another start"; return
            token = uuid.uuid4().hex
            for side in sides:
                self.views[side].pending = action.upper() + " request pending"
                self.views[side].pending_token = token
                if action == "start": self.views[side].start_inflight = True
        def worker():
            try:
                self.log.write("command", action=action, sides=sides)
                if action == "start": message = self.start(sides)
                elif action == "stop":
                    result = parallel(sides, self.stop)
                    message = "; ".join(str(e or r) for r, e in result.values())
                elif action == "prepare_delete": message = self.prepare_delete(sides)
                else: message = self.delete()
                with self.lock: self.message = message
            except Exception as exc:
                with self.lock: self.message = str(exc)
                self.log.write("command_error", action=action, error=str(exc))
            finally:
                with self.lock:
                    for side in sides:
                        if action == "start": self.views[side].start_inflight = False
                        if self.views[side].pending_token == token:
                            self.views[side].pending = ""
        threading.Thread(target=worker, daemon=True).start()

    def lines(self):
        now = self.clock()
        with self.lock:
            lines = ["GRABETTE LAN CONTROL | software triggers, not exposure synchronization",
                     "1 left / 2 right / 3 both: START | 4 left / 5 right / 6 or s both: STOP",
                     "7 left / 8 right / 9 both: DELETE LAST | y confirm / n cancel | r reconnect | q quit", ""]
            for side, view in self.views.items():
                cap = capture(view.status)
                counters = cap
                if (not cap.get("is_capturing") and not cap.get("is_starting") and view.last_episode and
                        view.saved.get("episode_id") == view.last_episode and view.saved.get("metadata_ok") is True):
                    counters = view.saved
                duration = number(counters.get("duration_seconds", cap.get("duration_seconds"))) or 0
                lines += [f"{side.upper()}: {view.phase(now)}", f"  Episode {view.last_episode or '-'} | {duration:.1f}s | " +
                          " | ".join(label + " " + str(counters.get(k, "?")) for k, label in
                              zip(COUNTERS, ("Main", "IMU", "Angles", "Head"))),
                          "  " + " | ".join(f"{label} {view.system.get(k, '?')}{unit}" for k, label, unit in
                              (("battery_pct", "Battery", "%"), ("battery_charging", "Charging", ""),
                               ("cpu_temp_c", "CPU", " C"), ("disk_free_gb", "Disk free", " GiB"),
                               ("mem_available_mb", "Memory free", " MiB"))),
                          f"  Pending saves {cap.get('pending_saves', '?')} | buffer {cap.get('buffer_bytes', '?')}/{cap.get('buffer_limit_bytes', '?')} bytes"]
                if view.session.get("active"): lines.append("  Task session: " + str(view.session.get("task_id", "active")))
                lines.extend("  ! " + warning for warning in view.warnings(now))
                lines.append("")
            if self.delete_plan:
                if now - self.delete_plan["created"] >= 30:
                    self.delete_plan = None
                    self.message = "Deletion confirmation expired; nothing deleted"
                else:
                    lines += ["DELETE " + side + ": " + episode for side, episode in self.delete_plan["targets"].items()]
            lines += [self.message, self.log.error or "Log: " + str(self.log.path),
                      "Quit leaves recorder acquisition running. File completeness is separate from counter growth."]
            return ["".join(c if c.isprintable() else " " for c in str(line)) for line in lines]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Recorder JSON configuration")
    parser.add_argument("--logs", type=Path, default=Path.home() / ".local/state/grabette-control")
    parser.add_argument("--interval", type=float, default=1, help="Capture poll interval, seconds (minimum 0.3)")
    parser.add_argument("--once", action="store_true", help="Read-only status snapshot; no capture mutations")
    parser.add_argument("--json", action="store_true", help="JSON output with --once")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < .3: parser.error("--interval must be finite and >= 0.3")
    if args.json and not args.once: parser.error("--json requires --once")
    try: devices = load_devices(args.config)
    except (OSError, ValueError) as exc: parser.error(str(exc))
    controller = Controller(devices, EventLog(args.logs))
    def connect(side):
        try: controller.check_identity(side, reconnect=True)
        except Exception as exc:
            with controller.lock: controller.views[side].control_error = str(exc)
    parallel(tuple(devices), connect)
    if args.once:
        parallel(tuple(devices), lambda side: (controller.poll(side), controller.telemetry(side)))
        if args.json:
            print(json.dumps({s: dict(status=v.phase(controller.clock()), warnings=v.warnings(controller.clock()),
                identity=v.identity, control_error=v.control_error,
                daemon=v.status, system=v.system, session=v.session, saved=v.saved) for s, v in controller.views.items()}))
        else: print("\n".join(controller.lines()))
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty(): parser.error("Interactive mode needs a POSIX terminal; use --once for scripts")
    try:
        import termios
        import tty
    except ImportError: parser.error("Interactive mode requires macOS or Linux; use --once on other platforms")
    original = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        controller.run_pollers(args.interval)
        while True:
            size = shutil.get_terminal_size((110, 35))
            screen = []
            for line in controller.lines():
                screen.extend(textwrap.wrap(line, width=max(1, size.columns - 1), subsequent_indent="  ") or [""])
            print("\033[H\033[2J" + "\n".join(screen), flush=True)
            if not select.select([sys.stdin], [], [], .25)[0]: continue
            raw = os.read(sys.stdin.fileno(), 1)
            if not raw: break
            key = raw.decode("utf-8", "ignore").lower()
            if key in ("q", "\x03", "\x04"):
                if any(v.pending.startswith("DELETE") for v in controller.views.values()):
                    controller.message = "Deletion is in progress; wait for its result before quitting"; continue
                break
            if key in "123456789" and key:
                index = int(key) - 1
                sides = (("left",), ("right",), tuple(devices))[index % 3]
                controller.dispatch(("start", "stop", "prepare_delete")[index // 3], sides)
            elif key == "s": controller.dispatch("stop", tuple(devices))
            elif key == "y": controller.dispatch("delete", tuple(devices))
            elif key == "n":
                with controller.lock: controller.delete_plan = None
            elif key == "r": parallel(tuple(devices), connect)
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown.set()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        print("\nMonitor exited. Acquisition was NOT automatically stopped. Last observed state:\n" + "\n".join(controller.lines()[4:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
