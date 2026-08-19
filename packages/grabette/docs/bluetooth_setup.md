# Grabette — Bluetooth WiFi configuration

A standalone BLE GATT service (`grabette-bluetooth.service`) lets you configure WiFi credentials without SSH or a screen. It's installed + started by `make install-systemd` (see the [README](../README.md)). Once running:

Connect from a phone or laptop via Bluetooth Low Energy on the [BT Tool](https://pollen-robotics.github.io/grabette/) in **Chrome/Edge** and follow those steps:
1. Select Grabette and click on Connect
2. Select your Grabette on the pop-up, then Pair
3. Authenticate with the PIN
4. Scan networks, select your wifi and send WiFi credentials (the tool does the key exchange and seals the password for you).

> Be careful, Chrome may need to enable experimental features : <code>chrome://flags/#enable-experimental-web-platform-features</code>

PIN is configurable via the `GRABETTE_BT_PIN` env var (default: `00000`); set it in `systemd/grabette-bluetooth.service` (`Environment=GRABETTE_BT_PIN=...`) before installing.

**Commands** (written to the COMMAND characteristic as UTF-8; responses arrive as notifications):

| Command | Response |
|---|---|
| `PING` | `PONG` |
| `PIN_xxxxx` | `OK: Connected` / `ERROR: Incorrect PIN` (required before the WIFI commands; rate-limited — 5 wrong PINs lock further attempts for 30s, doubling on each repeat) |
| `WIFI_SCAN` | JSON array of nearby SSIDs (strongest first) |
| `WIFI_KEYEX` | `{"kid","pk","alg"}` — the robot's ephemeral X25519 public key (no auth needed: a public key leaks nothing) |
| `WIFI_CONNECT_ENC <json>` | `OK: Connecting to <ssid>` / `ERROR: ...` — connects via an explicit WPA-PSK profile |
| `WIFI_RESET` | `OK: WiFi connections cleared (<n> removed)` |

The WiFi password is **never sent in clear**. The client runs `WIFI_KEYEX`, derives a shared key from its own ephemeral X25519 key with `HKDF-SHA256` (salt = the PIN), seals the password with `AES-256-GCM` (AAD = the SSID), and sends `{ssid,kid,epk,nonce,ct}` as the `WIFI_CONNECT_ENC` payload. The PIN is therefore both the session auth and the sealing salt; `kid` rejects a payload sealed against a superseded key (`ERROR: Stale key — re-run WIFI_KEYEX`), and a wrong PIN surfaces as `ERROR: Decryption failed (wrong PIN?)`.

`WIFI_CONNECT_ENC` and `WIFI_RESET` **consume** the authentication — re-send `PIN_xxxxx` before another one. `WIFI_SCAN` does not, so a single PIN covers scan-then-connect.

The adapter advertises with `Pairable = True` and uses the `NoInputNoOutput` agent for silent Just Works pairing — required because some centrals (notably Windows and some Linux/BlueZ stacks) refuse GATT operations until they've bonded. macOS clients can still connect "connection-only" without bonding; both modes work.

> **A desynced bond recovers on its own.** If a stored pairing key survives on only one side, every reconnect dies in SMP before any GATT work and the link loops connect→drop every few seconds. After 3 such drops in a row from the same client it is bonded with (short, with no GATT traffic — see `DEADLOCK_STRIKES`) the Pi purges its own key and lets the client pair afresh, so a robot that isn't on WiFi yet can never be locked out of its only provisioning route. If the loop outlives that, the stale key is the *client's*: `bluetoothctl remove <mac.address.of.Grabette>` there, plus Forget the device in `chrome://bluetooth-internals`.
