# Choice Assay User Guide (RPi)

This is the top-level setup guide for running Choice Assay on Raspberry Pi with Expidite.

## 1) Standard Expidite LED GPIO pins

Choice Assay currently uses Expidite's standard LED manager (`manage_leds="Yes"` in `system.cfg`).

- **Green LED**: GPIO **16** (BCM numbering)
- **Red LED**: GPIO **26** (BCM numbering)

Source of truth: Expidite script `expidite_rpi/scripts/led_control.py` defaults:
- `LED_GPIO_PIN` -> `16` (green)
- `LED_GPIO_PIN_RED` -> `26` (red)

You can override these via environment variables if required.

## 2) Flash SD card and install Choice Assay code

### A. Flash Raspberry Pi OS

1. Use **Raspberry Pi Imager**.
2. Select a **64-bit** Raspberry Pi OS image (Expidite requires 64-bit).
3. In advanced options, set at least:
   - hostname
   - user/password
   - SSH enabled
   - (optional) preconfigure Wi-Fi for first boot
4. Write the image to SD card, insert card into Pi, and boot.

### B. Prepare Expidite runtime config on the Pi

On the Pi:

1. Create config directory:

   ```bash
   mkdir -p ~/.expidite
   ```

2. Copy your project `system.cfg` to:

   ```bash
   ~/.expidite/system.cfg
   ```

   Start from this repo file:
   - `src/choice_assay/rpi/system.cfg`

3. Create cloud/Git credentials file:

   ```bash
   ~/.expidite/keys.env
   ```

   (`rpi_installer.sh` requires both `system.cfg` and `keys.env`.)

### C. Ensure Choice Assay points to the correct code entrypoints

In `~/.expidite/system.cfg`, verify these keys:

- `my_git_repo_url="github.com/Bee-Ops-Lab/Choice-assay.git"`
- `my_git_branch="main"`
- `my_fleet_config="choice_assay.rpi.my_fleet_config.INVENTORY"`
- `my_start_script="choice_assay.rpi.run_my_sensor"`

### D. Run installer

Run Expidite installer on the Pi (after Expidite is installed):

```bash
rpi_installer.sh
```

If `rpi_installer.sh` is not yet on PATH, run it from the installed Expidite scripts location.

What this does (summary):
- Creates/activates the virtual env
- Installs OS deps + Expidite + Choice Assay code from your repo
- Configures services/firewall/logging
- Sets startup behavior from `system.cfg`

## 3) Where Wi-Fi configuration lives

Wi-Fi AP/client configuration is in:

- `src/choice_assay/rpi/my_fleet_config.py`

Specifically in `WIFI_CLIENTS`, e.g.:

- `WifiClient(ssid=..., pw=..., priority=...)`

This list is loaded by `run_my_sensor.py` through `root_cfg.load_configuration()` and applied by Expidite DeviceManager.

## 4) Where camera configuration lives

Camera configuration is also in:

- `src/choice_assay/rpi/my_fleet_config.py`

Inside `create_choice_assay_device()` under `RpicamSensorCfg(...)`, especially:

- `sensor_model`
- `sensor_index`
- `outputs`
- `rpicam_cmd` (frame rate, resolution, recording duration)

Current command:

```text
rpicam-vid --framerate 5 --width 1640 --height 1232 -o FILENAME -t 180000
```

## 5) Runtime path actually in use

For current Choice Assay runtime:

- Active start script: `src/choice_assay/rpi/run_my_sensor.py`
- Active fleet config: `src/choice_assay/rpi/my_fleet_config.py`

`my_choice_assay_sensor.py` is intentionally **not** part of this setup.
