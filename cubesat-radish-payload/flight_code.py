"""Autonomous CubeSat payload firmware for Pico 2, AS7341 and SCD-40.

For flight integration, copy this file to ``CIRCUITPY/code.py``. Telemetry and
commands use newline-delimited JSON over UART0, GP0 TX and GP1 RX, at 115200 baud.
The spacecraft OBC supplies the authoritative mission timestamp.
"""

import json
import time

import adafruit_scd4x
import board
import busio
import digitalio
import microcontroller
from adafruit_as7341 import AS7341


DEVICE = "mito-spectral-cubesat"
FIRMWARE_VERSION = "0.4.0"
BROADBAND = -1
DARK = 0
SPECTRAL_CYCLE_INTERVAL_S = 900
ENVIRONMENT_INTERVAL_S = 60
SAMPLES_PER_STATE = 3
SETTLE_S = 0.15
REHYDRATION_PULSE_S = 3.0
REHYDRATION_ARM_WINDOW_S = 60
REHYDRATION_NVM_MARKER = 0xA5

CHANNELS = (415, 445, 480, 515, 555, 590, 630, 680, "clear", "nir")
LED_PINS = {
    365: board.GP6,
    450: board.GP7,
    535: board.GP8,
    550: board.GP9,
    565: board.GP10,
    575: board.GP11,
    605: board.GP12,
    630: board.GP13,
    660: board.GP14,
    700: board.GP15,
    730: board.GP16,
    940: board.GP17,
}
ILLUMINATION_SEQUENCE = (DARK, BROADBAND) + tuple(LED_PINS)

i2c = busio.I2C(board.GP5, board.GP4, frequency=400_000)
uart = busio.UART(
    board.GP0,
    board.GP1,
    baudrate=115200,
    timeout=0.01,
    receiver_buffer_size=2048,
)

spectrometer = AS7341(i2c)
spectrometer.atime = 29
spectrometer.astep = 599
spectrometer.led_current = 20

environment_sensor = adafruit_scd4x.SCD4X(i2c)
environment_sensor.self_calibration_enabled = False
environment_sensor.start_periodic_measurement()

leds = {}
for wavelength, pin in LED_PINS.items():
    output = digitalio.DigitalInOut(pin)
    output.direction = digitalio.Direction.OUTPUT
    output.value = False
    leds[wavelength] = output

rehydration_output = digitalio.DigitalInOut(board.GP18)
rehydration_output.direction = digitalio.Direction.OUTPUT
rehydration_output.value = False
mixing_fan = digitalio.DigitalInOut(board.GP19)
mixing_fan.direction = digitalio.Direction.OUTPUT
mixing_fan.value = False

receive_buffer = bytearray()
last_environment = {
    "co2_ppm": None,
    "temperature_c": None,
    "relative_humidity_pct": None,
}
rehydration_armed_until = 0.0
next_environment = 0.0
next_spectral_cycle = 0.0
cycle_number = 0


def send(payload):
    uart.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))


def all_sources_off():
    spectrometer.led = False
    for output in leds.values():
        output.value = False


def safe_state():
    all_sources_off()
    rehydration_output.value = False
    mixing_fan.value = False


def set_illumination(wavelength_nm):
    all_sources_off()
    if wavelength_nm == BROADBAND:
        spectrometer.led = True
    elif wavelength_nm in leds:
        leds[wavelength_nm].value = True
    elif wavelength_nm != DARK:
        raise ValueError("unsupported illumination wavelength")


def read_spectrum():
    totals = [0.0] * len(CHANNELS)
    for sample_index in range(SAMPLES_PER_STATE):
        visible = tuple(float(value) for value in spectrometer.all_channels)
        values = visible + (
            float(spectrometer.channel_clear),
            float(spectrometer.channel_nir),
        )
        for index, value in enumerate(values):
            totals[index] += value
        if sample_index + 1 < SAMPLES_PER_STATE:
            time.sleep(0.05)
    return [value / SAMPLES_PER_STATE for value in totals]


def read_environment():
    global last_environment
    mixing_fan.value = True
    time.sleep(1.0)
    try:
        if environment_sensor.data_ready:
            last_environment = {
                "co2_ppm": int(environment_sensor.CO2),
                "temperature_c": float(environment_sensor.temperature),
                "relative_humidity_pct": float(environment_sensor.relative_humidity),
            }
    finally:
        mixing_fan.value = False
    return last_environment


def run_spectral_cycle():
    global cycle_number
    spectra = []
    try:
        for illumination in ILLUMINATION_SEQUENCE:
            set_illumination(illumination)
            time.sleep(SETTLE_S)
            spectra.append(
                {
                    "illumination_nm": illumination,
                    "signals": read_spectrum(),
                }
            )
    finally:
        all_sources_off()
    send(
        {
            "type": "spectral_cycle",
            "device": DEVICE,
            "mission_elapsed_s": time.monotonic(),
            "cycle": cycle_number,
            "channels": list(CHANNELS),
            "spectra": spectra,
            "environment": read_environment(),
        }
    )
    cycle_number += 1


def rehydrate_once():
    if microcontroller.nvm[0] == REHYDRATION_NVM_MARKER:
        raise RuntimeError("rehydration already completed")
    rehydration_output.value = True
    try:
        time.sleep(REHYDRATION_PULSE_S)
    finally:
        rehydration_output.value = False
    microcontroller.nvm[0] = REHYDRATION_NVM_MARKER
    send(
        {
            "type": "event",
            "event": "rehydration_completed",
            "mission_elapsed_s": time.monotonic(),
        }
    )


def handle_command(command):
    global rehydration_armed_until, next_spectral_cycle
    name = command.get("command")
    if name == "hello":
        send(
            {
                "type": "hello",
                "device": DEVICE,
                "firmware_version": FIRMWARE_VERSION,
                "channels": list(CHANNELS),
                "illuminations_nm": list(ILLUMINATION_SEQUENCE),
                "rehydrated": microcontroller.nvm[0] == REHYDRATION_NVM_MARKER,
            }
        )
    elif name == "arm_rehydration":
        rehydration_armed_until = time.monotonic() + REHYDRATION_ARM_WINDOW_S
        send({"type": "ack", "command": name, "armed_for_s": REHYDRATION_ARM_WINDOW_S})
    elif name == "rehydrate":
        if time.monotonic() > rehydration_armed_until:
            raise RuntimeError("rehydration is not armed")
        rehydration_armed_until = 0.0
        rehydrate_once()
    elif name == "run_spectral_cycle":
        next_spectral_cycle = 0.0
        send({"type": "ack", "command": name})
    elif name == "get_environment":
        send({"type": "environment", **read_environment()})
    else:
        raise ValueError("unknown command")


def poll_commands():
    if uart.in_waiting:
        incoming = uart.read(uart.in_waiting)
        if incoming:
            receive_buffer.extend(incoming)
    while b"\n" in receive_buffer:
        line, _, remainder = receive_buffer.partition(b"\n")
        receive_buffer[:] = remainder
        if line:
            handle_command(json.loads(line.decode("utf-8")))


safe_state()
send({"type": "boot", "device": DEVICE, "firmware_version": FIRMWARE_VERSION})

while True:
    now = time.monotonic()
    try:
        poll_commands()
        if now >= next_environment:
            send(
                {
                    "type": "environment",
                    "mission_elapsed_s": now,
                    **read_environment(),
                }
            )
            next_environment = now + ENVIRONMENT_INTERVAL_S
        if now >= next_spectral_cycle:
            run_spectral_cycle()
            next_spectral_cycle = time.monotonic() + SPECTRAL_CYCLE_INTERVAL_S
    except Exception as error:
        safe_state()
        send({"type": "error", "mission_elapsed_s": now, "error": str(error)})
    time.sleep(0.02)
