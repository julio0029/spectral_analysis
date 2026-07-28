"""Mito-spectral acquisition firmware for Raspberry Pi Pico + AS7341.

Copy this file, boot.py and the required CircuitPython libraries to CIRCUITPY.
The desktop GUI communicates over the USB CDC data port with JSON lines.
"""

import json
import time

import board
import busio
import digitalio
import usb_cdc
from adafruit_as7341 import AS7341


DEVICE = "mito-spectral-pico"
CHANNELS = (415, 445, 480, 515, 555, 590, 630, 680, "clear", "nir")
BROADBAND = -1
ILLUMINATION_WATCHDOG_S = 10.0
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


i2c = busio.I2C(board.GP5, board.GP4, frequency=400_000)
sensor = AS7341(i2c)
sensor.atime = 29
sensor.astep = 599
sensor.led_current = 20

leds = {}
for wavelength, pin in LED_PINS.items():
    output = digitalio.DigitalInOut(pin)
    output.direction = digitalio.Direction.OUTPUT
    output.value = False
    leds[wavelength] = output

data_port = usb_cdc.data
buffer = bytearray()
illumination_active = False
last_command_time = time.monotonic()


def all_off():
    global illumination_active
    sensor.led = False
    for output in leds.values():
        output.value = False
    illumination_active = False


def set_illumination(wavelength_nm):
    global illumination_active
    wavelength_nm = int(wavelength_nm)
    if wavelength_nm not in (0, BROADBAND) and wavelength_nm not in leds:
        raise ValueError("unsupported illumination wavelength")
    all_off()
    if wavelength_nm == BROADBAND:
        sensor.led = True
        illumination_active = True
    elif wavelength_nm in leds:
        leds[wavelength_nm].value = True
        illumination_active = True


def read_once():
    visible = tuple(float(value) for value in sensor.all_channels)
    auxiliary = (float(sensor.channel_clear), float(sensor.channel_nir))
    return visible + auxiliary


def read_average(samples, interval_s):
    samples = max(1, min(int(samples), 100))
    interval_s = max(0.0, min(float(interval_s), 2.0))
    sums = [0.0] * len(CHANNELS)
    for sample_index in range(samples):
        values = read_once()
        for index, value in enumerate(values):
            sums[index] += value
        if sample_index + 1 < samples:
            time.sleep(interval_s)
    return [value / samples for value in sums]


def handle(request):
    command = request.get("command")
    if command == "hello":
        return {
            "ok": True,
            "device": DEVICE,
            "channels": list(CHANNELS),
            "illuminations_nm": [BROADBAND] + list(LED_PINS),
            "firmware_version": "0.3.0",
        }
    if command == "set_illumination":
        set_illumination(request.get("wavelength_nm", 0))
        return {"ok": True}
    if command == "read":
        return {
            "ok": True,
            "signals": read_average(request.get("samples", 5), request.get("interval_s", 0.05)),
        }
    if command == "close":
        all_off()
        return {"ok": True}
    raise ValueError("unknown command")


def send(response):
    data_port.write((json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8"))


all_off()
while True:
    if data_port.in_waiting:
        incoming = data_port.read(data_port.in_waiting)
        if incoming:
            buffer.extend(incoming)
        while b"\n" in buffer:
            line, _, remainder = buffer.partition(b"\n")
            buffer = bytearray(remainder)
            try:
                request = json.loads(line.decode("utf-8"))
                send(handle(request))
                last_command_time = time.monotonic()
            except Exception as error:  # CircuitPython sends a concise device error to the GUI.
                all_off()
                send({"ok": False, "error": str(error)})
    if illumination_active and time.monotonic() - last_command_time > ILLUMINATION_WATCHDOG_S:
        all_off()
    time.sleep(0.005)
