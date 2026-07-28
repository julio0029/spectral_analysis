# CubeSat radish germination and mitochondrial spectroscopy payload

## Purpose and boundary

This protocol adapts the Pico/AS7341 instrument for one radish seed germinated
after commanded rehydration in a sealed CubeSat biological cassette. It records
mitochondrial cytochrome indices, chlorophyll emergence, red-edge development,
CO₂, temperature and relative humidity within a maximum external envelope of
80 × 40 × 40 mm.

The design is a laboratory engineering exemplar. Raspberry Pi Pico, DFRobot and
Adafruit development boards are not radiation-qualified spacecraft components.
The cassette, materials, electronics and software require launch-provider
review, biocontainment review, vibration, shock, EMC, radiation, leak,
thermal-vacuum and end-to-end mission qualification before flight.

## Fixed payload envelope and internal allocation

| Region | Maximum allocation | Function |
|---|---:|---|
| Complete payload | 80 × 40 × 40 mm | External keep-in envelope |
| Growth cassette | 76 × 36 × 24 mm internal | Seed, cotton, gas headspace and early sprout |
| Seed/cotton tray | 28 × 24 × 8 mm external | Captive reservoir interface, wick and one seed |
| Dry side channel | 54 × 5 × 24 mm | Vertically mounted Pico 2 |
| Upper deck | 80 × 40 × 10 mm usable | Optical PCB, drivers, pump/valve and fan |
| SCD-40 bay | 7.7 × 22.8 × 25.5 mm | Vertical breakout with protected gas opening |

The accompanying OpenSCAD model is `cubesat_radish_payload.scad`. The biological cassette is below a gasketed
optical deck. The Pico is vertical in an isolated side channel. The SCD-40 is
vertical at the end of the chamber, with only its sensing opening coupled to
the headspace through a hydrophobic, gas-permeable membrane.

One well is recommended for the first flight unit. Two biological wells would
require a second optical head or an optical multiplexer, plus independent water
delivery, and should not be added until the single-well instrument is validated.

## Required components

| Component | Flight-prototype requirement | Approximate keep-out |
|---|---|---:|
| Controller | Raspberry Pi Pico 2 without headers | 51 × 21 mm board |
| Spectral detector | DFRobot Fermion AS7341, address 0x39 | 18 × 14 mm |
| Environmental detector | Adafruit SCD-40, address 0x62 | 25.5 × 22.8 × 7.7 mm |
| Optical PCB | Custom black 4-layer PCB with sensor aperture, LED ring and drivers | 28 × 28 × 3 mm |
| External LEDs | 365, 450, 535, 550, 565, 575, 605, 630, 660, 700, 730 and 940 nm | 12 SMD packages |
| LED drive | Twelve low-side constant-current channels, disabled by pull-downs | On optical PCB |
| White source | AS7341-board white LED or calibrated broadband SMD source | On optical PCB |
| Rehydration | 3 mL collapsible reservoir, captive wick and normally-closed micro-pump or latching valve | ≤30 × 22 × 6 mm reservoir, ≤20 × 10 × 7 mm actuator |
| Mixing | 15 × 15 × 4 mm low-vibration fan or microblower | Upper deck |
| Chamber | Opaque, low-outgassing lower body and gasketed clear/opaque deck | 80 × 40 × 28 mm |
| Optical window | Thin fused silica or space-compatible clear polymer | 8–12 mm aperture |
| Biological insert | Autoclavable removable tray, cotton or cellulose wick and one radish seed | 28 × 24 × 8 mm |
| Electrical protection | Load switches, current limiting, flyback suppression and spacecraft-side fuse | Carrier PCB |
| Harness | UART, regulated power, ground and optional reset/enable | Mission-specific |

The Pico 2 board is 21 × 51 mm. The DFRobot AS7341 board is 18 × 14 mm and
specifies 5–85% RH, so it should remain behind the optical window rather than
inside the saturated wet chamber. The Adafruit SCD-40 breakout is
25.5 × 22.8 × 7.7 mm. Its bare sensing element is 10.1 × 10.1 × 6.5 mm, but the
larger breakout dimensions are used in the CAD keep-out.

## Wavelength plan

| LED | Payload observation | AS7341 response used |
|---:|---|---|
| 365 nm | NAD(P)H/autofluorescence excitation | 445–555 nm emission bands |
| 450 nm | FAD/FMN and chlorophyll excitation | 555–680 nm, with 680 nm chlorophyll feature |
| 535 nm | Cytochrome c reference | F5, 555 nm |
| 550 nm | Cytochrome c α-band target | F5, 555 nm |
| 565 nm | Cytochrome b α-band target | Mean of F5/F6 |
| 575 nm | Cytochrome b reference | Mean of F5/F6 |
| 605 nm | Cytochrome aa₃ target | F7, 630 nm |
| 630 nm | Cytochrome aa₃ reference | F7, 630 nm |
| 660 nm | Chlorophyll red absorption and legacy oxygenation observation | F8, 680 nm |
| 700 nm | Red-edge lower observation | Clear channel, calibrated source |
| 730 nm | Far-red red-edge reference | Clear channel, calibrated source |
| 940 nm | Water/scattering reference | NIR and clear channels |

The 730 nm source is added specifically for the plant payload. The AS7341 does
not have a narrow 730 nm detector; therefore 700/730 is a calibrated,
device-specific reflectance index measured through the clear channel. It is not
a full red-edge spectrum.

The 450-to-680 nm observation provides a chlorophyll fluorescence indicator.
The hardware cannot calculate validated Fv/Fm or photosystem-II quantum yield
without controlled dark adaptation, actinic light, saturating pulses and
appropriate fluorescence timing. Report it as relative chlorophyll
fluorescence until compared with a PAM fluorometer.

## Pico connections

| Function | Pico connection |
|---|---|
| Spacecraft UART TX/RX | GP0/GP1 |
| AS7341 and SCD-40 SDA/SCL | GP4/GP5 |
| 365–700 nm LED enables | GP6–GP15, in ascending wavelength order |
| 730 nm LED enable | GP16 |
| 940 nm LED enable | GP17 |
| Rehydration driver enable | GP18 |
| Mixing fan driver enable | GP19 |

All GPIO pins drive logic inputs or MOSFET gates only. No LED, pump, valve or
fan load is powered from a GPIO. Provide a hardware default-off state and a
spacecraft-side current limit. Keep wet-chamber ground paths and switching
currents away from the AS7341 and SCD-40 supplies.

## Rehydration cassette

1. Use one surface-sterilised radish seed from a characterised lot.
2. Place the seed in a shallow recess between two pre-weighed sterile cellulose
   layers. Add a porous restraint so the seed and cotton cannot float.
3. Connect the cotton to the reservoir through three captive wick ports. Water
   must remain inside the reservoir, tubing and wick at every mission phase.
4. Load up to 3 mL sterile water, but calibrate the three-second actuator pulse
   to deliver only the ground-validated volume, initially approximately
   0.8–1.5 mL for one seed and pad.
5. Include a physical pinch point or normally-closed valve so launch vibration
   and pressure changes cannot wet the seed prematurely.
6. Verify rehydration delivery in every expected payload orientation and during
   clinostat testing. Microgravity fluid behaviour cannot be inferred from a
   normal-gravity drip test.

Flight firmware requires an `arm_rehydration` command followed within 60 seconds
by `rehydrate`. After a completed pulse, a nonvolatile latch prevents a second
release after reset. Clear this latch only during controlled ground servicing.

## Gas, humidity and SCD-40 constraints

The SCD-40 performs photoacoustic CO₂ measurement and also reports temperature
and relative humidity. Its specified CO₂ accuracy applies over 400–2000 ppm,
although its digital output extends higher. A small sealed chamber can exceed
2000 ppm during seed respiration, so high readings are scientifically useful
but must be marked outside the specified accuracy range. Consider SCD-41 if the
validated ground profile repeatedly exceeds 2000 ppm.

Disable automatic self-calibration. That algorithm assumes periodic exposure to
fresh air near 400 ppm, which a sealed biological cassette does not provide.
Calibrate the assembled chamber with certified CO₂ mixtures before sealing.
Record chamber pressure during ground tests and define a fixed or telemetered
pressure compensation strategy for flight.

The SCD-40 operating humidity limit is 0–95% RH, non-condensing. Keep liquid and
condensate away from the sensing opening with a hydrophobic membrane, a drip
shield and controlled thermal gradients. Use the mixing fan briefly before a
gas reading; continuous airflow is unnecessary and can alter seed drying and
sensor temperature.

CO₂ is not a substitute for oxygen. For mechanistic interpretation of
mitochondrial respiration, a future flight revision should add a miniature
optical oxygen spot or another independently validated O₂ measurement.

## Assembly procedure

1. Print the CAD exemplar for fit checks. Manufacture the flight cassette from
   a launch-approved, low-outgassing material after materials review.
2. Assemble the Pico carrier and optical PCB. Fit current limiting, pull-downs,
   load switching and flyback suppression before installing LEDs or actuators.
3. Measure the centre wavelength, FWHM and optical output of every installed
   LED. Store calibration identifiers in the payload configuration.
4. Mount the AS7341 behind the optical window, centred over the seed/cotyledon
   field. Add a black source-detector baffle and suppress internal reflections.
5. Mount the Pico vertically inside the dry side channel. Do not use protruding
   2.54 mm headers in the flight stack.
6. Mount the SCD-40 vertically with a protected gas path to the chamber. Keep
   its regulator and thermal sources outside the biological volume.
7. Install the reservoir, normally-closed actuator, captive tube and wick.
   Perform a dry electrical actuation before adding water.
8. Install the fan so it mixes the headspace without blowing directly onto the
   seed or cotton.
9. Fit the biological tray, optical deck gasket and lid. Perform helium or
   pressure-decay leak testing at the project-defined acceptance level.
10. Measure total mass, centre of mass, peak current, average energy and thermal
    dissipation in the exact spacecraft mounting orientation.

## Software installation and modes

For benchtop development, copy `code.py` to the Pico. Install `adafruit_as7341.mpy`,
`adafruit_scd4x.mpy`,
`adafruit_bus_device/` and `adafruit_register/` in `CIRCUITPY/lib`. The desktop
GUI controls the payload over the CircuitPython USB data port.

For spacecraft integration, copy `flight_code.py` to `CIRCUITPY/code.py`.
Connect UART0 to the spacecraft OBC. The flight program:

- emits environmental telemetry every 60 seconds;
- performs a complete spectral cycle every 15 minutes;
- accepts a commanded immediate spectral cycle;
- requires a two-command rehydration interlock;
- latches successful rehydration in nonvolatile memory;
- switches sources and actuators off after errors.

The spacecraft OBC must add mission UTC, store the JSON telemetry redundantly,
enforce payload power limits and transmit health/status data independently of
the Pico.

## Baseline and mission sequence

1. Sterilise and integrate the dry biological cassette. Record seed lot, seed
   mass, cotton mass, water volume, assembly mass and calibration identifiers.
2. At flight temperature and pressure, acquire at least ten dry baseline
   spectral cycles and at least 30 minutes of SCD-40 data.
3. After orbit commissioning, send `arm_rehydration`, confirm acknowledgement,
   then send `rehydrate` within 60 seconds.
4. Acquire a spectral cycle immediately before and after rehydration, then every
   15 minutes for 72 hours. Record CO₂, temperature and humidity every 60 seconds.
5. Each spectral cycle records:

   dark → white → 365 → 450 → 535 → 550 → 565 → 575 → 605 → 630 → 660 →
   700 → 730 → 940 nm.

6. Include matched 1 g ground controls, an unhydrated flight-control cassette
   where feasible, and clinostat controls. Run the same optical and environmental
   schedule for all controls.

## Analysis outputs

Retain all raw detector counts. The software produces these longitudinal,
device-specific observations:

- cytochrome c index, 550 minus 535 nm;
- cytochrome b index, 565 minus 575 nm;
- cytochrome aa₃ index, 605 minus 630 nm;
- 450 nm-excited, 680 nm chlorophyll fluorescence log change;
- broadband 680-versus-555 nm red-absorption change;
- 700-versus-730 nm red-edge change;
- CO₂, temperature and relative humidity.

Do not interpret these as absolute cytochrome concentration, chlorophyll
concentration, respiration rate, germination percentage or photosynthetic
efficiency until calibrated against chemical redox endpoints, a reference
spectrophotometer, PAM fluorometry, extracted chlorophyll, respirometry and
direct germination imaging or post-flight scoring.

## Qualification gates

Do not progress to flight integration until the following pass:

- 100% successful commanded rehydration in all tested orientations;
- no free droplets after vibration, clinostat or thermal cycling;
- stable chamber pressure and no cross-leak into spacecraft volume;
- dark and reference drift within predetermined limits over 72 hours;
- no AS7341 or SCD-40 condensation events;
- LED wavelength and output stability after environmental testing;
- successful brownout recovery without repeated rehydration;
- UART telemetry recovery after OBC reset;
- radiation and single-event-upset risk accepted by the mission;
- biological containment and launch-safety approval.

## Component and scientific references

- [Raspberry Pi Pico specifications](https://www.raspberrypi.com/products/raspberry-pi-pico/)
- [DFRobot Fermion AS7341 specifications](https://wiki.dfrobot.com/sen0365/)
- [Adafruit SCD-40 product dimensions and specifications](https://www.adafruit.com/product/5187)
- [Sensirion SCD40 specifications](https://sensirion.com/products/catalog/SCD40)
- [Adafruit SCD-40 CircuitPython guide](https://learn.adafruit.com/adafruit-scd-40-and-scd-41/python-circuitpython)
- [Adafruit CircuitPython SCD4X API](https://docs.circuitpython.org/projects/scd4x/en/latest/api.html)
- Horler, D. N. H., Dockray, M. and Barber, J. (1983). The red edge of plant
  leaf reflectance. *International Journal of Remote Sensing*, 4, 273–288.
- Baker, N. R. (2008). Chlorophyll fluorescence: a probe of photosynthesis in
  vivo. *Annual Review of Plant Biology*, 59, 89–113.
