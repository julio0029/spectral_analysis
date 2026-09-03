// Custom interconnect PCBs for the portable AS7341 spectrophotometer case,
// built with NopSCADlib (https://github.com/nophead/NopSCADlib), the
// library recommended at https://openscad.org/libraries.html for exactly
// this kind of custom-PCB modelling.
//
// Two boards:
//   1. SPECTRO_CARRIER  - main interconnect board: Pico 2 socket footprint,
//      AS7341 I2C header, battery JST connector, and a ribbon header out to
//      the LED ring board. Sits in place of point-to-point wiring between
//      the case's existing components.
//   2. SPECTRO_LED_RING - small board that carries the 8 illumination LEDs
//      in a donut, sized and positioned to exactly match the LED ring holes
//      already cut into spectrophotometer_case.scad's lid (led_ring_r=6.2,
//      led_count=8, probe_boss_od=18.6, fiber_counterbore_d=4.0), with a
//      centre bore for the fibre-optic light pipe.
//
// Set `board` to "carrier" or "led_ring" to render just that PCB, or
// "both" for a side-by-side review layout (not an installed assembly).
//
// NOTE ON SCOPE: these boards are dimensionally consistent with the
// existing case geometry (component positions, LED ring, fibre bore) but
// the case's base tray standoffs have NOT yet been revised to mount the
// carrier board directly -- see the comment above base_mounting_holes()
// below for the follow-up needed before these are drop-in replacements
// for the loose wiring assumed in the current base.scad layout.

include <NopSCADlib/core.scad>
include <NopSCADlib/vitamins/pin_headers.scad>
include <NopSCADlib/vitamins/leds.scad>
use <NopSCADlib/vitamins/pcb.scad>

board = "both"; // ["carrier", "led_ring", "both"]

// ---------------------------------------------------------------------
// Geometry pulled from spectrophotometer_case.scad so the boards match
// the printed case exactly. Duplicated here (NopSCADlib PCB definitions
// are plain data, not something to `include` a second unrelated file
// into) -- keep these in sync if the case file's constants change.
// ---------------------------------------------------------------------
case_led_count           = 8;
case_led_ring_r          = 6.2;    // mm, centre to LED hole centre
case_probe_boss_od       = 18.6;   // mm
case_fiber_counterbore_d = 4.0;    // mm, centre bore through the ring board

// ---------------------------------------------------------------------
// Main carrier PCB.
//
// Pico 2 connection: modelled as two 20-way, 0.1" (2.54mm) pitch female
// sockets, row pitch 17.78mm (0.7in), matching the Pico family's
// documented castellated-pad footprint when fitted with pin headers.
// This is the standard way to make a Pico both socketed (removable) and
// hand-solderable without SMD reflow.
// ---------------------------------------------------------------------
carrier_size   = [70, 22, 1.6];
pico_row_pitch = 17.78;
pico_pins      = 20;

SPECTRO_CARRIER = pcb(
    "SPECTRO_CARRIER", "Spectrophotometer interconnect carrier board",
    carrier_size,
    corner_r = 2, hole_d = 2.5, land_d = 5, colour = "green", parts_on_bom = true,
    holes = [
        // 4 corner mounting holes, 3mm inset from each edge.
        [3, 3], [3, carrier_size.y - 3],
        [carrier_size.x - 3, 3], [carrier_size.x - 3, carrier_size.y - 3],
    ],
    components = [
        // Pico 2 socket, two 20-way 0.1" female headers.
        [carrier_size.x / 2, carrier_size.y / 2 + pico_row_pitch / 2, 0, "2p54socket", pico_pins, 1],
        [carrier_size.x / 2, carrier_size.y / 2 - pico_row_pitch / 2, 0, "2p54socket", pico_pins, 1],

        // AS7341 I2C header: 3V3, GND, SDA, SCL (4-way JST-PH, matches the
        // DFRobot/Adafruit breakout's usual 4-pin Qwiic/STEMMA-style pitch).
        [6, 3, 90, "jst_ph", 4],

        // Battery connector, 2-way JST-PH (matches a standard 301230 LiPo
        // cable terminated in JST-PH, as used by most small LiPo cells).
        [6, carrier_size.y - 3, -90, "jst_ph", 2],

        // Ribbon-out header to the LED ring board: 8 channel lines plus a
        // shared return, one 9-way 0.1" pin header.
        [carrier_size.x - 6, carrier_size.y / 2, 0, "2p54header", 9, 1],
    ],
    accessories = [
        ": Raspberry Pi Pico 2",
        ": DFRobot Fermion AS7341 breakout",
        ": 301230 LiPo battery, ~130mAh, JST-PH terminated",
        ": 9-way 0.1in ribbon cable to LED ring board",
    ]
);

// ---------------------------------------------------------------------
// LED ring board.
//
// A small round board that mounts directly under the lid's probe boss.
// 8x LED3mm footprints at the same radius as the LID's LED through-holes,
// so each LED pokes straight up into its hole. Centre bore matches the
// fibre counterbore so the fibre passes straight through the board to
// the AS7341 below. Two small registration holes for alignment pins.
// ---------------------------------------------------------------------
ring_pcb_od = case_probe_boss_od - 1;   // slightly smaller than the boss ID

SPECTRO_LED_RING = pcb(
    "SPECTRO_LED_RING", "LED ring board (donut illuminator)",
    [ring_pcb_od, ring_pcb_od, 1.0],
    corner_r = ring_pcb_od / 2 - 0.01,   // rounded_square requires r < size/2 strictly; this is effectively a circle
    hole_d = 1.2, land_d = 2.4, colour = "black", parts_on_bom = true,
    holes = [
        // Two registration pins, opposite each other, outside the LED ring
        // radius but inside the board edge.
        [ring_pcb_od / 2 + (ring_pcb_od / 2 - 2) * cos(45), ring_pcb_od / 2 + (ring_pcb_od / 2 - 2) * sin(45)],
        [ring_pcb_od / 2 - (ring_pcb_od / 2 - 2) * cos(45), ring_pcb_od / 2 - (ring_pcb_od / 2 - 2) * sin(45)],
    ],
    components = [
        for (i = [0 : case_led_count - 1])
            let(a = i * 360 / case_led_count)
                [ring_pcb_od / 2 + case_led_ring_r * cos(a),
                 ring_pcb_od / 2 + case_led_ring_r * sin(a),
                 a, "led", LED3mm, 2, "white"],
    ],
    accessories = [
        str(": ", case_led_count, "x 3mm THT LED"),
    ]
);

// ---------------------------------------------------------------------
// Board renderers
// ---------------------------------------------------------------------
module carrier_board() {
    pcb(SPECTRO_CARRIER);
}

module led_ring_board() {
    // The pcb() constructor has no built-in support for a large centre
    // bore (its `holes` list is for small, uniform-diameter mounting
    // holes only), so the fibre bore is cut here as a genuine difference
    // against the fully-rendered library board -- silkscreen, LED
    // footprints, copper and all.
    difference() {
        translate([-ring_pcb_od / 2, -ring_pcb_od / 2, 0])
            pcb(SPECTRO_LED_RING);
        translate([0, 0, -1])
            cylinder(h = 10, d = case_fiber_counterbore_d);
    }
}

if (board == "carrier") {
    carrier_board();
} else if (board == "led_ring") {
    led_ring_board();
} else {
    carrier_board();
    translate([carrier_size.x + 15, carrier_size.y / 2, 0])
        led_ring_board();
}
