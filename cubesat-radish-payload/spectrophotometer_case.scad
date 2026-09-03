// Portable AS7341 spectrophotometer case.
// Units: millimetres. Parametric OpenSCAD model for a benchtop/handheld
// enclosure (distinct from the flight cubesat_radish_payload.scad in this
// same folder, which targets a Pico 2 in an 80x40x40 mm CubeSat envelope).
//
// This model secures a Raspberry Pi Zero (W), an AS7341 breakout, and a
// ring ("donut") of illumination LEDs whose centre carries a fibre-optic
// light pipe down to the AS7341's optical aperture.
//
// Set `part` to one of:
//   "base"       - lower tray: Pi Zero standoffs, AS7341 shelf, wiring bay
//   "lid"        - upper shell with the LED-ring / fibre probe boss
//   "fiber_guide"- printable light-tight tube, lid boss -> sensor window
//   "assembly"   - base + lid + fibre guide + component keepouts, for review only
// Export "base", "lid" and "fiber_guide" separately as STL files for
// printing. The assembly view is for fit-checking only.

part = "assembly";
$fn = 64;

// ---------------------------------------------------------------------
// Global envelope and wall thickness
// ---------------------------------------------------------------------
outer      = [118, 65, 34];   // external L x W x H
wall       = 2.4;
lid_height = 12;
base_height = outer[2] - lid_height;

// ---------------------------------------------------------------------
// Raspberry Pi Zero (W) mechanical reference
// Board 65 x 30 mm, 4x M2.5 mounting holes 2.75 mm dia,
// hole centres 3.5 mm in from each edge (58 x 23 mm hole pattern).
// ---------------------------------------------------------------------
pi_size       = [65, 30, 1.4];
pi_hole_d     = 2.75;
pi_hole_inset = 3.5;
pi_origin     = [8, 17];        // base-tray XY offset of the Pi's corner
pi_standoff_h = 6;              // clearance under board for solder joints
pi_standoff_d = 6;
// microSD card protrudes ~2 mm below the PCB near the connector edge;
// relieved in the base floor, clear of every standoff (verified below).
sd_relief     = [12, 10];
sd_origin     = [pi_origin[0] + pi_size[0] / 2 - sd_relief[0] / 2, pi_origin[1] - 1];

// ---------------------------------------------------------------------
// AS7341 breakout shelf (generic footprint; fits DFRobot Fermion
// 18x14 mm and the slightly larger Adafruit STEMMA breakout with margin)
// Offset far enough past the Pi Zero's end (+14 mm gap) to avoid any
// footprint collision; a wiring bay fills that gap.
// ---------------------------------------------------------------------
as7341_size   = [21, 18, 1.6];
as7341_origin = [pi_origin[0] + pi_size[0] + 14, 23];
as7341_shelf_h = 9;             // board sits above the base floor
sensor_window_d = 6;            // optical aperture above the sensor die
sensor_center = [
    as7341_origin[0] + as7341_size[0] / 2,
    as7341_origin[1] + as7341_size[1] / 2
];

// ---------------------------------------------------------------------
// LED ring ("donut") and central fibre-optic channel
// ---------------------------------------------------------------------
led_count      = 8;      // number of illumination LEDs around the ring
led_d          = 5.2;    // through-hole for a 5 mm THT LED
led_ring_r     = 13;     // ring radius, centre to LED hole centre
probe_boss_od  = 34;     // outer diameter of the raised probe boss on the lid
probe_boss_h   = 10;     // height the boss stands proud of the lid top
fiber_bore_d   = 3.4;    // clearance for a jacketed optical fibre / ferrule
fiber_counterbore_d = 6.5;  // wider mouth to glue/seat a fibre ferrule
fiber_counterbore_h = 4;
probe_center = sensor_center;   // probe boss sits directly above the sensor

// ---------------------------------------------------------------------
// Sanity checks
// ---------------------------------------------------------------------
assert(pi_origin[0] + pi_size[0] + pi_hole_inset <= outer[0] - wall);
assert(pi_origin[1] + pi_size[1] + pi_hole_inset <= outer[1] - wall);
assert(as7341_origin[0] + as7341_size[0] <= outer[0] - wall);
assert(as7341_origin[1] + as7341_size[1] <= outer[1] - wall);
assert(led_ring_r * 2 + led_d <= probe_boss_od);
assert(probe_boss_od <= outer[1] - 2 * wall);

// ---------------------------------------------------------------------
// Base tray
// ---------------------------------------------------------------------
module pi_zero_standoffs() {
    for (dx = [pi_hole_inset, pi_size[0] - pi_hole_inset])
        for (dy = [pi_hole_inset, pi_size[1] - pi_hole_inset])
            translate([pi_origin[0] + dx, pi_origin[1] + dy, wall])
            difference() {
                cylinder(h = pi_standoff_h, d = pi_standoff_d);
                translate([0, 0, -0.1])
                    cylinder(h = pi_standoff_h + 0.2, d = pi_hole_d);
            }
}

module as7341_shelf() {
    translate([as7341_origin[0], as7341_origin[1], wall])
    difference() {
        cube([as7341_size[0], as7341_size[1], as7341_shelf_h]);
        // Central pocket so only a support lip touches the board underside.
        translate([2, 2, 2])
            cube([as7341_size[0] - 4, as7341_size[1] - 4, as7341_shelf_h]);
    }
    // Light-tight baffle socket that the fibre_guide tube's foot seats into,
    // directly over the sensor's optical aperture.
    translate([sensor_center[0], sensor_center[1], wall + as7341_shelf_h + as7341_size[2]])
        difference() {
            cylinder(h = 4, d = sensor_window_d + 4);
            translate([0, 0, -0.1]) cylinder(h = 4.2, d = sensor_window_d);
        }
}

module wiring_bay_standoffs() {
    // Small battery / wiring shelf between the Pi and the AS7341 board.
    bay_x = pi_origin[0] + pi_size[0] + 6;
    bay_w = as7341_origin[0] - bay_x - 6;
    translate([bay_x, 8, wall])
        cube([max(bay_w, 1), outer[1] - 2 * wall - 16, 3]);
}

module base_shell() {
    difference() {
        cube([outer[0], outer[1], base_height]);
        translate([wall, wall, wall])
            cube([outer[0] - 2 * wall, outer[1] - 2 * wall, base_height]);
        // microSD relief so the card is not resting on plastic.
        translate([sd_origin[0], sd_origin[1], -0.1])
            cube([sd_relief[0], sd_relief[1], wall + 0.2]);
        // USB power / data access on the short end nearest the Pi's ports.
        translate([-0.1, pi_origin[1] + 4, wall + 1])
            cube([wall + 0.2, 18, 8]);
    }
}

module base() {
    base_shell();
    pi_zero_standoffs();
    as7341_shelf();
    wiring_bay_standoffs();
}

// ---------------------------------------------------------------------
// Lid with LED-ring probe boss
// ---------------------------------------------------------------------
module led_ring_holes(h) {
    for (i = [0 : led_count - 1]) {
        a = i * 360 / led_count;
        translate([probe_center[0] + led_ring_r * cos(a),
                    probe_center[1] + led_ring_r * sin(a), -0.1])
            cylinder(h = h + 0.2, d = led_d);
    }
}

module lid_shell() {
    difference() {
        cube([outer[0], outer[1], wall]);
        // (Solid lid skin; the probe boss and fibre bore are cut below.)
    }
    // Downturned skirt that overlaps the base wall for a dust/light seal.
    difference() {
        translate([0, 0, -3])
            cube([outer[0], outer[1], 3]);
        translate([wall, wall, -3.1])
            cube([outer[0] - 2 * wall, outer[1] - 2 * wall, 3.2]);
    }
}

module probe_boss() {
    translate([probe_center[0], probe_center[1], wall])
        cylinder(h = probe_boss_h, d = probe_boss_od);
}

module lid() {
    difference() {
        union() {
            lid_shell();
            probe_boss();
        }
        // Central fibre channel: wide counterbore at the very top to seat
        // a ferrule/connector, narrowing through the lid and boss.
        translate([probe_center[0], probe_center[1], wall + probe_boss_h - fiber_counterbore_h])
            cylinder(h = fiber_counterbore_h + 0.1, d = fiber_counterbore_d);
        translate([probe_center[0], probe_center[1], -0.1])
            cylinder(h = wall + probe_boss_h + 0.2, d = fiber_bore_d);
        // Donut of LED through-holes around the fibre channel.
        led_ring_holes(wall + probe_boss_h);
    }
}

// ---------------------------------------------------------------------
// Printable fibre-optic guide tube: registers into the lid boss above
// and the AS7341 baffle socket below, keeping the light path enclosed
// and light-tight along its length. Print in black filament, or paint
// the bore matte black before assembly.
// ---------------------------------------------------------------------
module fiber_guide() {
    gap = base_height - (wall + as7341_shelf_h + as7341_size[2] + 4)
        + probe_boss_h;   // interior span the tube must bridge
    tube_len = max(gap, 10);
    difference() {
        union() {
            cylinder(h = tube_len, d = fiber_counterbore_d);
            // Register flange that seats in the AS7341 baffle socket.
            translate([0, 0, tube_len - 2])
                cylinder(h = 2, d = sensor_window_d + 3.6);
        }
        translate([0, 0, -0.1])
            cylinder(h = tube_len + 0.2, d = fiber_bore_d);
    }
}

// ---------------------------------------------------------------------
// Assembly / keepout preview (not for printing)
// ---------------------------------------------------------------------
module keepouts() {
    // Raspberry Pi Zero (W) footprint.
    color([0.10, 0.45, 0.25, 0.85])
        translate([pi_origin[0], pi_origin[1], wall + pi_standoff_h])
            cube(pi_size);
    // AS7341 breakout.
    color([0.25, 0.25, 0.75, 0.85])
        translate([as7341_origin[0], as7341_origin[1], wall + as7341_shelf_h])
            cube(as7341_size);
    // Illumination LEDs, shown seated in the lid ring.
    for (i = [0 : led_count - 1]) {
        a = i * 360 / led_count;
        color([0.95, 0.85, 0.20, 0.9])
            translate([probe_center[0] + led_ring_r * cos(a),
                        probe_center[1] + led_ring_r * sin(a),
                        base_height + wall + probe_boss_h - 5])
                cylinder(h = 5, d = led_d - 0.4);
    }
    // Fibre-optic light pipe running from the probe tip to the sensor.
    color([0.85, 0.15, 0.15, 0.6])
        translate([probe_center[0], probe_center[1], wall + as7341_shelf_h + as7341_size[2]])
            cylinder(h = base_height + wall + probe_boss_h
                          - (wall + as7341_shelf_h + as7341_size[2]),
                       d = fiber_bore_d);
}

if (part == "base") {
    base();
} else if (part == "lid") {
    lid();
} else if (part == "fiber_guide") {
    fiber_guide();
} else {
    color([0.80, 0.80, 0.82, 0.35]) base();
    color([0.85, 0.85, 0.88, 0.35])
        translate([0, 0, base_height]) lid();
    keepouts();
}
