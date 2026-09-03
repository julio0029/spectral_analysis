// Portable AS7341 spectrophotometer case (compact / Pico 2 revision).
// Units: millimetres. Parametric OpenSCAD model for a benchtop/handheld
// enclosure (distinct from the flight cubesat_radish_payload.scad in this
// same folder, which targets a Pico 2 in an 80x40x40 mm CubeSat envelope).
//
// Rev 2 (2026-09-04): minimised footprint. Switched the controller keepout
// from a Raspberry Pi Zero (65x30mm) to a Raspberry Pi Pico 2 (51x21mm) as
// requested, corrected the LED holes from 5mm to 3mm THT, tightened the LED
// ring/probe boss to match, and tucked a small LiPo cell (301230, ~130mAh)
// into the floor pocket directly under the Pico rather than giving it its
// own footprint. External envelope shrank from 118x65x34mm to ~80x27x24mm.
//
// This model secures a Raspberry Pi Pico 2, an AS7341 breakout, a small
// LiPo battery, and a ring ("donut") of illumination LEDs whose centre
// carries a fibre-optic light pipe down to the AS7341's optical aperture.
//
// Set `part` to one of:
//   "base"       - lower tray: Pico standoffs, battery pocket, AS7341 shelf
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
wall       = 1.8;
margin     = 1.0;   // clearance from a component footprint to the inner wall
gap        = 2.5;   // clearance between adjacent component footprints
lid_height = 9;

// ---------------------------------------------------------------------
// Raspberry Pi Pico 2 mechanical reference (51 x 21 mm board). Corner
// standoffs are a packaging approximation (this is not an exact drill
// template), inset 3 mm from each edge, matching the approach used for
// the Pico keepout in cubesat_radish_payload.scad.
// ---------------------------------------------------------------------
pico_size       = [51, 21, 1.0];
pico_hole_d     = 2.2;
pico_hole_inset = 3.0;
pico_origin     = [wall + margin, wall + margin];
pico_standoff_h = 5;             // clearance under board for solder joints
pico_standoff_d = 4.5;

// ---------------------------------------------------------------------
// Small LiPo battery (301230 form factor, ~130 mAh: 30 x 12 x 3.5 mm).
// Nested in a floor pocket directly under the Pico footprint, inside its
// standoff ring, so it adds zero extra footprint. Confirmed clear of all
// four Pico standoffs with margin.
// ---------------------------------------------------------------------
batt_size   = [30, 12, 3.5];
batt_origin = [
    pico_origin[0] + (pico_size[0] - batt_size[0]) / 2,
    pico_origin[1] + (pico_size[1] - batt_size[1]) / 2
];

// ---------------------------------------------------------------------
// AS7341 breakout shelf (generic footprint; fits DFRobot Fermion
// 18x14 mm and the slightly larger Adafruit STEMMA breakout with margin).
// Placed alongside the Pico with a clearance gap between footprints.
// ---------------------------------------------------------------------
as7341_size   = [21, 18, 1.6];
as7341_origin = [pico_origin[0] + pico_size[0] + gap, wall + margin];
as7341_shelf_h = 7;              // board sits above the base floor
sensor_window_d = 5;             // optical aperture above the sensor die
sensor_center = [
    as7341_origin[0] + as7341_size[0] / 2,
    as7341_origin[1] + as7341_size[1] / 2
];

// ---------------------------------------------------------------------
// LED ring ("donut") and central fibre-optic channel.
// LEDs are 3 mm THT (led_d includes printer clearance over the nominal
// 3.0 mm lead diameter). Ring radius and probe boss sized to the minimum
// that (a) keeps adjacent LED holes from touching and (b) clears the
// central fibre counterbore.
// ---------------------------------------------------------------------
led_count      = 8;      // number of illumination LEDs around the ring
led_d          = 3.2;    // through-hole for a 3 mm THT LED (+0.2 mm clearance)
led_ring_r     = 6.2;    // ring radius, centre to LED hole centre
probe_boss_od  = 18.6;   // outer diameter of the raised probe boss on the lid
probe_boss_h   = 6;      // height the boss stands proud of the lid top
fiber_bore_d   = 2.2;    // clearance for a jacketed optical fibre / ferrule
fiber_counterbore_d = 4.0;  // wider mouth to glue/seat a fibre ferrule
fiber_counterbore_h = 3;
probe_center = sensor_center;   // probe boss sits directly above the sensor

// ---------------------------------------------------------------------
// Global envelope, derived from the component layout above.
// ---------------------------------------------------------------------
outer_xy_min = [
    max(pico_origin[0] + pico_size[0], as7341_origin[0] + as7341_size[0]) + wall + margin,
    max(pico_size[1], as7341_size[1], probe_boss_od + 2) + 2 * (wall + margin)
];
outer = [outer_xy_min[0], outer_xy_min[1], 24];
base_height = outer[2] - lid_height;

// ---------------------------------------------------------------------
// Sanity checks
// ---------------------------------------------------------------------
assert(pico_origin[0] + pico_size[0] + margin <= outer[0] - wall);
assert(pico_origin[1] + pico_size[1] + margin <= outer[1] - wall);
assert(as7341_origin[0] + as7341_size[0] + margin <= outer[0] - wall);
assert(as7341_origin[1] + as7341_size[1] + margin <= outer[1] - wall);
assert(led_ring_r * 2 + led_d <= probe_boss_od);
assert(probe_boss_od <= outer[1] - 2 * (wall + margin));
assert(batt_origin[0] >= pico_origin[0] && batt_origin[0] + batt_size[0] <= pico_origin[0] + pico_size[0]);
assert(batt_origin[1] >= pico_origin[1] && batt_origin[1] + batt_size[1] <= pico_origin[1] + pico_size[1]);
assert(batt_size[2] <= pico_standoff_h);

// ---------------------------------------------------------------------
// Base tray
// ---------------------------------------------------------------------
module pico_standoffs() {
    for (dx = [pico_hole_inset, pico_size[0] - pico_hole_inset])
        for (dy = [pico_hole_inset, pico_size[1] - pico_hole_inset])
            translate([pico_origin[0] + dx, pico_origin[1] + dy, wall])
            difference() {
                cylinder(h = pico_standoff_h, d = pico_standoff_d);
                translate([0, 0, -0.1])
                    cylinder(h = pico_standoff_h + 0.2, d = pico_hole_d);
            }
}

module battery_pocket() {
    // Shallow recess under the Pico so the cell sits flush with the floor,
    // clear of every standoff (verified by assertion above).
    translate([batt_origin[0], batt_origin[1], wall])
        cube([batt_size[0], batt_size[1], batt_size[2] + 0.4]);
}

module as7341_shelf() {
    translate([as7341_origin[0], as7341_origin[1], wall])
    difference() {
        cube([as7341_size[0], as7341_size[1], as7341_shelf_h]);
        // Central pocket so only a support lip touches the board underside.
        translate([1.5, 1.5, 1.5])
            cube([as7341_size[0] - 3, as7341_size[1] - 3, as7341_shelf_h]);
    }
    // Light-tight baffle socket that the fibre_guide tube's foot seats into,
    // directly over the sensor's optical aperture.
    translate([sensor_center[0], sensor_center[1], wall + as7341_shelf_h + as7341_size[2]])
        difference() {
            cylinder(h = 3, d = sensor_window_d + 3);
            translate([0, 0, -0.1]) cylinder(h = 3.2, d = sensor_window_d);
        }
}

module base_shell() {
    difference() {
        cube([outer[0], outer[1], base_height]);
        translate([wall, wall, wall])
            cube([outer[0] - 2 * wall, outer[1] - 2 * wall, base_height]);
        // USB access on the short end nearest the Pico's USB connector.
        translate([-0.1, pico_origin[1] + pico_size[1] / 2 - 5, wall + 0.5])
            cube([wall + 0.2, 10, 5]);
    }
}

module base() {
    difference() {
        base_shell();
        battery_pocket();
    }
    pico_standoffs();
    as7341_shelf();
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
    cube([outer[0], outer[1], wall]);
    // Downturned skirt that overlaps the base wall for a dust/light seal.
    difference() {
        translate([0, 0, -2.4])
            cube([outer[0], outer[1], 2.4]);
        translate([wall, wall, -2.5])
            cube([outer[0] - 2 * wall, outer[1] - 2 * wall, 2.6]);
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
    gap_span = base_height - (wall + as7341_shelf_h + as7341_size[2] + 3)
        + probe_boss_h;   // interior span the tube must bridge
    tube_len = max(gap_span, 8);
    difference() {
        union() {
            cylinder(h = tube_len, d = fiber_counterbore_d);
            // Register flange that seats in the AS7341 baffle socket.
            translate([0, 0, tube_len - 1.5])
                cylinder(h = 1.5, d = sensor_window_d + 2.6);
        }
        translate([0, 0, -0.1])
            cylinder(h = tube_len + 0.2, d = fiber_bore_d);
    }
}

// ---------------------------------------------------------------------
// Assembly / keepout preview (not for printing)
// ---------------------------------------------------------------------
module keepouts() {
    // Raspberry Pi Pico 2 footprint.
    color([0.10, 0.45, 0.25, 0.85])
        translate([pico_origin[0], pico_origin[1], wall + pico_standoff_h])
            cube(pico_size);
    // Battery, nested under the Pico.
    color([0.15, 0.15, 0.15, 0.85])
        translate([batt_origin[0], batt_origin[1], wall])
            cube(batt_size);
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
                        base_height + wall + probe_boss_h - 4])
                cylinder(h = 4, d = led_d - 0.4);
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
