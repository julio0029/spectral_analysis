// Parametric 80 x 40 x 40 mm CubeSat radish payload exemplar.
// Units are millimetres. This is a packaging and prototyping model, not a
// flight-qualified pressure vessel. Set part to lower, deck, lid, tray or
// assembly before exporting an STL.

part = "assembly";
$fn = 48;

outer = [80, 40, 40];
wall = 2;
growth_height = 26;
deck_thickness = 2;
lid_height = outer[2] - growth_height - deck_thickness;

assert(outer[0] <= 80 && outer[1] <= 40 && outer[2] <= 40);

module lower_cassette() {
    difference() {
        cube([outer[0], outer[1], growth_height]);
        // Open-top growth volume. A gasketed deck closes this cavity.
        translate([wall, wall, wall])
            cube([outer[0] - 2*wall, outer[1] - 2*wall, growth_height]);
    }
    // Dry side service channel for a vertically mounted 51 x 21 mm Pico.
    translate([24, 7, wall]) cube([54, 1.2, growth_height - wall]);
    // End stops for the removable biological cassette.
    translate([3, 8.2, wall]) cube([1.2, 28, 8]);
    translate([32, 8.2, wall]) cube([1.2, 28, 8]);
}

module optical_environment_deck() {
    difference() {
        translate([0, 0, growth_height])
            cube([outer[0], outer[1], deck_thickness]);
        // AS7341 optical aperture above the seed and cotyledon field.
        translate([18, 22, growth_height - 0.1]) cylinder(h=deck_thickness + 0.2, d=8);
        // SCD-40 gas port, covered by a hydrophobic membrane in flight.
        translate([66, 14, growth_height - 0.1])
            cube([8, 12, deck_thickness + 0.2]);
        // Captive water line from reservoir to cotton wick.
        translate([5, 20, growth_height - 0.1])
            cylinder(h=deck_thickness + 0.2, d=3);
    }
}

module electronics_lid() {
    translate([0, 0, growth_height + deck_thickness])
    difference() {
        cube([outer[0], outer[1], lid_height]);
        translate([wall, wall, 0])
            cube([outer[0] - 2*wall, outer[1] - 2*wall, lid_height - wall]);
    }
}

module biological_tray() {
    // Two-part tray: lower collapsible reservoir and upper cotton/seed cup.
    difference() {
        cube([28, 24, 8]);
        translate([2, 2, 4]) cube([24, 20, 5]);
        translate([4, 4, 1]) cube([20, 16, 3.2]);
    }
    // Wick transfer ports, no unrestrained water droplets.
    for (x = [8, 14, 20])
        translate([x, 12, 3.5]) cylinder(h=4.8, d=2.2);
}

module keepouts() {
    // Biological tray, x=4..32, y=10..34, z=2..10.
    color([0.70, 0.85, 1.0, 0.75])
        translate([4, 10, 2]) biological_tray();
    // Cotton pad and radish seed.
    color([0.95, 0.95, 0.90, 0.8])
        translate([7, 13, 6]) cube([22, 18, 3]);
    color([0.55, 0.25, 0.12])
        translate([18, 22, 9]) sphere(d=3.5);
    // Early sprout growth keepout, about 15 mm from the seed.
    color([0.2, 0.75, 0.25, 0.35])
        translate([18, 22, 10]) cylinder(h=14, d1=3, d2=8);

    // Pico 2, mounted vertically inside the isolated side service channel.
    color([0.10, 0.45, 0.25, 0.8])
        translate([26, 2.5, 3]) cube([51, 4, 21]);
    // Adafruit SCD-40 breakout, vertical with sensing face toward headspace.
    color([0.25, 0.25, 0.75, 0.8])
        translate([70.3, 10, 1]) cube([7.7, 22.8, 25.5]);
    // Custom 28 mm optical PCB: AS7341, LED ring, drivers and baffle.
    color([0.45, 0.20, 0.55, 0.8])
        translate([4, 8, growth_height + deck_thickness]) cube([28, 28, 3]);
    // Pump/valve and manifold allocation.
    color([0.75, 0.45, 0.10, 0.8])
        translate([35, 25, growth_height + deck_thickness]) cube([20, 10, 7]);
    // 15 mm chamber mixing fan allocation.
    color([0.35, 0.35, 0.35, 0.8])
        translate([58, 22, growth_height + deck_thickness]) cube([15, 15, 4]);
}

if (part == "lower") {
    lower_cassette();
} else if (part == "deck") {
    optical_environment_deck();
} else if (part == "lid") {
    electronics_lid();
} else if (part == "tray") {
    biological_tray();
} else {
    color([0.75, 0.75, 0.78, 0.35]) lower_cassette();
    color([0.80, 0.80, 0.82, 0.55]) optical_environment_deck();
    color([0.85, 0.85, 0.88, 0.22]) electronics_lid();
    keepouts();
}
