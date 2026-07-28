# CubeSat payload CAD exemplar

`cubesat_radish_payload.scad` is a parametric packaging model constrained to
80 × 40 × 40 mm. Open it in OpenSCAD and set `part` to `lower`, `deck`, `lid`,
`tray` or `assembly`. Export the printable parts separately as STL files.

The coloured objects in assembly mode are component keep-outs. They are not
printable representations of the electronics. The model reserves space for a
vertical Pico 2, vertical Adafruit SCD-40 breakout, 28 mm custom optical PCB,
3 mL reservoir and cotton tray, pump or valve, and 15 mm mixing fan.

Before fabrication, add the selected gasket groove, fasteners, launch restraint,
spacecraft mounting interface, tubing details and material-specific clearances.
The printed model must not be treated as a pressure vessel or flight article.
