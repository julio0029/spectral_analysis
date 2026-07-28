"""Tk GUI for plug-and-run acquisition."""

from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .acquisition import AcquisitionRunner
from .hardware import (
    AS726xRaspberryPi,
    BROADBAND_ILLUMINATION,
    PICO_ILLUMINATION_SEQUENCE,
    PicoSerialSpectrometer,
    SimulatedSpectrometer,
)
from .offline import analyse_plant_recording, analyse_recording


DEFAULT_PINS = {385: 13, 400: 19, 457: 26, 650: 5}
AS7341_VISIBLE_CHANNELS = (415, 445, 480, 515, 555, 590, 630, 680)


class SpectralApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mitochondrial Spectral Device")
        self.geometry("960x650")
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.events: queue.Queue = queue.Queue()
        self.runner: AcquisitionRunner | None = None
        self.worker: threading.Thread | None = None
        self.latest_dark: dict | None = None
        self.elapsed: list[float] = []
        self.traces = {w: [] for w in AS7341_VISIBLE_CHANNELS}
        self._build()
        self.after(100, self._drain_events)

    def _build(self) -> None:
        controls = ttk.Frame(self, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)
        self.mode = tk.StringVar(value="Simulation, CubeSat")
        self.subject = tk.StringVar(value="test")
        self.folder = tk.StringVar(value=str(Path.cwd() / "data"))
        self.duration = tk.StringVar(value="60")
        self.pathlength = tk.StringVar(value="1.0")
        self.serial_port = tk.StringVar(value="")
        self.status = tk.StringVar(value="Ready")
        self.environment_status = tk.StringVar(value="CO₂: — ppm   Temperature: — °C   RH: — %")

        for column, (label, variable) in enumerate(
            [("Mode", self.mode), ("Sample ID", self.subject), ("Duration, s", self.duration)]
        ):
            ttk.Label(controls, text=label).grid(row=0, column=column, sticky="w", padx=4)
            if label == "Mode":
                ttk.Combobox(
                    controls,
                    textvariable=variable,
                    values=("Simulation, CubeSat", "Pico USB, CubeSat", "Raspberry Pi, AS7262 legacy"),
                    width=25,
                    state="readonly",
                ).grid(row=1, column=column, padx=4)
            else:
                ttk.Entry(controls, textvariable=variable, width=18).grid(row=1, column=column, padx=4)

        ttk.Label(controls, text="Output folder").grid(row=0, column=3, sticky="w", padx=4)
        ttk.Entry(controls, textvariable=self.folder, width=38).grid(row=1, column=3, padx=4)
        ttk.Button(controls, text="Browse", command=self.choose_folder).grid(row=1, column=4, padx=4)
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.grid(row=1, column=5, padx=5)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=6, padx=5)
        self.rehydrate_button = ttk.Button(controls, text="Release water…", command=self.rehydrate)
        self.rehydrate_button.grid(row=2, column=5, columnspan=2, padx=5, pady=(8, 0))
        ttk.Label(controls, text="Pico data port, optional").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Entry(controls, textvariable=self.serial_port, width=25).grid(
            row=2, column=1, columnspan=2, sticky="w", padx=4, pady=(8, 0)
        )

        analysis_controls = ttk.Frame(self, padding=(12, 2))
        analysis_controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(analysis_controls, text="Completed recordings can be fitted with an instrument-calibrated coefficient file.").pack(side=tk.LEFT)
        ttk.Button(analysis_controls, text="Analyse plant recording…", command=self.analyse_plant).pack(side=tk.RIGHT, padx=4)
        ttk.Button(analysis_controls, text="Analyse recording…", command=self.analyse).pack(side=tk.RIGHT, padx=4)
        ttk.Entry(analysis_controls, textvariable=self.pathlength, width=7).pack(side=tk.RIGHT, padx=2)
        ttk.Label(analysis_controls, text="Pathlength, cm").pack(side=tk.RIGHT)

        notice = (
            "CubeSat Pico mode records dark and white states plus 365, 450, 535, 550, 565, 575, 605, 630, 660, 700, 730 and 940 nm, "
            "with SCD-40 CO₂, temperature and humidity telemetry."
        )
        ttk.Label(self, text=notice, foreground="#8a4b08", wraplength=900).pack(fill=tk.X, padx=16)
        ttk.Label(self, textvariable=self.environment_status, anchor="center").pack(fill=tk.X, padx=16, pady=(4, 0))

        figure = Figure(figsize=(9, 4.8), dpi=100)
        self.axis = figure.add_subplot(111)
        self.axis.set_xlabel("Elapsed time, s")
        self.axis.set_ylabel("Dark-corrected detector counts")
        self.lines = {w: self.axis.plot([], [], label=f"{w} nm")[0] for w in self.traces}
        self.axis.legend(ncol=3)
        self.canvas = FigureCanvasTkAgg(figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor="w").pack(fill=tk.X, side=tk.BOTTOM)

    def _set_plot_channels(self, channels: tuple[int | str, ...]) -> None:
        numeric = tuple(int(value) for value in channels if isinstance(value, (int, float)))
        if tuple(self.traces) == numeric:
            return
        self.traces = {wavelength: [] for wavelength in numeric}
        self.axis.clear()
        self.axis.set_xlabel("Elapsed time, s")
        self.axis.set_ylabel("Dark-corrected detector counts")
        self.lines = {
            wavelength: self.axis.plot([], [], label=f"{wavelength} nm")[0]
            for wavelength in self.traces
        }
        self.axis.legend(ncol=4)
        self.canvas.draw_idle()

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.folder.get())
        if selected:
            self.folder.set(selected)

    def start(self) -> None:
        try:
            duration = float(self.duration.get())
            if duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid duration", "Duration must be a positive number of seconds.")
            return
        subject = "".join(c for c in self.subject.get().strip() if c.isalnum() or c in "-.")
        if not subject:
            messagebox.showerror("Invalid sample", "Enter a sample ID.")
            return
        mode_name = self.mode.get()
        try:
            if mode_name == "Simulation, CubeSat":
                hardware = SimulatedSpectrometer()
                illuminations = PICO_ILLUMINATION_SEQUENCE
            elif mode_name == "Pico USB, CubeSat":
                hardware = PicoSerialSpectrometer(port=self.serial_port.get().strip() or None)
                illuminations = PICO_ILLUMINATION_SEQUENCE
            else:
                hardware = AS726xRaspberryPi(DEFAULT_PINS)
                illuminations = (BROADBAND_ILLUMINATION, *DEFAULT_PINS)
        except Exception as exc:
            messagebox.showerror("Device connection failed", str(exc))
            return
        self._set_plot_channels(hardware.channels)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path(self.folder.get()) / f"{subject}-{stamp}.csv"
        self.runner = AcquisitionRunner(hardware, illuminations, samples=5)
        self.elapsed.clear()
        self.latest_dark = None
        for values in self.traces.values():
            values.clear()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.rehydrate_button.configure(state=tk.DISABLED)
        self.status.set(f"Recording to {output}")

        def work() -> None:
            try:
                self.runner.run(output, duration_s=duration, callback=self.events.put, metadata={"sample_id": subject, "mode": mode_name})
                self.events.put({"event": "finished", "output": str(output)})
            except Exception as exc:
                self.events.put({"event": "error", "message": str(exc)})
            finally:
                try:
                    hardware.close()
                except Exception as exc:
                    self.events.put({"event": "error", "message": f"Device shutdown failed: {exc}"})

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        if self.runner:
            self.runner.stop()
            self.status.set("Stopping after the current measurement…")

    def rehydrate(self) -> None:
        if self.mode.get() != "Pico USB, CubeSat":
            messagebox.showinfo("Pico mode required", "Select Pico USB, CubeSat before releasing water.")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Acquisition active", "Stop the acquisition before releasing water.")
            return
        confirmed = messagebox.askyesno(
            "Confirm one-shot rehydration",
            "Release water for 3 seconds? Only continue when the flight reservoir and wick are installed.",
        )
        if not confirmed:
            return
        hardware = None
        try:
            hardware = PicoSerialSpectrometer(port=self.serial_port.get().strip() or None)
            hardware.rehydrate(duration_s=3.0)
            self.status.set("Rehydration pulse completed")
            messagebox.showinfo("Rehydration complete", "The three-second water-release pulse completed.")
        except Exception as exc:
            messagebox.showerror("Rehydration failed", str(exc))
        finally:
            if hardware is not None:
                try:
                    hardware.close()
                except Exception as exc:
                    self.status.set(f"Rehydration completed, but device shutdown reported: {exc}")

    def analyse(self) -> None:
        recording = filedialog.askopenfilename(title="Select acquisition CSV", filetypes=(("CSV", "*.csv"),))
        if not recording:
            return
        coefficients = filedialog.askopenfilename(title="Select calibrated coefficient CSV", filetypes=(("CSV", "*.csv"),))
        if not coefficients:
            return
        try:
            pathlength = float(self.pathlength.get())
            if pathlength <= 0:
                raise ValueError
        except ValueError:
            pathlength = 1.0
        try:
            results = analyse_recording(recording, coefficients, pathlength_cm=pathlength)
            output = Path(recording).with_name(Path(recording).stem + "-redox.csv")
            results.to_csv(output, index=False)
            valid = int(results["valid"].sum())
            messagebox.showinfo("Analysis complete", f"Saved {len(results)} cycles ({valid} valid fits) to:\n{output}")
            self.status.set(f"Analysis saved: {output}")
        except Exception as exc:
            messagebox.showerror("Analysis failed", str(exc))

    def analyse_plant(self) -> None:
        recording = filedialog.askopenfilename(
            title="Select CubeSat acquisition CSV",
            filetypes=(("CSV", "*.csv"),),
        )
        if not recording:
            return
        try:
            results = analyse_plant_recording(recording)
            output = Path(recording).with_name(Path(recording).stem + "-plant.csv")
            results.to_csv(output, index=False)
            messagebox.showinfo(
                "Plant analysis complete",
                f"Saved {len(results)} cycles to:\n{output}",
            )
            self.status.set(f"Plant analysis saved: {output}")
        except Exception as exc:
            messagebox.showerror("Plant analysis failed", str(exc))

    def _drain_events(self) -> None:
        changed = False
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.get("event") == "finished":
                self.status.set(f"Finished: {event['output']}")
                self.start_button.configure(state=tk.NORMAL)
                self.stop_button.configure(state=tk.DISABLED)
                self.rehydrate_button.configure(state=tk.NORMAL)
            elif event.get("event") == "error":
                self.status.set("Acquisition failed")
                self.start_button.configure(state=tk.NORMAL)
                self.stop_button.configure(state=tk.DISABLED)
                self.rehydrate_button.configure(state=tk.NORMAL)
                messagebox.showerror("Acquisition failed", event["message"])
            elif event.get("illumination_nm") == 0:
                self.latest_dark = event
            elif event.get("illumination_nm") == BROADBAND_ILLUMINATION and self.latest_dark:
                self.elapsed.append(event["elapsed_s"])
                for wavelength in self.traces:
                    value = event[f"signal_{wavelength}"] - self.latest_dark[f"signal_{wavelength}"]
                    self.traces[wavelength].append(value)
                changed = True
            if event.get("co2_ppm") is not None:
                self.environment_status.set(
                    f"CO₂: {event['co2_ppm']:.0f} ppm   "
                    f"Temperature: {event['temperature_c']:.1f} °C   "
                    f"RH: {event['relative_humidity_pct']:.1f} %"
                )
        if changed:
            for wavelength, line in self.lines.items():
                line.set_data(self.elapsed, self.traces[wavelength])
            self.axis.relim()
            self.axis.autoscale_view()
            self.canvas.draw_idle()
        self.after(100, self._drain_events)

    def close_app(self) -> None:
        self.stop()
        self.destroy()


def main() -> None:
    SpectralApp().mainloop()


if __name__ == "__main__":
    main()
