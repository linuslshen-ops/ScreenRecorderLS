"""
app.py — MainWindow: all UI layout, signal-slot wiring, and recording state machine.
"""

import os

import cv2
import mss
import numpy as np
import soundcard
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from recorder import RecorderThread


PREVIEW_W = 640
PREVIEW_H = 360


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Recorder")
        self.setMinimumWidth(700)

        self._recorder_thread = None
        self._is_recording = False
        self._output_folder = os.path.expanduser("~\\Videos")

        self._build_ui()
        self._populate_monitors()
        self._populate_system_audio_devices()
        self._populate_mic_devices()

        # Start idle preview timer
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(100)  # 10 fps
        self._preview_timer.timeout.connect(self._update_idle_preview)
        self._preview_timer.start()

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(self._build_settings_group())
        layout.addWidget(self._build_preview_group())
        layout.addLayout(self._build_controls_layout())

        self.setStatusBar(QStatusBar())

    def _build_settings_group(self):
        group = QGroupBox("Settings")
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # Monitor selection
        self.monitor_combo = QComboBox()
        form.addRow("Monitor:", self.monitor_combo)

        # Output folder
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(self._output_folder)
        self.folder_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._on_browse_folder)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(browse_btn)
        form.addRow("Output Folder:", folder_row)

        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")
        form.addRow("Frame Rate:", self.fps_spin)

        # System audio
        sys_audio_row = QHBoxLayout()
        self.sys_audio_check = QCheckBox("Enable")
        self.sys_audio_combo = QComboBox()
        self.sys_audio_combo.setEnabled(False)
        self.sys_audio_check.toggled.connect(self.sys_audio_combo.setEnabled)
        sys_audio_row.addWidget(self.sys_audio_check)
        sys_audio_row.addWidget(self.sys_audio_combo, stretch=1)
        form.addRow("System Audio:", sys_audio_row)

        # Microphone
        mic_row = QHBoxLayout()
        self.mic_check = QCheckBox("Enable")
        self.mic_combo = QComboBox()
        self.mic_combo.setEnabled(False)
        self.mic_check.toggled.connect(self.mic_combo.setEnabled)
        mic_row.addWidget(self.mic_check)
        mic_row.addWidget(self.mic_combo, stretch=1)
        form.addRow("Microphone:", mic_row)

        return group

    def _build_preview_group(self):
        group = QGroupBox("Preview")
        vbox = QVBoxLayout(group)
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1a1a1a;")
        vbox.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        return group

    def _build_controls_layout(self):
        hbox = QHBoxLayout()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setFixedHeight(40)
        self._set_btn_idle()
        self.record_btn.clicked.connect(self._on_record_toggled)

        self.timer_label = QLabel("00:00")
        self.timer_label.setFixedWidth(60)
        font = self.timer_label.font()
        font.setPointSize(14)
        font.setBold(True)
        self.timer_label.setFont(font)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hbox.addStretch()
        hbox.addWidget(self.timer_label)
        hbox.addWidget(self.record_btn)
        hbox.addStretch()
        return hbox

    # ------------------------------------------------------------------ #
    #  Populate combos                                                     #
    # ------------------------------------------------------------------ #

    def _populate_monitors(self):
        self.monitor_combo.clear()
        with mss.mss() as sct:
            for i, m in enumerate(sct.monitors[1:]):
                is_primary = m["left"] == 0 and m["top"] == 0
                label = f"Monitor {i + 1}  —  {m['width']}×{m['height']}"
                if is_primary:
                    label += "  (Primary)"
                self.monitor_combo.addItem(label, userData=i)

    def _populate_system_audio_devices(self):
        self.sys_audio_combo.clear()
        try:
            speakers = soundcard.all_speakers()
            for sp in speakers:
                self.sys_audio_combo.addItem(sp.name, userData=sp.name)
            # Default to the default speaker
            default = soundcard.default_speaker()
            idx = self.sys_audio_combo.findData(default.name)
            if idx >= 0:
                self.sys_audio_combo.setCurrentIndex(idx)
        except Exception:
            self.sys_audio_combo.addItem("(No loopback devices found)")

    def _populate_mic_devices(self):
        self.mic_combo.clear()
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    self.mic_combo.addItem(d["name"], userData=i)
            # Select default input device
            default_idx = sd.default.device[0]  # input device index
            for combo_idx in range(self.mic_combo.count()):
                if self.mic_combo.itemData(combo_idx) == default_idx:
                    self.mic_combo.setCurrentIndex(combo_idx)
                    break
        except Exception:
            self.mic_combo.addItem("(No microphone devices found)")

    # ------------------------------------------------------------------ #
    #  Idle preview                                                        #
    # ------------------------------------------------------------------ #

    def _update_idle_preview(self):
        if self._is_recording:
            return
        monitor_idx = self.monitor_combo.currentData()
        if monitor_idx is None:
            return
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[monitor_idx + 1]
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
            self._display_frame(frame)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Frame display                                                       #
    # ------------------------------------------------------------------ #

    def _display_frame(self, frame):
        """Convert a BGRA or BGR numpy array to QPixmap and display in preview_label."""
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # Scale down with cv2 (faster than Qt SmoothTransformation on full-res frames)
        h, w = frame.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).copy()
        qt_image = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
        qt_image = qt_image.copy()  # detach from numpy buffer
        self.preview_label.setPixmap(QPixmap.fromImage(qt_image))

    # ------------------------------------------------------------------ #
    #  Recording control                                                   #
    # ------------------------------------------------------------------ #

    def _on_browse_folder(self):
        # Pause the idle preview timer while the dialog is open — mss screen
        # captures running during a native Windows dialog can cause a crash.
        self._preview_timer.stop()
        try:
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Output Folder",
                self._output_folder,
                QFileDialog.Option.DontUseNativeDialog,
            )
        finally:
            self._preview_timer.start()
        if folder:
            self._output_folder = folder
            self.folder_edit.setText(folder)

    def _on_record_toggled(self):
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self._validate_settings():
            return

        monitor_idx = self.monitor_combo.currentData()
        sys_device = (
            self.sys_audio_combo.currentData()
            if self.sys_audio_check.isChecked()
            else None
        )
        mic_sd_index = (
            self.mic_combo.currentData()
            if self.mic_check.isChecked()
            else None
        )
        # Determine channel count for selected mic
        mic_channels = 1
        if mic_sd_index is not None:
            try:
                mic_channels = min(
                    sd.query_devices(mic_sd_index)["max_input_channels"], 2
                )
            except Exception:
                mic_channels = 1

        self._recorder_thread = RecorderThread(
            monitor_index=monitor_idx,
            output_folder=self._output_folder,
            fps=self.fps_spin.value(),
            record_system_audio=self.sys_audio_check.isChecked(),
            system_audio_device=sys_device,
            record_microphone=self.mic_check.isChecked(),
            mic_device=mic_sd_index,
            mic_channels=mic_channels,
        )
        self._recorder_thread.preview_frame_ready.connect(self._on_preview_frame)
        self._recorder_thread.status_update.connect(self._on_status_update)
        self._recorder_thread.recording_finished.connect(self._on_recording_finished)
        self._recorder_thread.error_occurred.connect(self._on_error)
        self._recorder_thread.finished.connect(self._on_thread_done)

        self._set_ui_recording(True)
        self._recorder_thread.start()

    def _stop_recording(self):
        if self._recorder_thread:
            self._recorder_thread.stop()
        # Disable button until thread finishes
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Finishing…")
        self.record_btn.setStyleSheet("")

    def _validate_settings(self):
        if not os.path.isdir(self._output_folder):
            QMessageBox.warning(self, "Invalid Output Folder",
                                f"The output folder does not exist:\n{self._output_folder}")
            return False
        if self.monitor_combo.count() == 0:
            QMessageBox.warning(self, "No Monitor", "No monitors detected.")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Slots (called from Qt main thread via signals)                     #
    # ------------------------------------------------------------------ #

    def _on_preview_frame(self, frame):
        self._display_frame(frame)

    def _on_status_update(self, text):
        # Extract "MM:SS" from "Recording... MM:SS"
        parts = text.split()
        if parts:
            self.timer_label.setText(parts[-1])
        self.statusBar().showMessage(text)

    def _on_recording_finished(self, path):
        self._set_ui_recording(False)
        self.timer_label.setText("00:00")
        self.statusBar().showMessage(f"Saved: {path}", 5000)
        QMessageBox.information(
            self,
            "Recording Saved",
            f"Your recording has been saved to:\n\n{path}",
        )

    def _on_error(self, msg):
        self._set_ui_recording(False)
        self.timer_label.setText("00:00")
        QMessageBox.critical(self, "Recording Error", f"An error occurred:\n\n{msg}")

    def _on_thread_done(self):
        # Fallback: re-enable UI if signals didn't already do it
        if not self._is_recording:
            self._set_ui_recording(False)

    # ------------------------------------------------------------------ #
    #  UI state helpers                                                    #
    # ------------------------------------------------------------------ #

    def _set_ui_recording(self, recording: bool):
        self._is_recording = recording
        controls = [
            self.monitor_combo,
            self.folder_edit,
            self.fps_spin,
            self.sys_audio_check,
            self.sys_audio_combo,
            self.mic_check,
            self.mic_combo,
        ]
        for ctrl in controls:
            ctrl.setEnabled(not recording)

        self.record_btn.setEnabled(True)
        if recording:
            self._set_btn_recording()
        else:
            self._set_btn_idle()

    def _set_btn_idle(self):
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background-color: #388e3c; }"
        )

    def _set_btn_recording(self):
        self.record_btn.setText("Stop Recording")
        self.record_btn.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; "
            "font-weight: bold; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
        )

    def closeEvent(self, event):
        if self._is_recording:
            reply = QMessageBox.question(
                self,
                "Recording in Progress",
                "A recording is in progress. Stop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._stop_recording()
            if self._recorder_thread:
                self._recorder_thread.wait(10000)  # up to 10s on close
        event.accept()
