"""
recorder.py — Screen capture, audio recording, and webcam capture threads.
"""

import os
import sys
import time
import wave
import tempfile
import threading
import datetime
import subprocess

import numpy as np
import cv2
import mss
import soundcard
import sounddevice as sd
import imageio_ffmpeg
from PyQt6.QtCore import QThread, pyqtSignal


def _get_ffmpeg_exe():
    """Return the ffmpeg executable path (works both in development and PyInstaller bundle)."""
    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------- #
#  Webcam capture                                                              #
# --------------------------------------------------------------------------- #

class WebcamCapture(threading.Thread):
    """Reads webcam frames in a background thread. Call get_frame() from any thread."""

    def __init__(self, device_index=0):
        super().__init__(daemon=True)
        self.device_index = device_index
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame = None

    def run(self):
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
                else:
                    time.sleep(0.01)
        finally:
            cap.release()

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop_event.set()


# --------------------------------------------------------------------------- #
#  Audio recording                                                             #
# --------------------------------------------------------------------------- #

class AudioRecorder(threading.Thread):
    """Records audio from a loopback (system) or microphone source."""

    def __init__(self, device_name, wav_path, mode, sample_rate=44100, channels=2):
        super().__init__(daemon=True)
        self.device_name = device_name
        self.wav_path = wav_path
        self.mode = mode            # "loopback" or "microphone"
        self.sample_rate = sample_rate
        self.channels = channels
        self._stop_event = threading.Event()
        self._frames = []
        self.error = None           # Stores exception if recording fails

    def run(self):
        try:
            if self.mode == "loopback":
                self._record_loopback()
            else:
                self._record_microphone()
        except Exception as e:
            self.error = e
        finally:
            self._write_wav()

    def _record_loopback(self):
        mic = soundcard.get_microphone(self.device_name, include_loopback=True)
        with mic.recorder(samplerate=self.sample_rate, channels=self.channels) as recorder:
            while not self._stop_event.is_set():
                data = recorder.record(numframes=1024)
                self._frames.append(data.copy())

    def _record_microphone(self):
        # Query actual device capabilities and use its native sample rate
        device_info = sd.query_devices(self.device_name)
        native_rate = int(device_info["default_samplerate"])
        actual_channels = min(self.channels, int(device_info["max_input_channels"]))
        if actual_channels < 1:
            actual_channels = 1

        self.sample_rate = native_rate
        self.channels = actual_channels

        with sd.InputStream(
            device=self.device_name,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=1024,
        ) as stream:
            while not self._stop_event.is_set():
                data, _ = stream.read(1024)
                self._frames.append(data.copy())

    def stop(self):
        self._stop_event.set()

    def _write_wav(self):
        if not self._frames:
            return
        data = np.concatenate(self._frames, axis=0)
        data_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(self.wav_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(data_int16.tobytes())


# --------------------------------------------------------------------------- #
#  Screen recording thread                                                     #
# --------------------------------------------------------------------------- #

class RecorderThread(QThread):
    """Captures screen frames, composites webcam, manages audio. Muxes into MP4."""

    preview_frame_ready = pyqtSignal(object)   # BGR numpy array
    status_update = pyqtSignal(str)            # "Recording... MM:SS"
    recording_finished = pyqtSignal(str)       # final output file path
    error_occurred = pyqtSignal(str)           # error message

    def __init__(
        self,
        monitor_index,
        output_folder,
        fps,
        record_system_audio,
        system_audio_device,
        record_microphone,
        mic_device,
        mic_channels,
        webcam_capture=None,
        webcam_pos=(0.75, 0.75),
        webcam_diameter_frac=0.15,
    ):
        super().__init__()
        self.monitor_index = monitor_index
        self.output_folder = output_folder
        self.fps = float(fps)
        self.record_system_audio = record_system_audio
        self.system_audio_device = system_audio_device
        self.record_microphone = record_microphone
        self.mic_device = mic_device
        self.mic_channels = mic_channels

        # Webcam
        self.webcam_capture = webcam_capture           # shared WebcamCapture instance
        self.webcam_pos = webcam_pos                   # normalised (x, y) centre position
        self.webcam_diameter_frac = webcam_diameter_frac  # fraction of screen height

        self._stop_event = threading.Event()
        self._temp_avi = os.path.join(tempfile.gettempdir(), "_screenrec_video.avi")
        self._temp_sys_wav = os.path.join(tempfile.gettempdir(), "_screenrec_sys.wav")
        self._temp_mic_wav = os.path.join(tempfile.gettempdir(), "_screenrec_mic.wav")
        self._sys_audio_rec = None
        self._mic_rec = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            # Start audio threads before video (reduces AV desync)
            if self.record_system_audio and self.system_audio_device:
                self._sys_audio_rec = AudioRecorder(
                    device_name=self.system_audio_device,
                    wav_path=self._temp_sys_wav,
                    mode="loopback",
                    sample_rate=44100,
                    channels=2,
                )
                self._sys_audio_rec.start()

            if self.record_microphone and self.mic_device is not None:
                self._mic_rec = AudioRecorder(
                    device_name=self.mic_device,
                    wav_path=self._temp_mic_wav,
                    mode="microphone",
                    sample_rate=44100,       # overridden by device default in _record_microphone
                    channels=self.mic_channels,
                )
                self._mic_rec.start()

            # Capture video frames
            self._capture_loop()

            # Stop audio and wait for WAV files
            if self._sys_audio_rec:
                self._sys_audio_rec.stop()
                self._sys_audio_rec.join()
            if self._mic_rec:
                self._mic_rec.stop()
                self._mic_rec.join()

            # Check for audio errors
            errors = []
            if self._sys_audio_rec and self._sys_audio_rec.error:
                errors.append(f"System audio: {self._sys_audio_rec.error}")
            if self._mic_rec and self._mic_rec.error:
                errors.append(f"Microphone: {self._mic_rec.error}")

            # Mux into MP4
            final_path = self._mux_to_mp4()

            if errors:
                self.recording_finished.emit(final_path)
                self.error_occurred.emit(
                    "Recording saved but audio had issues:\n" + "\n".join(errors)
                )
            else:
                self.recording_finished.emit(final_path)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self._cleanup_temp_files()

    def _capture_loop(self):
        frame_interval = 1.0 / self.fps
        preview_interval = 1.0 / 10.0
        last_preview_time = 0.0
        last_status_time = 0.0

        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index + 1]
            w, h = monitor["width"], monitor["height"]

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(self._temp_avi, fourcc, self.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {self._temp_avi}")

            start_time = time.monotonic()

            try:
                while not self._stop_event.is_set():
                    loop_start = time.monotonic()

                    sct_img = sct.grab(monitor)
                    frame_bgra = np.array(sct_img)
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

                    # Composite webcam overlay
                    if self.webcam_capture:
                        cam_frame = self.webcam_capture.get_frame()
                        if cam_frame is not None:
                            diameter = int(h * self.webcam_diameter_frac)
                            cx = int(self.webcam_pos[0] * w)
                            cy = int(self.webcam_pos[1] * h)
                            _composite_webcam(frame_bgr, cam_frame, cx, cy, diameter)

                    writer.write(frame_bgr)

                    now = time.monotonic()

                    if (now - last_preview_time) >= preview_interval:
                        self.preview_frame_ready.emit(frame_bgr.copy())
                        last_preview_time = now

                    if (now - last_status_time) >= 1.0:
                        elapsed = int(now - start_time)
                        mins, secs = divmod(elapsed, 60)
                        self.status_update.emit(f"Recording... {mins:02d}:{secs:02d}")
                        last_status_time = now

                    elapsed_loop = time.monotonic() - loop_start
                    sleep_time = frame_interval - elapsed_loop
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                writer.release()

    def _mux_to_mp4(self):
        ffmpeg_exe = _get_ffmpeg_exe()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_folder, f"recording_{timestamp}.mp4")

        has_sys = self.record_system_audio and os.path.exists(self._temp_sys_wav)
        has_mic = self.record_microphone and os.path.exists(self._temp_mic_wav)

        cmd = [ffmpeg_exe, "-y", "-i", self._temp_avi]

        if has_sys:
            cmd += ["-i", self._temp_sys_wav]
        if has_mic:
            cmd += ["-i", self._temp_mic_wav]

        cmd += ["-map", "0:v"]

        if has_sys and has_mic:
            cmd += [
                "-filter_complex",
                "[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map", "[aout]",
            ]
        elif has_sys:
            cmd += ["-map", "1:a"]
        elif has_mic:
            cmd += ["-map", "1:a"]

        cmd += [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]

        # If no audio streams, remove codec flags for audio
        if not has_sys and not has_mic:
            cmd = [ffmpeg_exe, "-y", "-i", self._temp_avi,
                   "-map", "0:v",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   output_path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg muxing failed:\n{result.stderr}")

        return output_path

    def _cleanup_temp_files(self):
        for path in (self._temp_avi, self._temp_sys_wav, self._temp_mic_wav):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
#  Webcam compositing helper                                                   #
# --------------------------------------------------------------------------- #

def _composite_webcam(screen, cam, cx, cy, diameter):
    """Draw a circular webcam overlay centred at (cx, cy) on *screen* (in-place)."""
    sh, sw = screen.shape[:2]
    ch, cw = cam.shape[:2]

    # Crop webcam to square and resize
    side = min(ch, cw)
    y0, x0 = (ch - side) // 2, (cw - side) // 2
    square = cam[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(square, (diameter, diameter), interpolation=cv2.INTER_LINEAR)

    # Top-left corner of the overlay
    x1 = cx - diameter // 2
    y1 = cy - diameter // 2

    # Clamp to screen bounds
    x1 = max(0, min(x1, sw - diameter))
    y1 = max(0, min(y1, sh - diameter))

    # Circular mask
    mask = np.zeros((diameter, diameter), dtype=np.uint8)
    cv2.circle(mask, (diameter // 2, diameter // 2), diameter // 2, 255, -1)
    mask_bool = mask[:, :, np.newaxis] > 0

    roi = screen[y1:y1 + diameter, x1:x1 + diameter]
    np.copyto(roi, resized, where=mask_bool)

    # White border
    cv2.circle(screen, (x1 + diameter // 2, y1 + diameter // 2),
               diameter // 2, (255, 255, 255), 3)
