"""
recorder.py — Screen capture and audio recording threads.
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
    """Return the ffmpeg executable path, working both in development and as a PyInstaller bundle.

    imageio_ffmpeg's own PyInstaller hook places the binary inside the bundle's
    imageio_ffmpeg/binaries/ folder, and get_ffmpeg_exe() uses importlib.resources
    to locate it — so it works correctly in frozen apps without extra handling.
    """
    return imageio_ffmpeg.get_ffmpeg_exe()


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

    def run(self):
        try:
            if self.mode == "loopback":
                self._record_loopback()
            else:
                self._record_microphone()
        finally:
            self._write_wav()

    def _record_loopback(self):
        mic = soundcard.get_microphone(self.device_name, include_loopback=True)
        with mic.recorder(samplerate=self.sample_rate, channels=self.channels) as recorder:
            while not self._stop_event.is_set():
                data = recorder.record(numframes=1024)
                self._frames.append(data.copy())

    def _record_microphone(self):
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
        # Clip and convert float32 → int16
        data_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(self.wav_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(data_int16.tobytes())


class RecorderThread(QThread):
    """Captures screen frames and manages audio threads. Muxes everything into MP4 on stop."""

    preview_frame_ready = pyqtSignal(object)   # carries BGR numpy array
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
    ):
        super().__init__()
        self.monitor_index = monitor_index          # 0-based (maps to sct.monitors[index+1])
        self.output_folder = output_folder
        self.fps = float(fps)
        self.record_system_audio = record_system_audio
        self.system_audio_device = system_audio_device
        self.record_microphone = record_microphone
        self.mic_device = mic_device
        self.mic_channels = mic_channels

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
                    sample_rate=44100,
                    channels=self.mic_channels,
                )
                self._mic_rec.start()

            # Capture video frames
            self._capture_loop()

            # Stop audio and wait for WAV files to be written
            if self._sys_audio_rec:
                self._sys_audio_rec.stop()
                self._sys_audio_rec.join()
            if self._mic_rec:
                self._mic_rec.stop()
                self._mic_rec.join()

            # Mux into MP4
            final_path = self._mux_to_mp4()
            self.recording_finished.emit(final_path)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self._cleanup_temp_files()

    def _capture_loop(self):
        frame_interval = 1.0 / self.fps
        preview_interval = 1.0 / 10.0   # ~10 fps preview
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

                    writer.write(frame_bgr)

                    now = time.monotonic()

                    # Emit preview frame (throttled)
                    if (now - last_preview_time) >= preview_interval:
                        self.preview_frame_ready.emit(frame_bgr.copy())
                        last_preview_time = now

                    # Emit status update (every second)
                    if (now - last_status_time) >= 1.0:
                        elapsed = int(now - start_time)
                        mins, secs = divmod(elapsed, 60)
                        self.status_update.emit(f"Recording... {mins:02d}:{secs:02d}")
                        last_status_time = now

                    # Frame pacing
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
