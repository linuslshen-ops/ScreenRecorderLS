# screen_recorder.spec — PyInstaller build specification
import os, sys, site

# --------------------------------------------------------------------------- #
#  Resolve paths to data files that must be bundled                           #
# --------------------------------------------------------------------------- #

sp = next(p for p in site.getsitepackages() if os.path.isdir(os.path.join(p, '_sounddevice_data')))
portaudio_dir = os.path.join(sp, '_sounddevice_data', 'portaudio-binaries')

datas = [
    # PortAudio DLLs required by sounddevice
    # (imageio_ffmpeg's own PyInstaller hook handles the ffmpeg binary)
    (portaudio_dir, '_sounddevice_data/portaudio-binaries'),
]

# --------------------------------------------------------------------------- #
#  Analysis                                                                    #
# --------------------------------------------------------------------------- #

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'mss',
        'mss.windows',
        'soundcard',
        'soundcard.mediafoundation',
        'sounddevice',
        'cv2',
        'numpy',
        'imageio_ffmpeg',
        'cffi',
        '_cffi_backend',
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'recorder',
        'app',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages to keep bundle smaller
        'torch', 'torchvision', 'scipy', 'matplotlib', 'pandas',
        'sympy', 'skimage', 'PIL', 'tkinter',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ScreenRecorder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,        # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ScreenRecorder',
)
