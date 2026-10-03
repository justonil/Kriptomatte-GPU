# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

# CuPy is optional: when it is installed the executable can use the GPU, so its
# native libraries and lazily imported submodules have to be bundled explicitly.
# collect_all tolerates a missing package and simply returns empty lists.
cupy_datas, cupy_binaries, cupy_hiddenimports = collect_all('cupy')
backends_datas, backends_binaries, backends_hiddenimports = collect_all('cupy_backends')

a = Analysis(
    ['kriptomatte\\interface\\cli.py'],
    pathex=[],
    binaries=cupy_binaries + backends_binaries,
    datas=cupy_datas + backends_datas,
    hiddenimports=cupy_hiddenimports + backends_hiddenimports + ['graphlib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# PyInstaller also pulls the CUDA Toolkit runtime DLLs found on the build
# machine (cublas, cufft, nvrtc, ...). Bundling them breaks CuPy's runtime
# detection: it finds the bundled nvrtc, derives a wrong CUDA root and crashes
# with "[WinError 2] ...Temp\\bin". Drop them so the executable uses the CUDA
# Toolkit installed on the target machine, exactly like a normal `import cupy`.
# This also cuts the executable size by several hundred megabytes.
a.binaries = [
    entry for entry in a.binaries
    if 'NVIDIA GPU Computing Toolkit' not in str(entry[1])
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='km',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
