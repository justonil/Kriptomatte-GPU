# -*- mode: python ; coding: utf-8 -*-
#
# Folder ("onedir") build. Compared to km.spec (single-file build) this avoids
# unpacking the bundle on every launch, which saves several seconds per run.
# Output: dist/km/km.exe together with a dist/km/_internal folder.

from PyInstaller.utils.hooks import collect_all

# CuPy is optional: when it is installed the executable can use the GPU, so its
# native libraries and lazily imported submodules have to be bundled explicitly.
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

# Drop the CUDA Toolkit runtime DLLs PyInstaller found on the build machine.
# Bundling nvrtc breaks CuPy's runtime detection and crashes with
# "[WinError 2] ...Temp\bin"; the target machine's CUDA Toolkit is used instead.
a.binaries = [
    entry for entry in a.binaries
    if 'NVIDIA GPU Computing Toolkit' not in str(entry[1])
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='km',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='km',
)
