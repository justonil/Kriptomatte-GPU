import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_array_backend(use_gpu: bool | None = None):
    """
    Selects the array backend used for mask computation.

    Returns a tuple (array_module, device_label). CuPy is used only when it is
    installed and a CUDA device is actually available; otherwise the function
    falls back to NumPy. ``use_gpu=False`` forces the NumPy path.
    """
    if use_gpu is False:
        return np, "cpu (numpy)"

    try:
        import cupy  # type: ignore
    except Exception as exc:
        # ImportError when CuPy is absent; other errors happen in frozen builds
        # with a broken CUDA setup. Either way, fall back to NumPy.
        if use_gpu:
            logger.warning(
                "GPU requested but CuPy is unavailable (%s); falling back to NumPy "
                "(install 'cupy-cuda13x' and a CUDA Toolkit for GPU support).",
                exc,
            )
        return np, "cpu (numpy)"

    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device found")
        device_name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", "replace")
    except Exception as exc:  # pragma: no cover - depends on the host machine
        logger.warning("CuPy found but no usable CUDA device (%s); falling back to NumPy.", exc)
        return np, "cpu (numpy)"

    logger.info("GPU acceleration enabled via CuPy on %s", device_name)
    return cupy, f"gpu (cupy, {device_name})"


def to_numpy(array, xp):
    """Copies a backend array back to host NumPy (no copy for the NumPy backend)."""
    if xp is np:
        return np.asarray(array)
    return xp.asnumpy(array)
