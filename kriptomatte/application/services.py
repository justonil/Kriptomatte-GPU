import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kriptomatte.domain.repositories import ImageRepository
from kriptomatte.domain.services.masking import MaskCompositionService
from kriptomatte.domain.services.visualization import BitwiseColorService
from kriptomatte.domain.model.value_objects import CryptoID
from kriptomatte.infrastructure.compute_backend import get_array_backend, to_numpy
from kriptomatte.infrastructure.io.image_writer import ImageWriter

logger = logging.getLogger(__name__)


class CryptomatteExtractionService:
    def __init__(self, repo: ImageRepository, max_workers: int | None = None, use_gpu: bool | None = None):
        self.repo = repo
        self.max_workers = max_workers
        self.use_gpu = use_gpu

    def extract_all(self, file_path: str, output_dir: str | None = None):
        """
        Extracts all masks from the given EXR file.
        """
        if output_dir is None:
            output_dir = os.path.dirname(file_path)

        logger.info(f"Starting extraction for {file_path}")

        # 1. Reconstitute Aggregate
        try:
            exr_image = self.repo.load_header(file_path)
        except Exception as e:
            logger.error(f"Failed to load EXR header: {e}")
            raise

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        xp, device_label = get_array_backend(self.use_gpu)
        workers = self.max_workers or (os.cpu_count() or 1)
        logger.info(f"Compute backend: {device_label}; worker threads: {workers}")

        # Mask computation (GPU when available) runs on this thread while a pool
        # of worker threads encodes the PNG files. The semaphore bounds how many
        # finished masks may wait in the queue, which keeps RAM usage flat while
        # still letting the producer run ahead and overlap I/O with encoding.
        # The queue is also capped by a byte budget so huge frames stay safe.
        pixels = max(exr_image.window.height * exr_image.window.width, 1)
        in_flight = max(8, min(workers * 4, (1_000_000_000 // pixels) or 1))
        slots = threading.BoundedSemaphore(in_flight)
        futures = []

        def submit_save(path: str, mask: np.ndarray):
            slots.acquire()
            future = save_pool.submit(ImageWriter.save_mask, path, mask)
            future.add_done_callback(lambda _f: slots.release())
            futures.append(future)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="km-save") as save_pool:
            for layer in exr_image.layers:
                logger.info(f"Processing layer: {layer.name}")

                # Create a folder for this layer
                layer_folder = os.path.join(output_dir, f"{base_name}_{layer.name}")
                os.makedirs(layer_folder, exist_ok=True)

                # 2. Load heavy data only when needed
                logger.info(f"Reading channels for {layer.name}")
                raw_data = self.repo.read_channels(file_path, layer.channel_names)

                # --- OPTIMIZATION START ---
                logger.info(f"Analyzing visible objects in {layer.name}...")

                # Cryptomatte channels are alternating: [ID, Coverage, ID, Coverage, ...]
                # We slice raw_data to get only the ID channels (indices 0, 2, 4, etc.)
                id_channels = raw_data[:, :, 0::2]

                # Get all unique IDs present in the actual pixels
                visible_ids = np.unique(id_channels)

                # Convert to a set for O(1) lookup speed
                visible_ids_set = set(visible_ids.tolist())
                logger.info(f"Found {len(visible_ids_set)} visible objects out of {len(layer.manifest)} in manifest.")
                # --- OPTIMIZATION END ---

                # 3. Domain logic to get masks
                # layer.manifest is Dict[str, float]
                # sort keys for deterministic order
                sorted_names = sorted(layer.manifest.keys())

                # Upload the channel data to the GPU once when a GPU backend is used.
                compute_data = xp.asarray(raw_data) if xp is not np else raw_data

                # Combined ID map for the summary preview. It is accumulated on
                # the same backend as the masks, so the GPU does the whole
                # "combine -> encode RGB" chain and only the final image is
                # transferred back to the host.
                combined_id_map = xp.zeros(raw_data.shape[:2], dtype=xp.uint32)
                best_coverage = xp.zeros(raw_data.shape[:2], dtype=xp.uint8)
                better = xp.empty(raw_data.shape[:2], dtype=bool)
                has_masks = False

                for obj_name in sorted_names:
                    obj_id = layer.manifest[obj_name]

                    # --- FAST CHECK ---
                    # If the ID isn't in the pixel data, skip expensive computation entirely
                    if obj_id not in visible_ids_set:
                        continue
                    # ------------------

                    # compute_mask returns [H, W] uint8
                    mask = MaskCompositionService.compute_mask(obj_id, compute_data, xp=xp)

                    # Optimization: check if empty (Double check, though visible_ids_set should handle 99% of cases)
                    host_mask = to_numpy(mask, xp)
                    if host_mask.min() == host_mask.max():
                        logger.debug(f"Skipping empty mask for {obj_name}")
                        continue

                    has_masks = True

                    # Fold this mask into the combined preview (same ordering as
                    # the previous sequential implementation). The buffers are
                    # reused in place to avoid per-object allocations.
                    id_uint32 = CryptoID(obj_id).to_uint32()
                    xp.greater(mask, best_coverage, out=better)
                    combined_id_map[better] = id_uint32
                    xp.maximum(best_coverage, mask, out=best_coverage)

                    # 4. Save (encoded on a worker thread)
                    safe_name = "".join([c for c in obj_name if c.isalnum() or c in (' ', '.', '_')]).strip()
                    save_path = os.path.join(layer_folder, f"{safe_name}_mask.png")
                    logger.debug(f"Queued mask for {obj_name}")
                    submit_save(save_path, host_mask)

                # --- SUMMARY PREVIEW GENERATION ---
                if has_masks:
                    logger.info(f"Generating summary preview for layer {layer.name}...")

                    packed_preview = BitwiseColorService.encode_ids_to_rgb(combined_id_map, xp=xp)

                    preview_filename = f"{base_name}_{layer.name}_mask.png"
                    preview_path = os.path.join(output_dir, preview_filename)

                    logger.info(f"Saving packed ID preview to {preview_path}")
                    submit_save(preview_path, to_numpy(packed_preview, xp))
                else:
                    logger.warning(f"No masks found for layer {layer.name}, skipping preview.")

                # Release the per-layer backend memory before moving on.
                del compute_data, combined_id_map, best_coverage, better

        # Surface any error raised inside a worker thread.
        for future in futures:
            future.result()

        logger.info("Extraction complete.")
