import argparse
import sys
import logging
from kriptomatte.infrastructure.logging.logger import setup_logger
from kriptomatte.infrastructure.persistence.exr_repository import OpenExrRepository
from kriptomatte.application.services import CryptomatteExtractionService

def get_args():
    parser = argparse.ArgumentParser(description='Decode Cryptomattes in EXR file to PNG files (DDD Refactored).')
    parser.add_argument('--input', '-i', dest='input_path', type=str, required=True,
                        help='Provide path of exr file')
    parser.add_argument('--output', '-o', dest='output_path', type=str, default=None,
                        help='Output directory (default: directory of the input file)')
    parser.add_argument('--workers', '-w', dest='workers', type=int, default=None,
                        help='Number of worker threads (default: CPU count)')
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='Force GPU (CuPy) acceleration (--gpu) or disable it (--no-gpu)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable DEBUG logging')
    return parser.parse_args()


def main():
    args = get_args()

    # Setup Infrastructure
    logger = setup_logger(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug(f"CLI args: {args}")

    repo = OpenExrRepository()
    service = CryptomatteExtractionService(repo, max_workers=args.workers, use_gpu=args.gpu)

    try:
        service.extract_all(args.input_path, args.output_path)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
