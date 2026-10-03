"""
VFX IBK & ViT Masking Toolkit - Model Download & Environment Setup Script.

Downloads required Vision Transformer models (ViTMatte) using
workspace-relative paths and Griptape Model Management integration.

Usage:
  python scripts/download_models.py [options]

Options:
  --model {all,vitmatte}   Specify model to download (default: all)
  --relative               Download into workspace relative models/ folder
  --force                  Force re-download even if already downloaded
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

from ibkvit_core.griptape_model_manager import (
    REQUIRED_MODELS,
    download_model_via_griptape,
    get_workspace_root,
    is_model_downloaded,
    setup_griptape_environment,
    sync_griptape_config_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VFX_IBK_ViT.Downloader")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VFX IBK & ViT AI Models (Relative Path & Griptape Integration)")
    parser.add_argument(
        "--model",
        choices=["all", "vitmatte"],
        default="all",
        help="Model to download (default: all)",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Store models in workspace relative 'models/' directory instead of default cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already present",
    )
    args = parser.parse_args()

    setup_griptape_environment()
    sync_griptape_config_models()

    target_models = []
    for model_spec in REQUIRED_MODELS:
        if args.model == "all" or model_spec.get("family") == args.model:
            target_models.append(model_spec)

    ws_root = get_workspace_root()
    logger.info("Griptape Workspace Root: %s", ws_root)
    logger.info("Library Path (Relative): libraries/%s", LIB_ROOT.name)

    all_success = True
    for m in target_models:
        mid = m["id"]
        rel_dir = m.get("relative_dir") if args.relative else None

        ready, path = is_model_downloaded(mid)
        if ready and not args.force:
            logger.info("✓ Model '%s' is already downloaded at: %s", mid, path)
            continue

        logger.info("Downloading '%s'...", mid)
        success, out_dir = download_model_via_griptape(
            model_id=mid,
            force=args.force,
            relative_dir=rel_dir,
        )
        if success:
            logger.info("✓ Successfully downloaded '%s' -> %s", mid, out_dir)
        else:
            logger.error("✗ Failed downloading '%s'", mid)
            all_success = False

    if all_success:
        logger.info("All requested models are ready.")
        sys.exit(0)
    else:
        logger.warning("One or more models failed to download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
