"""
VFX IBK & ViT Masking Toolkit - Model Download & Environment Setup Script.

Downloads required Vision Transformer models (ViTMatte & SAM 2.1) using
workspace-relative paths and Griptape Model Management integration.

Usage:
  python scripts/download_models.py [options]

Options:
  --model {all,vitmatte,sam2}   Specify model to download (default: all)
  --relative                     Download into workspace relative models/ folder
  --install-sam2                 Install SAM 2 package into Griptape's Python environment
  --force                        Force re-download even if already downloaded
"""

import argparse
import logging
import os
import subprocess
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
    get_griptape_python_executable,
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


def install_sam2_into_griptape() -> bool:
    """Install Meta's SAM 2 package into Griptape's bundled python environment."""
    gt_py = get_griptape_python_executable()
    if not gt_py:
        logger.warning("Griptape bundled Python executable not found. Skipping Griptape sam2 installation.")
        return False

    logger.info("Installing SAM 2 into Griptape Python: %s", gt_py)
    cmd = [
        gt_py,
        "-m",
        "pip",
        "install",
        "--break-system-packages",
        "git+https://github.com/facebookresearch/sam2.git",
    ]
    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0:
            logger.info("✓ SAM 2 successfully installed in Griptape Python environment.")
            return True
        else:
            logger.warning("SAM 2 installation exited with return code: %d", res.returncode)
            return False
    except Exception as e:
        logger.error("Failed installing SAM 2 into Griptape Python: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VFX IBK & ViT AI Models (Relative Path & Griptape Integration)")
    parser.add_argument(
        "--model",
        choices=["all", "vitmatte", "sam2"],
        default="all",
        help="Model to download (default: all)",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Store models in workspace relative 'models/' directory instead of default cache",
    )
    parser.add_argument(
        "--install-sam2",
        action="store_true",
        help="Install SAM 2 package into Griptape's bundled Python environment",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already present",
    )
    args = parser.parse_args()

    setup_griptape_environment()
    sync_griptape_config_models()

    if args.install_sam2:
        install_sam2_into_griptape()

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
