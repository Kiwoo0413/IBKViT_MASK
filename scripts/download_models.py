"""
scripts/download_models.py
Automated Model Downloader & Griptape Nodes Desktop Model Synchronizer.

Downloads required Vision Transformer models for VFX Keying & ViT Masking:
1. hustvl/vitmatte-small-composition-1k (Sub-pixel Alpha Matting ViT)
2. facebook/sam2.1-hiera-large (Video Object Tracking & Segmentation ViT)

Also registers download status directly into Griptape Nodes Desktop's Model Manager
and config (griptape_nodes_config.json) so the models are immediately recognized.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelDownloader")

REQUIRED_MODELS: List[Dict[str, str]] = [
    {
        "id": "hustvl/vitmatte-small-composition-1k",
        "name": "ViTMatte Small (Composition-1k)",
        "desc": "Sub-pixel high precision alpha matting Vision Transformer",
        "family": "ViTMatte",
    },
    {
        "id": "facebook/sam2.1-hiera-large",
        "name": "SAM 2.1 Hiera Large",
        "desc": "Spatio-temporal video object segmentation & tracking",
        "family": "SAM 2",
    },
]


def sanitize_model_id_for_griptape(model_id: str) -> str:
    """Format model ID into Griptape's status JSON filename convention."""
    return re.sub(r"[^\w\-_]", "--", model_id) + ".json"


def get_griptape_directories() -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve Griptape Nodes Desktop config and data directories if installed."""
    appdata = os.environ.get("APPDATA")
    localappdata = os.environ.get("LOCALAPPDATA")

    config_dir: Optional[Path] = None
    data_dir: Optional[Path] = None

    if appdata:
        cfg = Path(appdata) / "Griptape Nodes" / "xdg_config_home" / "griptape_nodes"
        if cfg.exists():
            config_dir = cfg

    if localappdata:
        dat = Path(localappdata) / "Griptape Nodes" / "xdg_data_home" / "griptape_nodes"
        if dat.exists():
            data_dir = dat

    return config_dir, data_dir


def register_models_in_griptape_config(model_ids: List[str], config_dir: Path) -> bool:
    """Add model IDs to griptape_nodes_config.json under models_to_download."""
    config_file = config_dir / "griptape_nodes_config.json"
    if not config_file.exists():
        return False

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        app_events = cfg.setdefault("app_events", {}).setdefault("on_app_initialization_complete", {})
        models_list = app_events.setdefault("models_to_download", [])

        updated = False
        for mid in model_ids:
            if mid not in models_list:
                models_list.append(mid)
                updated = True

        if updated:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            logger.info("Successfully registered models in Griptape config: %s", config_file)
        return True
    except Exception as e:
        logger.warning("Could not update griptape_nodes_config.json: %s", e)
        return False


def register_griptape_status(model_id: str, local_path: Path, data_dir: Path) -> bool:
    """Create Griptape model_downloads status file for UI recognition."""
    status_dir = data_dir / "model_downloads"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / sanitize_model_id_for_griptape(model_id)

    total_bytes = 0
    if local_path.exists():
        for p in local_path.rglob("*"):
            if p.is_file():
                total_bytes += p.stat().st_size

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status_data = {
        "model_id": model_id,
        "status": "completed",
        "started_at": now_iso,
        "updated_at": now_iso,
        "total_bytes": total_bytes,
        "downloaded_bytes": total_bytes,
        "progress_percent": 100.0,
        "completed": True,
        "completed_at": now_iso,
    }

    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
        logger.info("Griptape model download status saved: %s", status_file.name)
        return True
    except Exception as e:
        logger.warning("Could not write Griptape status file %s: %s", status_file, e)
        return False


def download_model(model_info: Dict[str, str], force: bool = False) -> Tuple[bool, Optional[str]]:
    """Download single model via huggingface_hub snapshot_download."""
    model_id = model_info["id"]
    name = model_info["name"]
    logger.info(">>> Preparing model: %s (%s)", name, model_id)

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=model_id,
            resume_download=True,
            force_download=force,
        )
        logger.info("✓ Model '%s' successfully ready at: %s", name, local_dir)
        return True, local_dir
    except Exception as e:
        logger.error("✗ Failed to download model '%s': %s", name, e)
        return False, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and synchronize required models for VFX Keying & ViT Masking Library"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--verify-only", action="store_true", help="Only verify model existence without downloading")
    args = parser.parse_args()

    config_dir, data_dir = get_griptape_directories()
    if config_dir:
        logger.info("Found Griptape Nodes Desktop config at: %s", config_dir)
    if data_dir:
        logger.info("Found Griptape Nodes Desktop data dir at: %s", data_dir)

    all_success = True
    downloaded_ids: List[str] = []

    for item in REQUIRED_MODELS:
        mid = item["id"]
        if args.verify_only:
            try:
                from huggingface_hub import try_to_load_from_cache
                # Check repo existence in cache
                from huggingface_hub import scan_cache_dir
                cache_info = scan_cache_dir()
                cached_repos = [r.repo_id for r in cache_info.repos]
                if mid in cached_repos:
                    logger.info("✓ Verified: %s is present in Hugging Face cache", mid)
                else:
                    logger.warning("! Missing: %s is NOT in Hugging Face cache", mid)
                    all_success = False
            except Exception as ex:
                logger.warning("Verification error for %s: %s", mid, ex)
                all_success = False
            continue

        success, local_dir = download_model(item, force=args.force)
        if success and local_dir:
            downloaded_ids.append(mid)
            if data_dir:
                register_griptape_status(mid, Path(local_dir), data_dir)
        else:
            all_success = False

    # Sync Griptape config
    if config_dir and downloaded_ids:
        register_models_in_griptape_config(downloaded_ids, config_dir)

    if all_success:
        logger.info("🎉 All required models for VFX IBK & ViT Masking are ready and synchronized with Griptape!")
        return 0
    else:
        logger.error("⚠️ Some models could not be verified or downloaded.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
