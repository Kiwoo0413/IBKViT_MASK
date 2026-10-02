"""
core/griptape_model_manager.py
Unified Griptape Model Management Bridge for VFX IBK & ViT Masking.

Provides automatic model synchronization, verification, and downloading directly
through Griptape Nodes Desktop's native Model Management system (ModelManager / CLI).
Enables seamless model onboarding in fresh local environments without manual intervention.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GriptapeModelManager")

# Required Vision Transformer models for VFX Keying & ViT Masking
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


def setup_griptape_environment() -> None:
    """
    Ensure XDG environment variables are aligned with Griptape Nodes Desktop.
    """
    localappdata = os.environ.get("LOCALAPPDATA")
    appdata = os.environ.get("APPDATA")

    # Set XDG data and config home to Griptape defaults if not already set
    if "XDG_DATA_HOME" not in os.environ and localappdata:
        os.environ["XDG_DATA_HOME"] = str(Path(localappdata) / "Griptape Nodes" / "xdg_data_home")
    if "XDG_CONFIG_HOME" not in os.environ and appdata:
        os.environ["XDG_CONFIG_HOME"] = str(Path(appdata) / "Griptape Nodes" / "xdg_config_home")


def get_griptape_directories() -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve Griptape Nodes Desktop config and data directories."""
    appdata = os.environ.get("APPDATA")
    localappdata = os.environ.get("LOCALAPPDATA")

    config_dir: Optional[Path] = None
    data_dir: Optional[Path] = None

    if os.environ.get("XDG_CONFIG_HOME"):
        cfg = Path(os.environ["XDG_CONFIG_HOME"]) / "griptape_nodes"
        if cfg.exists():
            config_dir = cfg
    elif appdata:
        cfg = Path(appdata) / "Griptape Nodes" / "xdg_config_home" / "griptape_nodes"
        if cfg.exists():
            config_dir = cfg

    if os.environ.get("XDG_DATA_HOME"):
        dat = Path(os.environ["XDG_DATA_HOME"]) / "griptape_nodes"
        if dat.exists():
            data_dir = dat
    elif localappdata:
        dat = Path(localappdata) / "Griptape Nodes" / "xdg_data_home" / "griptape_nodes"
        if dat.exists():
            data_dir = dat

    return config_dir, data_dir


def get_griptape_python_executable() -> Optional[str]:
    """
    Locate Griptape Nodes Desktop's bundled Python executable across platforms.
    """
    # 1. If currently executing within Griptape's bundled python
    if "ai.griptape.nodes.desktop" in sys.executable or "engine-bundle" in sys.executable:
        return sys.executable

    # 2. Check Windows install paths
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        candidates = [
            Path(localappdata) / "ai.griptape.nodes.desktop" / "current" / "resources" / "engine-bundle" / "python" / "python.exe",
            Path(localappdata) / "Programs" / "Griptape Nodes" / "resources" / "engine-bundle" / "python" / "python.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

    # 3. Check macOS install paths
    home = Path.home()
    mac_candidates = [
        Path("/Applications/Griptape Nodes.app/Contents/Resources/engine-bundle/python/bin/python3"),
        home / "Library" / "Application Support" / "Griptape Nodes" / "engine-bundle" / "python" / "bin" / "python3",
    ]
    for c in mac_candidates:
        if c.exists():
            return str(c)

    # 4. Check Linux install paths
    linux_candidates = [
        home / ".local" / "share" / "griptape_nodes" / "engine-bundle" / "python" / "bin" / "python3",
        Path("/opt/griptape-nodes/resources/engine-bundle/python/bin/python3"),
    ]
    for c in linux_candidates:
        if c.exists():
            return str(c)

    # 5. Check if griptape_nodes is available in current interpreter
    try:
        import griptape_nodes  # noqa: F401
        return sys.executable
    except ImportError:
        pass

    return None


def sync_griptape_config_models() -> bool:
    """
    Automatically register required models into Griptape's `models_to_download` config
    and register this library's manifest using a workspace-relative path.
    Ensures that Griptape Desktop automatically downloads and manages the models.
    """
    config_dir, _ = get_griptape_directories()
    if not config_dir:
        return False

    config_file = config_dir / "griptape_nodes_config.json"
    if not config_file.exists():
        return False

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        app_events = cfg.setdefault("app_events", {}).setdefault("on_app_initialization_complete", {})

        # 1. Register models_to_download
        models_list = app_events.setdefault("models_to_download", [])
        changed = False
        for item in REQUIRED_MODELS:
            mid = item["id"]
            if mid not in models_list:
                models_list.append(mid)
                changed = True

        # 2. Register library as workspace-relative path
        lib_root = Path(__file__).resolve().parent.parent
        manifest_file = lib_root / "griptape_nodes_library.json"
        ws_dir_str = cfg.get("workspace_directory")
        if ws_dir_str:
            try:
                rel_lib_path = manifest_file.relative_to(Path(ws_dir_str).resolve()).as_posix()
            except ValueError:
                rel_lib_path = f"libraries/{lib_root.name}/griptape_nodes_library.json"
        else:
            rel_lib_path = f"libraries/{lib_root.name}/griptape_nodes_library.json"

        for target in [app_events, cfg]:
            if "libraries_to_register" in target or target is app_events:
                lib_list = target.setdefault("libraries_to_register", [])
                cleaned_list: List[str] = []
                replaced = False
                for entry in lib_list:
                    p_str = str(entry).replace("\\", "/")
                    if "IBKViT_MASK" in p_str:
                        if not replaced:
                            cleaned_list.append(rel_lib_path)
                            replaced = True
                    else:
                        cleaned_list.append(entry)
                if not replaced:
                    cleaned_list.append(rel_lib_path)
                    changed = True
                if cleaned_list != lib_list:
                    target["libraries_to_register"] = cleaned_list
                    changed = True

        if changed:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            logger.info("[Griptape Model Manager] Auto-synced models_to_download in Griptape config: %s", config_file)
        return True
    except Exception as e:
        logger.debug("[Griptape Model Manager] Config sync skipped: %s", e)
        return False


def is_model_downloaded(model_id: str) -> Tuple[bool, Optional[Path]]:
    """
    Check if a model is downloaded and ready in local relative folder,
    Griptape model cache, or Hugging Face cache.
    """
    lib_root = Path(__file__).resolve().parent.parent
    workspace_root = lib_root.parent.parent

    # 1. Check local relative models/ directory inside library or workspace
    subfolder_names = [
        model_id.replace("/", "--"),
        model_id.split("/")[-1],
    ]
    for parent_dir in [lib_root / "models", workspace_root / "models", Path("models")]:
        for sub in subfolder_names:
            candidate = parent_dir / sub
            if candidate.exists() and any(candidate.iterdir()):
                return True, candidate

    # 2. Check Hugging Face hub cache via scan_cache_dir
    try:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        hf_cache = Path(HUGGINGFACE_HUB_CACHE)
        if hf_cache.exists():
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir()
            for repo in info.repos:
                if repo.repo_id == model_id:
                    for rev in repo.revisions:
                        p_snap = Path(rev.snapshot_path)
                        if p_snap.exists() and any(p_snap.iterdir()):
                            return True, p_snap
    except Exception:
        pass

    return False, None


def download_model_via_griptape(
    model_id: str,
    force: bool = False,
    local_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Download model using Griptape's native Model Management system.
    1. Check if already downloaded.
    2. Try direct ModelManager invocation if in Griptape bundled environment.
    3. Try Griptape CLI (`griptape_nodes.cli.commands.models download`).
    4. Fall back to huggingface_hub.snapshot_download if Griptape is unavailable.
    """
    # Ensure Griptape config registers this model
    sync_griptape_config_models()
    setup_griptape_environment()

    # If already downloaded and not forcing, return immediately
    if not force:
        ready, path = is_model_downloaded(model_id)
        if ready and path:
            logger.info("[Griptape Model Manager] Model '%s' is already ready at: %s", model_id, path)
            return True, str(path)

    # Strategy 1: Native Griptape ModelManager if importable without dependency conflicts
    try:
        from griptape_nodes.retained_mode.retained_mode import GriptapeNodes
        mm = GriptapeNodes.ModelManager()
        logger.info("[Griptape Model Manager] Initiating native download for: %s", model_id)
        target_str = str(local_dir) if local_dir else None
        downloaded_path = mm.download_model(model_id=model_id, local_dir=target_str)

        # Mark status in Griptape's data directory so UI Model Manager immediately recognizes it
        try:
            status_file = mm._get_status_file_path(model_id)
            now_iso = datetime.now(timezone.utc).isoformat()
            final_data = {
                "model_id": model_id,
                "status": "completed",
                "started_at": now_iso,
                "updated_at": now_iso,
                "completed_at": now_iso,
                "progress_percent": 100.0,
                "completed": True,
            }
            mm._write_download_status(status_file, final_data)
        except Exception:
            pass

        logger.info("[Griptape Model Manager] ✓ Successfully downloaded '%s' to: %s", model_id, downloaded_path)
        return True, str(downloaded_path)
    except Exception as ex_direct:
        logger.debug("[Griptape Model Manager] Direct ModelManager skipped (%s). Trying Griptape CLI...", ex_direct)

    # Strategy 2: Griptape CLI using bundled python executable
    gt_py = get_griptape_python_executable()
    if gt_py:
        try:
            logger.info("[Griptape Model Manager] Triggering Griptape CLI download for '%s'...", model_id)
            cmd = [gt_py, "-m", "griptape_nodes.cli.commands.models", "download", model_id]
            if local_dir:
                cmd.extend(["--local-dir", str(local_dir)])
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                ready, path = is_model_downloaded(model_id)
                if ready and path:
                    logger.info("[Griptape Model Manager] ✓ Griptape CLI successfully downloaded '%s' to: %s", model_id, path)
                    return True, str(path)
            else:
                logger.warning("[Griptape Model Manager] Griptape CLI exited with %d: %s", res.returncode, res.stderr)
        except Exception as ex_cli:
            logger.warning("[Griptape Model Manager] Griptape CLI invocation failed: %s", ex_cli)

    # Strategy 3: Fallback snapshot_download via huggingface_hub
    try:
        from huggingface_hub import snapshot_download

        kwargs = {
            "repo_id": model_id,
            "resume_download": True,
            "force_download": force,
        }
        if local_dir:
            model_subfolder = local_dir / model_id.replace("/", "--")
            model_subfolder.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(model_subfolder)

        out_path = snapshot_download(**kwargs)
        logger.info("[Griptape Model Manager] ✓ Downloaded '%s' via fallback snapshot_download: %s", model_id, out_path)
        return True, str(out_path)
    except Exception as e:
        logger.error("[Griptape Model Manager] ✗ All download methods failed for '%s': %s", model_id, e)
        return False, None


def ensure_all_models_ready(auto_download: bool = True) -> bool:
    """
    Ensure all required Vision Transformer models are downloaded and ready
    for VFX Keying & ViT Masking. Automatically downloads missing models via
    Griptape's Model Management.
    """
    sync_griptape_config_models()

    all_ready = True
    for item in REQUIRED_MODELS:
        mid = item["id"]
        ready, path = is_model_downloaded(mid)
        if not ready:
            if auto_download:
                logger.info("[Griptape Model Manager] Model '%s' not found locally. Auto-downloading via Griptape Model Management...", mid)
                success, _ = download_model_via_griptape(mid)
                if not success:
                    all_ready = False
            else:
                all_ready = False

    return all_ready


def resolve_model_weights_and_config(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve model weight checkpoint and config paths for ViTMatte or SAM 2.
    If the model is not found, automatically invokes Griptape's Model Management to download it.

    Returns:
        (checkpoint_path, config_path)
    """
    ready, model_dir = is_model_downloaded(model_id)
    if not ready or not model_dir:
        logger.info("[Griptape Model Manager] Model '%s' not present. Triggering Griptape auto-download...", model_id)
        success, dl_dir_str = download_model_via_griptape(model_id)
        if success and dl_dir_str:
            model_dir = Path(dl_dir_str)
        else:
            ready, model_dir = is_model_downloaded(model_id)

    if not model_dir or not model_dir.exists():
        return None, None

    # For SAM 2: check for .pt checkpoint and .yaml config
    if "sam2" in model_id.lower():
        pt_candidates = [
            model_dir / "sam2.1_hiera_large.pt",
            model_dir / "sam2_hiera_large.pt",
        ]
        yaml_candidates = [
            model_dir / "sam2.1_hiera_l.yaml",
            model_dir / "sam2_hiera_l.yaml",
        ]
        pt_path = next((str(p) for p in pt_candidates if p.exists()), None)
        yaml_path = next((str(y) for y in yaml_candidates if y.exists()), None)
        return pt_path, yaml_path

    # For ViTMatte: model_dir is directory containing config.json and weights
    return str(model_dir), None
