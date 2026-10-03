"""
Griptape Model Management Integration for VFX IBK & ViT Masking Toolkit.

Handles relative-path workspace registration, model catalog synchronization,
and automatic model downloading for:
  - hustvl/vitmatte-small-composition-1k (ViTMatte boundary matting)
  - facebook/sam2.1-hiera-large (SAM 2.1 Spatio-temporal video object tracking)
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("VFX_IBK_ViT.ModelManager")

# Required models for VFX IBK & ViT Masking Toolkit
REQUIRED_MODELS: List[Dict[str, Any]] = [
    {
        "id": "hustvl/vitmatte-small-composition-1k",
        "name": "ViTMatte Small (Composition-1k)",
        "family": "vitmatte",
        "key_support": "NO_KEY_REQUIRED",
        "relative_dir": "models/vitmatte-small-composition-1k",
        "essential_files": ["config.json", "model.safetensors"],
    },
]


def get_library_root() -> Path:
    """Return the absolute path of this library root (IBKViT_MASK)."""
    return Path(__file__).resolve().parent.parent


def get_workspace_root() -> Path:
    """
    Detect the Griptape workspace root directory using relative path traversal.
    Typically: <workspace_root>/libraries/IBKViT_MASK
    """
    lib_root = get_library_root()
    if lib_root.parent.name == "libraries":
        return lib_root.parent.parent
    return lib_root.parent


def get_relative_library_manifest_path() -> str:
    """
    Return the relative path of this library manifest from the workspace root.
    Example: 'libraries/IBKViT_MASK/griptape_nodes_library.json'
    """
    lib_root = get_library_root()
    ws_root = get_workspace_root()
    try:
        rel = (lib_root / "griptape_nodes_library.json").relative_to(ws_root)
        return str(rel).replace("\\", "/")
    except ValueError:
        return f"libraries/{lib_root.name}/griptape_nodes_library.json"


def get_griptape_config_path() -> Optional[Path]:
    """Find the Griptape Nodes desktop configuration json file."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        cfg = Path(appdata) / "Griptape Nodes" / "xdg_config_home" / "griptape_nodes" / "griptape_nodes_config.json"
        if cfg.exists():
            return cfg

    # Fallback to XDG config
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        cfg = Path(xdg_config) / "griptape_nodes" / "griptape_nodes_config.json"
        if cfg.exists():
            return cfg

    return None


def get_griptape_python_executable() -> Optional[str]:
    """
    Locate Griptape Desktop's bundled python executable.
    Example: %LOCALAPPDATA%\\ai.griptape.nodes.desktop\\current\\resources\\engine-bundle\\python\\python.exe
    """
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        p = Path(localappdata) / "ai.griptape.nodes.desktop" / "current" / "resources" / "engine-bundle" / "python" / "python.exe"
        if p.exists():
            return str(p)

    # Check if sys.executable is already Griptape bundled python
    if "engine-bundle" in sys.executable:
        return sys.executable

    return None


def setup_griptape_environment() -> None:
    """Ensure Hugging Face cache and Griptape environment variables are configured."""
    user_home = Path.home()
    default_hf_home = user_home / ".cache" / "huggingface"
    default_hf_hub = default_hf_home / "hub"

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(default_hf_home)
    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = str(default_hf_hub)

    default_hf_hub.mkdir(parents=True, exist_ok=True)


def sync_griptape_config_models() -> bool:
    """
    Synchronize Griptape Nodes configuration with relative paths:
    1. Register 'libraries/IBKViT_MASK/griptape_nodes_library.json' in 'libraries_to_register'
    2. Register required models in 'models_to_download'
    """
    cfg_path = get_griptape_config_path()
    if not cfg_path:
        return False

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        app_events = data.setdefault("app_events", {})
        init_complete = app_events.setdefault("on_app_initialization_complete", {})

        changed = False

        # 1. Register library using relative path
        rel_lib = get_relative_library_manifest_path()
        libs_list = init_complete.setdefault("libraries_to_register", [])
        # Remove absolute path duplicates if present
        cleaned_libs = []
        for item in libs_list:
            norm = item.replace("\\", "/")
            if "IBKViT_MASK" in norm and norm != rel_lib:
                changed = True
                continue
            cleaned_libs.append(item)
        if rel_lib not in cleaned_libs:
            cleaned_libs.append(rel_lib)
            changed = True
        init_complete["libraries_to_register"] = cleaned_libs

        # 2. Register models to download
        models_list = init_complete.setdefault("models_to_download", [])
        for model in REQUIRED_MODELS:
            mid = model["id"]
            if mid not in models_list:
                models_list.append(mid)
                changed = True

        if changed:
            temp_path = cfg_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(cfg_path)
            logger.info("[Griptape Model Manager] Successfully synchronized relative paths in config: %s", cfg_path)

        return True
    except Exception as e:
        logger.debug("[Griptape Model Manager] Config sync skipped: %s", e)
        return False


def mark_model_download_completed(model_id: str) -> None:
    """Write Griptape ModelManager status file so Desktop UI immediately recognizes completion."""
    try:
        user_home = Path.home()
        status_dir = user_home / ".local" / "share" / "griptape_nodes" / "model_downloads"
        status_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_id.replace("/", "--").replace(".", "--") + ".json"
        status_file = status_dir / safe_name
        now_iso = datetime.now(timezone.utc).isoformat()
        status_data = {
            "model_id": model_id,
            "status": "completed",
            "started_at": now_iso,
            "updated_at": now_iso,
            "completed_at": now_iso,
            "progress_percent": 100.0,
            "completed": True,
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        logger.debug("Failed marking model download completed: %s", e)


def is_model_downloaded(model_id: str) -> Tuple[bool, Optional[Path]]:
    """
    Check if a model is downloaded and ready in:
    1. Workspace/Library relative 'models/' directory
    2. Hugging Face hub cache via scan_cache_dir or directory inspection
    """
    lib_root = get_library_root()
    ws_root = get_workspace_root()

    subfolder_names = [
        model_id.replace("/", "--"),
        model_id.split("/")[-1],
        model_id,
    ]

    # Search in relative models folders:
    # - <workspace>/models/
    # - <library>/models/
    # - ./models/
    candidate_roots = [
        ws_root / "models",
        lib_root / "models",
        Path("models"),
    ]

    for parent_dir in candidate_roots:
        if not parent_dir.exists():
            continue
        for sub in subfolder_names:
            candidate = parent_dir / sub
            if candidate.exists() and any(candidate.iterdir()):
                mark_model_download_completed(model_id)
                return True, candidate

    # Search Hugging Face cache
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
                            mark_model_download_completed(model_id)
                            return True, p_snap
    except Exception:
        pass

    # Direct check of default Hugging Face hub directory
    user_home = Path.home()
    direct_hub_sub = user_home / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if direct_hub_sub.exists():
        for snap in direct_hub_sub.iterdir():
            if snap.is_dir() and any(snap.iterdir()):
                mark_model_download_completed(model_id)
                return True, snap

    return False, None


def download_model_via_griptape(
    model_id: str,
    force: bool = False,
    relative_dir: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Download model using relative paths & Griptape's native Model Management system.
    1. Check if already downloaded.
    2. Try direct ModelManager invocation if in Griptape bundled environment.
    3. Try Griptape CLI (`griptape_nodes.cli.commands.models download`).
    4. Fall back to huggingface_hub.snapshot_download.
    """
    sync_griptape_config_models()
    setup_griptape_environment()

    if not force:
        ready, path = is_model_downloaded(model_id)
        if ready and path:
            logger.info("[Griptape Model Manager] Model '%s' is ready at: %s", model_id, path)
            mark_model_download_completed(model_id)
            return True, str(path)

    # Determine relative download directory if requested
    local_dir_path: Optional[Path] = None
    if relative_dir:
        ws_root = get_workspace_root()
        local_dir_path = (ws_root / relative_dir).resolve()
        local_dir_path.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Native Griptape ModelManager if available
    try:
        from griptape_nodes.retained_mode.retained_mode import GriptapeNodes
        mm = GriptapeNodes.ModelManager()
        logger.info("[Griptape Model Manager] Initiating native download for: %s", model_id)
        target_str = str(local_dir_path) if local_dir_path else None
        downloaded_path = mm.download_model(model_id=model_id, local_dir=target_str)

        mark_model_download_completed(model_id)
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
            if local_dir_path:
                cmd.extend(["--local-dir", str(local_dir_path)])
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                ready, path = is_model_downloaded(model_id)
                if ready and path:
                    mark_model_download_completed(model_id)
                    logger.info("[Griptape Model Manager] ✓ Griptape CLI successfully downloaded '%s' to: %s", model_id, path)
                    return True, str(path)
            else:
                logger.warning("[Griptape Model Manager] Griptape CLI exited with %d: %s", res.returncode, res.stderr)
        except Exception as ex_cli:
            logger.warning("[Griptape Model Manager] Griptape CLI invocation failed: %s", ex_cli)

    # Strategy 3: Fallback snapshot_download via huggingface_hub
    try:
        from huggingface_hub import snapshot_download

        kwargs: Dict[str, Any] = {
            "repo_id": model_id,
            "resume_download": True,
            "force_download": force,
        }
        if local_dir_path:
            kwargs["local_dir"] = str(local_dir_path)

        out_path = snapshot_download(**kwargs)
        mark_model_download_completed(model_id)
        logger.info("[Griptape Model Manager] ✓ Downloaded '%s' via fallback snapshot_download: %s", model_id, out_path)
        return True, str(out_path)
    except Exception as e:
        logger.error("[Griptape Model Manager] ✗ All download methods failed for '%s': %s", model_id, e)
        return False, None


def ensure_all_models_ready(auto_download: bool = True) -> bool:
    """
    Ensure all required Vision Transformer models (SAM 2 & ViTMatte) are ready.
    """
    sync_griptape_config_models()

    all_ready = True
    for item in REQUIRED_MODELS:
        mid = item["id"]
        ready, path = is_model_downloaded(mid)
        if not ready:
            if auto_download:
                logger.info("[Griptape Model Manager] Model '%s' not found locally. Auto-downloading via Griptape Model Management...", mid)
                success, _ = download_model_via_griptape(mid, relative_dir=item.get("relative_dir"))
                if not success:
                    all_ready = False
            else:
                all_ready = False

    return all_ready


def resolve_model_weights_and_config(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve model weight checkpoint and config paths for ViTMatte or SAM 2.
    Returns:
        (checkpoint_path_or_dir, config_path_or_none)
    """
    ready, model_dir = is_model_downloaded(model_id)
    if not ready or not model_dir:
        logger.info("[Griptape Model Manager] Model '%s' not present. Triggering Griptape auto-download...", model_id)
        # Find relative dir preference
        rel_dir = None
        for item in REQUIRED_MODELS:
            if item["id"] == model_id:
                rel_dir = item.get("relative_dir")
                break
        success, dl_dir_str = download_model_via_griptape(model_id, relative_dir=rel_dir)
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
            model_dir / "sam2.1_hiera_l.pt",
        ]
        yaml_candidates = [
            model_dir / "sam2.1_hiera_l.yaml",
            model_dir / "sam2_hiera_l.yaml",
            model_dir / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
        ]
        pt_path = next((str(p) for p in pt_candidates if p.exists()), None)
        yaml_path = next((str(y) for y in yaml_candidates if y.exists()), None)

        # If yaml is not directly in the snapshot, fallback to sam2 built-in config name
        if yaml_path is None:
            yaml_path = "configs/sam2.1/sam2.1_hiera_l.yaml"

        return pt_path, yaml_path

    # For ViTMatte: model_dir contains config.json, preprocessor_config.json, model.safetensors
    return str(model_dir), None
