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


def ensure_hf_cache_dir() -> Path:
    """Ensure Hugging Face cache directory exists so scan_cache_dir doesn't raise CacheNotFound."""
    try:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        p = Path(HUGGINGFACE_HUB_CACHE)
    except Exception:
        p = Path.home() / ".cache" / "huggingface" / "hub"
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_huggingface_auth(
    token: Optional[str] = None,
    interactive: bool = False,
    config_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Handle Hugging Face authentication:
    1. If user passed token via CLI argument (--token), use it.
    2. If --login flag requested, prompt user interactively for HF token.
    3. Check HF_TOKEN in environment variable or griptape_nodes_config.json.
    4. Call huggingface_hub.login(token=...) to authenticate.
    5. Save token in griptape_nodes_config.json (secrets_to_register.HF_TOKEN) for Griptape Desktop.
    """
    import huggingface_hub

    resolved_token = token.strip() if token else None

    # Check Griptape config for HF_TOKEN if not provided
    if not resolved_token and config_dir:
        cfg_file = config_dir / "griptape_nodes_config.json"
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                secrets = cfg.get("app_events", {}).get("on_app_initialization_complete", {}).get("secrets_to_register", {})
                cfg_token = secrets.get("HF_TOKEN", "").strip()
                if cfg_token:
                    resolved_token = cfg_token
            except Exception:
                pass

    # Check environment variable
    if not resolved_token:
        env_token = os.environ.get("HF_TOKEN", "").strip()
        if env_token:
            resolved_token = env_token

    # Interactive prompt if --login was requested
    if interactive:
        print("\n" + "=" * 62)
        print(" 🤗 Hugging Face Authentication & Token Setup")
        print("=" * 62)
        print("Hugging Face User Access Token(토큰)을 연결합니다.")
        print("토큰이 없는 경우 웹브라우저에서 무료로 생성할 수 있습니다:")
        print(" 👉 https://huggingface.co/settings/tokens (Role: Read 권장)\n")

        try:
            import getpass
            user_input = getpass.getpass("Hugging Face Token 입력 (hf_..., 엔터 시 기존/익명 유지): ").strip()
            if user_input:
                resolved_token = user_input
        except Exception:
            try:
                user_input = input("Hugging Face Token 입력 (hf_..., 엔터 시 기존/익명 유지): ").strip()
                if user_input:
                    resolved_token = user_input
            except Exception:
                pass

    # Perform login if token is available
    if resolved_token:
        try:
            huggingface_hub.login(token=resolved_token, add_to_git_credential=False)
            logger.info("✓ Hugging Face authentication successful (Token: %s...%s)", resolved_token[:4], resolved_token[-4:])

            # Save into Griptape config secrets
            if config_dir:
                cfg_file = config_dir / "griptape_nodes_config.json"
                if cfg_file.exists():
                    try:
                        with open(cfg_file, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        secrets = cfg.setdefault("app_events", {}).setdefault("on_app_initialization_complete", {}).setdefault("secrets_to_register", {})
                        secrets["HF_TOKEN"] = resolved_token
                        with open(cfg_file, "w", encoding="utf-8") as f:
                            json.dump(cfg, f, indent=2)
                        logger.info("✓ HF_TOKEN registered in Griptape Desktop secrets_to_register.")
                    except Exception as e:
                        logger.warning("Could not persist HF_TOKEN to griptape config: %s", e)
        except Exception as e:
            logger.warning("Hugging Face login attempt failed: %s", e)
    else:
        existing = huggingface_hub.get_token()
        if existing:
            logger.info("✓ Found existing cached Hugging Face token.")
        else:
            logger.info("ℹ️ Hugging Face 익명(Anonymous) 모드로 진행합니다. (토큰 등록 필요 시 --login 또는 --token <토큰> 옵션 사용)")

    return resolved_token


def register_in_griptape_config(
    model_ids: List[str],
    config_dir: Path,
    hf_token: Optional[str] = None,
) -> bool:
    """
    Register library manifest and model IDs in griptape_nodes_config.json.
    Ensures ALL library paths are stored as WORKSPACE-RELATIVE paths to prevent
    drive letter and absolute path download errors across environments.
    """
    config_file = config_dir / "griptape_nodes_config.json"
    if not config_file.exists():
        return False

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        app_events = cfg.setdefault("app_events", {}).setdefault("on_app_initialization_complete", {})

        # 1. Resolve relative path for this library's manifest
        lib_root = Path(__file__).resolve().parent.parent
        manifest_file = lib_root / "griptape_nodes_library.json"

        ws_dir_str = cfg.get("workspace_directory")
        if ws_dir_str:
            ws_dir = Path(ws_dir_str).resolve()
            try:
                rel_lib_path = manifest_file.relative_to(ws_dir).as_posix()
            except ValueError:
                rel_lib_path = f"libraries/{lib_root.name}/griptape_nodes_library.json"
        else:
            rel_lib_path = f"libraries/{lib_root.name}/griptape_nodes_library.json"

        # 2. Sanitize and register libraries_to_register (under app_events and top-level if present)
        for target_dict in [app_events, cfg]:
            if "libraries_to_register" in target_dict or target_dict is app_events:
                lib_list = target_dict.setdefault("libraries_to_register", [])
                cleaned_list: List[str] = []
                replaced = False
                for item in lib_list:
                    p_str = str(item).replace("\\", "/")
                    if "IBKViT_MASK" in p_str:
                        if not replaced:
                            cleaned_list.append(rel_lib_path)
                            replaced = True
                    else:
                        cleaned_list.append(item)
                if not replaced:
                    cleaned_list.append(rel_lib_path)
                target_dict["libraries_to_register"] = cleaned_list

        # 3. Register models_to_download
        models_list = app_events.setdefault("models_to_download", [])
        for mid in model_ids:
            if mid not in models_list:
                models_list.append(mid)

        # 4. Save HF_TOKEN if available
        if hf_token:
            secrets = app_events.setdefault("secrets_to_register", {})
            secrets["HF_TOKEN"] = hf_token

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        logger.info("Successfully registered relative library (%s) and models in: %s", rel_lib_path, config_file)
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


def download_model(
    model_info: Dict[str, str],
    force: bool = False,
    local_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Download single model via huggingface_hub snapshot_download."""
    model_id = model_info["id"]
    name = model_info["name"]
    logger.info(">>> Preparing model: %s (%s)", name, model_id)

    # Ensure cache directory exists before downloading
    ensure_hf_cache_dir()

    try:
        from huggingface_hub import snapshot_download

        kwargs = {
            "repo_id": model_id,
            "resume_download": True,
            "force_download": force,
        }
        if token:
            kwargs["token"] = token

        if local_dir:
            model_subfolder = local_dir / model_id.replace("/", "--")
            model_subfolder.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(model_subfolder)
            logger.info("Downloading to relative local directory: %s", model_subfolder)

        out_path = snapshot_download(**kwargs)
        logger.info("✓ Model '%s' successfully ready at: %s", name, out_path)
        return True, str(out_path)
    except Exception as e:
        logger.error("✗ Failed to download model '%s': %s", name, e)
        return False, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and synchronize required models for VFX Keying & ViT Masking Library"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--verify-only", action="store_true", help="Only verify model existence without downloading")
    parser.add_argument(
        "--login",
        action="store_true",
        help="Interactive Hugging Face login (enter your Hugging Face user access token)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face User Access Token (e.g. hf_...)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Download models directly into relative ./models directory inside the library repository",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Custom relative or absolute directory path to download/check models",
    )
    args = parser.parse_args()

    config_dir, data_dir = get_griptape_directories()
    if config_dir:
        logger.info("Found Griptape Nodes Desktop config at: %s", config_dir)
    if data_dir:
        logger.info("Found Griptape Nodes Desktop data dir at: %s", data_dir)

    # 1. Ensure HF cache directory exists (prevents CacheNotFound for first-time HF users)
    ensure_hf_cache_dir()

    # 2. Setup Hugging Face authentication if requested or available
    hf_token = setup_huggingface_auth(
        token=args.token,
        interactive=args.login,
        config_dir=config_dir,
    )

    # Determine local models directory if requested
    target_local_dir: Optional[Path] = None
    lib_root = Path(__file__).resolve().parent.parent
    if args.local:
        target_local_dir = lib_root / "models"
    elif args.models_dir:
        p_arg = Path(args.models_dir)
        target_local_dir = p_arg if p_arg.is_absolute() else (lib_root / p_arg).resolve()

    if target_local_dir:
        logger.info("Using relative/local model target: %s", target_local_dir)

    all_success = True
    downloaded_ids: List[str] = []

    for item in REQUIRED_MODELS:
        mid = item["id"]
        if args.verify_only:
            # Check local relative models directory first
            found_locally = False
            for check_dir in [
                lib_root / "models" / mid.replace("/", "--"),
                lib_root / "models" / mid.split("/")[-1],
                lib_root.parent.parent / "models" / mid.replace("/", "--"),
                target_local_dir / mid.replace("/", "--") if target_local_dir else None,
            ]:
                if check_dir and check_dir.exists() and any(check_dir.iterdir()):
                    logger.info("✓ Verified: %s is present in local relative directory: %s", mid, check_dir)
                    found_locally = True
                    break

            if found_locally:
                continue

            try:
                # Check HF cache
                ensure_hf_cache_dir()
                from huggingface_hub import scan_cache_dir

                cache_info = scan_cache_dir()
                cached_repos = [r.repo_id for r in cache_info.repos]
                if mid in cached_repos:
                    logger.info("✓ Verified: %s is present in Hugging Face cache", mid)
                else:
                    logger.warning("! Missing: %s is NOT in Hugging Face cache or local models dir", mid)
                    all_success = False
            except Exception as ex:
                logger.warning("! Missing: %s not cached yet (cache scan note: %s)", mid, ex)
                all_success = False
            continue

        success, local_path_str = download_model(
            item,
            force=args.force,
            local_dir=target_local_dir,
            token=hf_token,
        )
        if success and local_path_str:
            downloaded_ids.append(mid)
            if data_dir:
                register_griptape_status(mid, Path(local_path_str), data_dir)
        else:
            all_success = False

    # Sync Griptape config with relative paths and HF token
    if config_dir:
        ids_to_sync = downloaded_ids if downloaded_ids else [m["id"] for m in REQUIRED_MODELS]
        register_in_griptape_config(ids_to_sync, config_dir, hf_token=hf_token)

    if all_success:
        logger.info("🎉 All required models for VFX IBK & ViT Masking are ready and synchronized with Griptape!")
        return 0
    else:
        logger.error("⚠️ Some models could not be verified or downloaded.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
