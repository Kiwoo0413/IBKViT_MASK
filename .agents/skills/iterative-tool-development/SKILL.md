---
name: iterative-tool-development
description: >-
  Standardized methodology and operational workflow for developing, testing,
  host-syncing, and iteratively refining custom tools, node libraries, and domain engines
  (e.g., Griptape, Nuke, ComfyUI, VFX/AI toolkits).
  Use whenever architecting tools from scratch, developing custom node libraries like this project,
  decoupling core algorithms from host frameworks, executing synthetic TDD validation,
  managing workspace-relative AI models, or synchronizing with host desktop runtimes.
---

# Iterative Tool Development & Validation Workflow

This skill encapsulates a battle-tested engineering methodology for developing, testing,
synchronizing, and refining domain tools and custom node libraries in pair programming.
It abstracts the process into domain-agnostic, repeatable phases that ensure architectural
cleanliness, zero host coupling, robust mathematical precision, and rapid feedback loops.

---

## 🧭 The 6-Phase Engineering Lifecycle

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Clarification & Non-Destructive Guardrails                    │
│ • Disambiguate user intent before any deletion                         │
│ • Determine host environment, target outputs, and feature scope        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 2: Decoupled Dual-Layer Architecture                             │
│ • core/: 100% headless, zero host dependencies, pure algorithmic math │
│ • adapters/: Host framework wrappers with standalone fallback shims    │
│ • Dual Interface: Modular step-by-step nodes + All-in-One convenience  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 3: Synthetic Test-Driven Validation (TDD)                       │
│ • Lightweight, CPU-runnable mathematical tests with synthetic tensors │
│ • Verify numerical edge cases, clamping, and codec requirements early  │
│ • Enforce 100% test pass before host integration                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 4: Host Environment Discovery, Relative Paths & Model Sync       │
│ • Inspect host config files (AppData/xdg, manifests, venvs)           │
│ • Use workspace-relative paths for libraries and models                │
│ • Handle PEP 668 external venvs & graceful multi-tier fallbacks       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 5: Agile Feedback Refinement & Boundary Optimization             │
│ • Aggressively prune unused features to eliminate computational drag   │
│ • Hole-filling & strict value locking to eliminate visual chatter      │
│ • Target resolution upscaling (e.g. 4K UHD via Lanczos-4)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Phase 6: Continuous VCS & Documentation Sync                           │
│ • Atomic, semantic git commits (feat:, fix:, refactor:)                │
│ • Clear visual architecture diagrams & dual-course user guides         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Phase-by-Phase Execution Guidelines

### Phase 1: Clarification & Guardrails Before Action

When a user requests to "clean up files", "delete everything and start over", or overhaul a tool:
1. **Never perform blind deletion**: Irreversible data loss breaks trust. Always present candidates and receive explicit confirmation before deleting files.
2. **Clarify three pillars via interactive inquiry**:
   - **Host Runtime & Interface**: Is it a custom node library (e.g. Griptape, ComfyUI, Nuke), a web app, a desktop GUI, or a CLI?
   - **Cleanup Scope**: Are we preserving `.git`, model checkpoints, and virtual environments, or wiping code only?
   - **Core Mathematical/Algorithmic Direction**: What specific algorithms, libraries, or methodologies are desired?
3. **Preserve High-Value Artifacts**: Always keep model checkpoints (`checkpoints/`, `models/`), virtual environments (`.venv/`), research documentation (`docs/`), and VCS history (`.git`).

---

### Phase 2: Decoupled Dual-Layer Architecture

Never mix host framework classes (e.g., node base classes, UI widgets) into core algorithm code.

#### 1. Core Engine Layer (`core/`)
- Pure Python / NumPy / PyTorch / OpenCV / domain libraries.
- Headless, command-line runnable, 100% decoupled from the host application.
- Must execute in standard unit test runners within seconds.

#### 2. Host Integration Layer (`nodes/` or `adapters/`)
- Thin wrappers translating host inputs/outputs to the core engine.
- Always implement a **Compatibility Shim** (`compat.py`):
  ```python
  try:
      from host_framework.types import Parameter, Node
  except ImportError:
      # Lightweight fallback stub for CLI and test environments
      class Node: ...
      class Parameter: ...
  ```
  This allows tests and CLI runners to load node definitions outside the host's private runtime.

#### 3. The Dual-Interface Pattern
Provide two access paths for users:
- **Modular Pipeline**: Discrete, single-responsibility nodes for power users needing granular parameter tuning.
- **All-in-One Node**: A unified node connecting the full pipeline with sensible defaults for fast, single-click workflows.

---

### Phase 3: Synthetic Test-Driven Validation (TDD)

Do not rely on opening heavy desktop applications or manual clicking to verify functionality.

1. **Synthetic Mathematical Testing**:
   - Create synthetic test inputs (e.g., NumPy gradient grids, geometric shapes, synthetic noise).
   - Test numerical correctness, boundary limits, and mathematical formulas deterministically.
2. **Library Quirk Handling**:
   - Verify environment flags and initialization order (e.g., environment variables required before library import).
   - Implement resilient multi-tier fallbacks (e.g., native C-binding -> OpenCV -> ImageIO -> Lossless fallback).
3. **100% Pass Threshold**:
   - Do not proceed to host synchronization until all unit and schema tests pass cleanly.

---

### Phase 4: Host Environment Discovery, Relative Paths & Model Sync

Host applications often maintain their own configuration stores, relative workspaces, and separate model caches.

#### 1. Relative-Path First Architecture
- **Never hardcode absolute paths**: Always use workspace-relative paths (`libraries/<name>`, `models/<name>`).
- Dynamically detect workspace root by ascending from the library root directory:
  ```python
  lib_root = Path(__file__).resolve().parent.parent
  workspace_root = lib_root.parent.parent if lib_root.parent.name == "libraries" else lib_root.parent
  ```
- Register relative paths in host configs (e.g. `griptape_nodes_config.json`).

#### 2. Model Management & Cache Synchronization
- When integrating AI models (e.g., ViTMatte, SAM 2):
  1. Check workspace-relative `models/` directory first.
  2. Check Hugging Face hub cache via `scan_cache_dir()` or default snapshot directories.
  3. Trigger host native ModelManager or CLI download if missing.
  4. Write completion status files (`~/.local/share/.../model_downloads/*.json`) so the host desktop UI immediately recognizes the models as installed.

#### 3. PEP 668 & Bundled Python Package Installation
- Host runtimes (like Griptape Desktop) often bundle an isolated Python managed by `uv` or marked as externally managed.
- When installing supplementary packages (e.g. Meta SAM 2 from GitHub) into the host environment, invoke the host Python directly with `--break-system-packages`:
  ```powershell
  & "<host_python_path>" -m pip install --break-system-packages git+https://github.com/facebookresearch/sam2.git
  ```

#### 4. Resilient Fallbacks for Heavy AI Modules
- If an optional AI module (like SAM 2 or specialized CUDA kernels) is not yet installed or compiled:
  - Do NOT crash the pipeline.
  - Automatically log an informational notice and fall back to a high-speed algorithmic equivalent (e.g., Adaptive ViT/Contour tracker or Guided Filter).

---

### Phase 5: Agile Feedback Refinement & Boundary Optimization

When iterating based on user review, apply these optimization patterns:

#### 1. Aggressive Feature Pruning
- If a secondary process is irrelevant to the core output, remove it.
- Eliminating unnecessary operations saves VRAM, compute time, and simplifies UI parameter clutter.

#### 2. Topological Hole-Filling & Pure Value Locking
- **Problem: Internal Jitter/Chatter (자글거림)**:
  Internal flickers and dropouts are caused by subtle lighting/shadow fluctuations dropping core values below 1.0.
  - **Remedy**: Detect external object boundaries and fill internal holes solid:
    ```python
    contours, _ = cv2.findContours(core_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled_core, contours, -1, 1.0, thickness=-1)
    final_output[filled_core == 1.0] = 1.0  # Strict pure-white lock
    ```
- **Problem: Background Edge Residuals**:
  Screen spill or float precision errors leave faint background haze.
  - **Remedy**: Strict black cutoff:
    ```python
    final_output[final_output < black_threshold] = 0.0  # Strict pure-black lock
    ```
- **Preserve Only True Transition Zones**:
  Ensure fractional values ($0.0 < \alpha < 1.0$) only exist along true sub-pixel boundaries.

#### 3. Resolution Upscaling (e.g. 4K UHD)
- Support scaling outputs to 4K UHD (3840x2160) using high-order interpolation (`cv2.INTER_LANCZOS4`).
- Re-apply pure-white and pure-black boundary clamps after interpolation to prevent resampling overshoot.

---

### Phase 6: Continuous VCS & Documentation Sync

Maintain an unbroken record of verified progress:
1. **Atomic Commits**: Stage all relevant modifications, write clear semantic commit messages, and push to remote immediately after milestone verification.
2. **Documentation Clarity**:
   - Provide visual pipeline diagrams.
   - Structure documentation into "Quick All-in-One Course" and "Modular Pipeline Course".
   - Include relative-path host registration JSON snippets and standalone download commands.

---

## 🛠️ Verification Checklist for Any Tool Project

Before declaring a tool development cycle complete, verify:
- [ ] Core algorithms run independently of any host app imports.
- [ ] Compatibility shim allows headless node/adapter testing.
- [ ] Unit tests cover core mathematical functions with synthetic data (100% pass).
- [ ] Host configuration uses workspace-relative paths (`libraries/...`, `models/...`).
- [ ] Model weights are resolved from relative folders, local caches, or auto-downloaded via host model management.
- [ ] All node classes import successfully in the host's Python runtime.
- [ ] Extreme boundary values are strictly locked (no float noise or internal jitter).
- [ ] Git working tree is clean and changes are pushed to remote.
