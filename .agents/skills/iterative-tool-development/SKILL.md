---
name: iterative-tool-development
description: >-
  Standardized methodology and operational workflow for developing, testing,
  host-syncing, and iteratively refining custom tools, node libraries, and domain engines.
  Use when architecting tools from scratch, decoupling core algorithms from host frameworks,
  executing synthetic TDD validation, synchronizing with host desktop runtimes, or tuning
  numerical boundary precision based on user feedback.
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
│ Phase 4: Host Environment Discovery & Safe Synchronization             │
│ • Inspect host config files (AppData/xdg, manifests, venvs)           │
│ • Sync code to host library paths while strictly preserving assets     │
│ • Verify tests and imports directly inside the host's virtualenv       │
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

When a user requests to "delete everything and start over" or overhaul a tool:
1. **Never perform blind deletion**: Irreversible data loss breaks trust.
2. **Clarify three pillars via interactive inquiry**:
   - **Host Runtime & Interface**: Is it a custom node library (e.g. Griptape, ComfyUI, Nuke), a web app, a desktop GUI, or a CLI?
   - **Cleanup Scope**: Are we preserving `.git`, model checkpoints, and virtual environments, or wiping code only?
   - **Core Mathematical/Algorithmic Direction**: What specific algorithms, libraries, or methodologies are desired?
3. **Preserve High-Value Artifacts**: Always keep model checkpoints (`checkpoints/`), virtual environments (`.venv/`), and VCS history (`.git`).

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

### Phase 4: Host Environment Discovery & Safe Synchronization

Host applications often maintain their own configuration stores and separate library directories.

1. **Locate Host Configuration**:
   - Scan standard application data directories (`%APPDATA%`, `~/.config`, etc.) for configuration JSONs.
   - Inspect library registration lists, active workspace directories, and IPC settings.
2. **Inspect Host Python & Virtualenv**:
   - Identify the host's Python bundle and package manager (e.g., `uv`, embedded Python).
   - Check the target library's `.venv` without polluting or destroying existing dependencies.
3. **Non-Destructive Sync**:
   - Remove only deprecated source code files.
   - Strictly preserve `.venv/`, `checkpoints/`, and user caches in the host directory.
   - Copy new `core/`, `nodes/`, `tests/`, and manifest files.
4. **In-Host Validation**:
   - Execute the test suite directly using the host virtual environment:
     `& "<host_venv>/python.exe" -m pytest tests/`
   - Test importing all node classes inside the host Python to guarantee zero startup crashes.

---

### Phase 5: Agile Feedback Refinement & Boundary Optimization

When iterating based on user review, apply these optimization patterns:

#### 1. Aggressive Feature Pruning
- If a secondary process (e.g. RGB color post-processing) is irrelevant to the core output (e.g. alpha mask generation), remove it.
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
   - Provide a visual ASCII/Mermaid pipeline diagram.
   - Structure documentation into "Quick All-in-One Course" and "Modular Pipeline Course".
   - Include copy-pasteable host registration JSON snippets.

---

## 🛠️ Verification Checklist for Any Tool Project

Before declaring a tool development cycle complete, verify:
- [ ] Core algorithms run independently of any host app imports.
- [ ] Compatibility shim allows headless node/adapter testing.
- [ ] Unit tests cover core mathematical functions with synthetic data.
- [ ] Host configuration points to the correct library manifest path.
- [ ] Virtual environment dependencies are synchronized via the host's package manager.
- [ ] All node classes import successfully in the host's Python runtime.
- [ ] Extreme boundary values are strictly locked (no float noise or internal jitter).
- [ ] Git working tree is clean and changes are pushed to remote.
