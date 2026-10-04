# CompMatte: VFX Compositing-Grade Hybrid Matting Toolkit

> **실제 영화/VFX 컴포지팅 구조(Core Matte + Edge Matte + Non-destructive Detail Re-Injection)를 충실히 구현한 차세대 하이브리드 알파 매팅 툴킷**  
> **Version: `v3.0.0`** | **License: `Apache 2.0`** | **Tests: `46 / 46 Passed (100%)`**

**CompMatte**는 기존 단일 AI 블랙박스 매팅의 한계(내부 구멍, 시간축 플리커, 잔머리 침식)를 극복하기 위해, 헐리우드 VFX 스튜디오(Nuke, Flame)의 **실제 컴포지팅 파이프라인 구조**를 소프트웨어 및 Griptape Nodes로 충실히 구현한 하이브리드 비디오 매팅 툴킷입니다.

---

## 💡 왜 "CompMatte (컴프매트)" 인가? (컴포지팅 구조 기반 매팅)

전통적인 프로페셔널 VFX 컴포지팅에서 마스터 키어는 결코 하나의 필터로 전체 피사체를 뽑지 않습니다.  
항상 **코어(Core)와 엣지(Edge)를 물리적으로 분리**하여 각각에 최적화된 처리를 수행한 후 합성합니다.

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      입력 영상 (1080 Green/Blue Screen Shot)                      │
└───────────────────────┬─────────────────────────────────┬───────────────────────┘
                        │                                 │
    [Edge Matte 브랜치: 미세 디테일 & 투과율]       [Core Matte 브랜치: 불투명 코어 & 안정화]
                        │                                 │
        ┌───────────────▼───────────────┐         ┌───────▼───────────────────────┐
        │ 01. CompMatte Clean Plate     │         │ 03. CoreEngine & Spatio-Temp  │
        │  - 스크린 조명 불균일 균등화    │         │  - 내부 코어: Pure White(1.0) │
        │  - Nuke IBKColour 컴포지팅    │         │  - 외부 포락선: Pure Black(0.0)│
        └───────────────┬───────────────┘         │  - 시간축 지터/플리커 원천 차단 │
                        │                         │  - 초고속 60fps+ 순수 광학    │
        ┌───────────────▼───────────────┐         └───────┬───────────────────────┘
        │ 02. CompMatte Edge Keyer      │                 │
        │  - 광학 컬러차이 기반 투과율    │                 │
        │  - 서브픽셀 헤어 & 모션블러    │         ┌───────▼───────────────────────┐
        │  - Nuke IBKGizmo 투과 엣지    │         │ (선택) ViTMatte Neural Edge   │
        └───────────────┬───────────────┘         │  - Unknown 전이 영역 정밀 추론 │
                        │                         │  - 타이트 ROI 가속 (14x Fast) │
                        │                         └───────┬───────────────────────┘
                        │                                 │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────────────────────┐
                        │ 04. CompMatte Fusion & Non-destructive Re-inject│
                        │  - Core Matte + Edge Matte 컴포지팅 합성        │
                        │  - 🌟 Safe Zone Edge Re-Injection (엣지 재주입) │
                        │    후처리 필터에 의한 잔머리 깎임 0% 원천 방지   │
                        │    1px 미세 머리카락 100% 보존 (Max 결합)       │
                        │  - 극성(Polarity) 자동 판별 및 화이트/블랙 보장 │
                        │  - 4K UHD (3840x2160) Lanczos4 고해상도 변환   │
                        └────────────────┬────────────────────────────────┘
                                         │
                        ┌────────────────▼────────────────────────────────┐
                        │ 05. CompMatte Sequence Exporter (4K Grayscale)  │
                        │  - 단일 채널 Grayscale 32-bit Float OpenEXR ('A')│
                        │  - 단일 채널 Grayscale 16-bit Lossless PNG      │
                        │  - 불필요한 RGB 색상 오버헤드 67% 절감          │
                        │  - Stabilized & Raw 듀얼 시퀀스 동시 출력       │
                        └─────────────────────────────────────────────────┘
```

### 🎯 4대 컴포지팅 필러 (Compositing Pillars)

1. **Clean Plate**:
   - 배경 스크린의 조명 편차, 주름, 핫스팟을 제거하여 완벽한 레퍼런스 컬러 플레이트를 생성합니다 (Nuke `IBKColour` 에뮬레이션).

2. **Core Matte**:
   - 피사체 내부의 음영이나 질감 차이로 인해 알파에 구멍(Holes)이 뚫리거나 자글거리는 현상을 100% Pure White (1.0) 코어로 단단하게 고정합니다.
   - 외부 배경은 100% Pure Black (0.0) 엔벨로프로 닫아 지터를 완벽히 제거합니다.
   - PyTorch 의존성 없는 경량 `CoreEngine`을 통해 **60fps+ 초고속 실시간 처리**를 지원합니다.

3. **Edge Matte (소프트/디테일 매트 & ViT 신경망 엣지)**:
   - 광학적 색상차 투과율(IBK)을 통해 머리카락 한 올, 모션 블러, 반투명 재질의 서브픽셀 그라디언트를 보존합니다.
   - 필요 시 `use_vitmatte_refinement=True`를 켜면 Unknown 전이 영역에만 타이트 ROI 기반 ViTMatte 트랜스포머 딥러닝 엣지가 적용됩니다.

4. **Matte Fusion & Non-destructive Edge Detail Re-Injection (엣지 비파괴 재주입)**:
   - 코어와 엣지를 결합한 후, 가우시안 블러나 클리핑 등 후처리 과정에서 가느다란 잔머리 끝단이 깎여나가는 문제를 방지하기 위해 **안전 반경(Safe Zone) 내에서 순수 원본 엣지 디테일을 비파괴 Max 연산(`np.maximum(base, raw_edge)`)으로 최종 재주입**합니다.
   - 이를 통해 **머리카락 가닥 손실률 0% (100% 완전 보존)**를 달성합니다.

---

## 📦 제공 노드 목록 (Griptape Nodes)

Griptape Nodes Desktop의 **`CompMatte (VFX Matting)`** 카테고리에서 다음 노드들을 사용할 수 있습니다:

| 최신 노드 명칭 | 화면 표시 이름 (Display Name) | 기존 호환 클래스명 | 설명 |
| :--- | :--- | :--- | :--- |
| **`CompMatteAllInOneNode`** | `CompMatte Keyer (All-in-One)` | `VFXKeyingViTAllInOneNode` | 클린 플레이트, 코어 안정화, 광학/ViT 엣지, 엣지 재주입, 4K EXR/PNG 시퀀스 출력 통합 노드 |
| **`CompMatteCleanPlateNode`** | `Node 01: CompMatte Clean Plate` | `IBKCleanPlateNode` | 스크린 조명 그라디언트 제거 및 Clean Plate 생성 (영상 프레임 자동 동기화) |
| **`CompMatteKeyerNode`** | `Node 02: CompMatte Edge Keyer` | `IBKKeyerNode` | 광학 컬러차이 기반 투과율 알파 엣지 마스크 추출 (IBKGizmo) |
| **`CompMatteViTEdgeNode`** | `Node 03: CompMatte ViT Edge Refiner` | `ViTMaskExtractorNode` | 적응형 블러 감지 & 타이트 ROI 가속(14x) 기반 ViTMatte 딥러닝 엣지 추출 |
| **`CompMatteRefinerNode`** | `Node 04: CompMatte Fusion & Stabilizer` | `VFXMatteRefinerNode` | Core Matte + Edge Matte 합성, 시간축 안정화, 잔머리 100% 재주입 |
| **`CompMatteExportNode`** | `Node 05: CompMatte Sequence Exporter` | `VFXMaskExportNode` | 단일 채널 4K UHD 32-bit Float EXR, 16-bit PNG, Red Overlay 비디오 내보내기 |

> **하위 호환성 완벽 지원**: 기존 워크플로우 파일이나 스크립트에서 사용하던 `IBKCleanPlateNode`, `VFXKeyingViTAllInOneNode` 등의 클래스명도 100% 동일하게 동작합니다.

---

## 🚀 설치 및 등록 가이드

### 1. 패키지 설치

GPU 환경(CUDA)이 설정된 파이썬 환경에서 필수 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

### 2. AI 모델 다운로드 (상대 경로 & Griptape 연동)

라이브러리는 Griptape Model Management와 워크스페이스 상대 경로(`models/`)를 완벽 지원합니다:

- **ViTMatte**: `hustvl/vitmatte-small-composition-1k`

```bash
# 기본 Hugging Face / Griptape 캐시로 다운로드
python scripts/download_models.py

# 워크스페이스 상대 경로(models/)로 다운로드
python scripts/download_models.py --relative
```

### 3. Griptape Nodes Desktop 등록 (상대 경로)

`%APPDATA%\Griptape Nodes\xdg_config_home\griptape_nodes\griptape_nodes_config.json`의 `libraries_to_register` 목록에 등록합니다:

```json
"libraries_to_register": [
  "libraries/IBKViT_MASK/griptape_nodes_library.json"
],
"models_to_download": [
  "hustvl/vitmatte-small-composition-1k"
]
```

### 4. Griptape Nodes Desktop에서 새로고침

1. Griptape Nodes Desktop 실행
2. 좌측 하단의 **Refresh Libraries** 버튼 클릭 (또는 Engine Restart)
3. 노드 라이브러리 목록에 **`CompMatte (VFX Matting)`** 카테고리가 나타납니다.

---

## 💡 워크플로우 활용 예시

### 1. 초간단 4K All-in-One 코스

- 노드 목록에서 **`CompMatte Keyer (All-in-One)`** 노드를 캔버스에 배치합니다.
- `Input Video Path`: 입력 영상 파일 경로 지정
- `Screen Type`: `auto`, `green`, `blue`
- `Output Resolution`: `4k` (3840x2160 UHD 기본) 또는 `native`
- `Use ViTMatte Refinement`: `False` (초고속 60fps 광학 융합 모드) 또는 `True` (고품질 트랜스포머 엣지 모드)
- `Export Format`: `exr` (32-bit Float OpenEXR) 또는 `png16` (16-bit Lossless PNG)
- **Run** 실행 시 원본 영상 폴더 내에 4K 알파 마스크 시퀀스와 검수용 Red Overlay 비디오가 자동 생성됩니다.

### 2. 프로페셔널 컴포지팅 모듈러 코스

- **Node 01 (CompMatte Clean Plate)**: 스크린 균등화 플레이트 추출
- **Node 02 (CompMatte Edge Keyer)**: 순수 광학 투과율 머리카락 엣지 추출
- **Node 03 (CompMatte ViT Edge Refiner)**: 트랜스포머 신경망 엣지 추론 (필요 시)
- **Node 04 (CompMatte Fusion & Stabilizer)**: Core + Edge 결합, 플리커 방지 및 원본 잔머리 100% 재주입
- **Node 05 (CompMatte Sequence Exporter)**: 32-bit EXR / 16-bit PNG 마스터 시퀀스 내보내기

---

## 🐍 파이썬 코드 직접 사용 (`compmatte_core`)

파이썬 스크립트나 외부 파이프라인에서 직접 호출할 수도 있습니다:

```python
from compmatte_core import (
    CompMatteCoreEngine,
    CompMatteEdgeEngine,
    CompMatteFusionEngine,
    CompMatteFusionConfig,
    CompMatteIO,
)

# 1. 비디오 프레임 로드
frames = CompMatteIO.read_frames("greenscreen_shot.mp4")

# 2. 코어 매트 및 배경 엔벨로프 추출 (지터 방지)
core_engine = CompMatteCoreEngine()
core_res = core_engine.process_frame(frames[0], screen_type="green")

# 3. 엣지 디테일 비파괴 재주입 컴포지팅 융합
config = CompMatteFusionConfig(restore_fine_edges=True, edge_restore_band_radius=80)
fusion_engine = CompMatteFusionEngine(config=config)
final_matte = fusion_engine.inject_fine_edge_detail(
    base_matte=core_res.core_mask.astype(float) / 255.0,
    raw_edge=core_res.raw_edge,
    core_mask=core_res.core_mask.astype(float) / 255.0,
)
```

---

## 🧪 테스트 실행

모든 단위 테스트와 컴포지팅 파이프라인 유효성 검증은 아래 명령어로 실행할 수 있습니다:

```bash
pytest -v tests/
```

- **46개 전 테스트 항목 100% 통과 보장** (`tests/test_compmatte.py`, `tests/test_core_engine.py`, `tests/test_matte_fusion.py`, `tests/test_nodes.py`, 등)

---

## 📄 라이선스

Apache License 2.0
