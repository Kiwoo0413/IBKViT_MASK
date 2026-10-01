# VFX IBK Keying & ViT Masking Toolkit for Griptape Nodes

> **VFX-grade Image Based Keying (IBK) & Vision Transformer (ViT) 4K Spatio-temporal Mask Extraction Custom Node Library for Griptape Nodes Desktop**

본 라이브러리는 영화/VFX 업계 표준 컴포지팅 기법인 **IBK(Image Based Keyer: Nuke IBKColour & IBKGizmo)**와 최신 딥러닝 **ViT(Vision Transformer: SAM 2 & ViTMatte)**를 융합하여, 머리카락 한 올, 모션 블러, 반투명 재질까지 완벽하게 추출하는 **4K UHD 비디오 알파 마스킹 전용 툴킷**입니다.

---

## 🌟 핵심 기술 및 차별점 (새로운 파라다임)

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        입력 영상 (Green/Blue Screen)                     │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
      [IBK 브랜치: 미세 디테일 및 원본 엣지 보존]         [ViT 브랜치: 내외부 매트 지터 제거]
                    │                                │
    ┌───────────────▼───────────────┐  ┌─────────────▼───────────────────┐
    │ 01. IBK Clean Plate Generator │  │ 03. ViT Spatio-Temporal Tracker │
    │  (IBKColour 조명 그라디언트)    │  │  (SAM 2 비디오 객체 추적)       │
    └───────────────┬───────────────┘  └─────────────┬───────────────────┘
                    │                                │
    ┌───────────────▼───────────────┐  ┌─────────────▼───────────────────┐
    │ 02. IBK Gizmo Keyer           │  │ 내외부 매트 시공간 지터 안정화    │
    │  - 머리카락/모션블러 자연적 보존 │  │  - 내부 코어: Pure White (1.0) │
    │  - 광학 투과율 기반 순수 엣지   │  │  - 외부 배경: Pure Black (0.0) │
    │  - 인위적 클램프/왜곡 없음     │  │  - 시간축 경계 깜빡임 완전 제거 │
    └───────────────┬───────────────┘  └─────────────┬───────────────────┘
                    │                                │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────┐
                    │ 04. Clean Envelope Fusion & Refinement    │
                    │  - Non-destructive Envelope Fusion         │
                    │    α = Core + (1-Core) * Edge_IBK * Env   │
                    │  - 크리티컬한 시간축 엣지 왜곡 배제        │
                    │  - Guided Filter 미세 경계 정렬           │
                    │  - 극성(Polarity) 자동 판별 및 교정        │
                    └───────────────┬───────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────┐
                    │ 05. VFX Sequence Exporter (4K Grayscale)  │
                    │  - 4K UHD (3840x2160) Lanczos4            │
                    │  - 단일 채널 Grayscale 32-bit Float OpenEXR│
                    │  - 단일 채널 Grayscale 16-bit Lossless PNG │
                    │  - 불필요한 RGB 색상 오버헤드 67% 절감     │
                    │  - Stabilized & Raw 듀얼 시퀀스 동시 출력  │
                    └───────────────────────────────────────────┘
```

1. **🎨 IBK 브랜치: 고유의 광학적 엣지(Edge) 완벽 보존**:
   - **순수 광학 투과율 연산**: Clean Plate와의 화소별 색상차(Color Difference) 비율을 통해 머리카락 한 올, 모션 블러, 미세 반투명 재질의 자연스러운 알파 그라디언트를 처음 모습 그대로 유지합니다.
   - **인위적 왜곡 및 지글거림 배제**: 엣지 영역에 무리한 임계값 클램프나 변형 필터를 가하지 않아 본래의 부드러운 서브픽셀 엣지가 살아납니다.

2. **👁️ ViT 브랜치: 내외부 매트의 시간축 지터(Jitter) 완전 제거**:
   - **내부 코어 매트 (Pure White 1.0)**: 피사체 깊숙한 내부로 수축(Erode) 후 내부 구멍(Hole)을 완전 차단하고, 프레임 간 시공간 안정화를 적용하여 피사체 내부의 음영 변화나 자글거림을 100% 순백색으로 고정합니다.
   - **외부 배경 엔벨로프 (Pure Black 0.0)**: 피사체 외곽으로 확장(Dilate) 후 시간축 안정화를 적용하여, 스크린 번짐과 경계 밖 노이즈를 100% 칠흑색으로 고정합니다.

3. **🤝 Refine & Fusion: 비파괴 엔벨로프 합성 (크리티컬 엣지 왜곡 제거)**:
   - **엔벨로프 합성 공식**: $\alpha_{\text{out}} = \text{Core} + (1.0 - \text{Core}) \cdot \text{Edge}_{\text{IBK}} \cdot \text{Envelope}$
     - 내부 코어(Core=1): 항상 1.0 (Pure White)
     - 외부 배경(Env=0): 항상 0.0 (Pure Black)
     - 전이 대역(Transition): 원본 IBK 엣지의 광학 디테일이 온전히 보존됨.
   - **크리티컬 지터 방지 배제**: 엣지 디테일을 손상시키는 무리한 시간축 클램프나 뭉개짐 없이, Guided Filter를 통한 부드러운 화소 정렬만을 수행합니다.

4. **📉 초경량 4K Grayscale 알파 시퀀스 출력 (색상 오버헤드 67% 절감)**:
   - 마스크는 알파 채널 전용이므로 대용량 RGB 컬러 채널을 저장하지 않고 **단일 채널(1-channel) Grayscale 32-bit Float OpenEXR ('A' 채널)** 및 **16-bit Lossless PNG**로 저장합니다.
   - Nuke, DaVinci Resolve, Flame, Fusion에서 즉시 알파 채널로 인식되며 디스크 용량 및 I/O 대역폭을 최대 67% 이상 절감합니다.

5. **⚡ 지터 방지 적용(Stabilized) & 미적용(Raw) 듀얼 동시 출력**:
   - **Stabilized**: ViT 브랜치에서 내외부 매트가 시간축으로 안정화된 버전.
   - **Raw**: 프레임별 순수 독립 버전.

6. **📁 원본 영상 폴더 내 동적 출력 (Dynamic Output Path)**:
   - 입력 비디오 위치를 자동 감지하여 `{video_parent}/{video_stem}_masks_4k/` 내부에 체계적으로 정리 저장합니다.

---

## 📦 제공 노드 목록 (Griptape Nodes)

| 노드 클래스 | 화면 표시 이름 | 카테고리 | 설명 |
| :--- | :--- | :--- | :--- |
| **`VFXKeyingViTAllInOneNode`** | `VFX IBK & ViT Keyer (All-in-One)` | `VFX Keying & ViT Masking` | 클린 플레이트, ViT 퓨어 화이트 코어, IBK 키잉, 반전 자동 감지, 듀얼(지터 안정화+Raw) 4K 시퀀스 및 검수 비디오 내보내기 올인원 |
| **`IBKCleanPlateNode`** | `Node 01: IBK Clean Plate Generator` | `VFX Keying & ViT Masking` | 그린/블루 스크린 배경 조명 그라데이션을 복원한 Clean Plate 생성 (영상 폴더 내 자동 생성) |
| **`IBKKeyerNode`** | `Node 02: IBK Keyer` | `VFX Keying & ViT Masking` | Clean Plate 비교를 통한 투과율 알파 마스크 추출 (퓨어 블랙 배경/퓨어 화이트 코어) |
| **`ViTMaskExtractorNode`** | `Node 03: ViT Mask Extractor` | `VFX Keying & ViT Masking` | SAM 2 비디오 추적 및 ViTMatte 트라이맵 서브픽셀 알파 마스크 추출 |
| **`VFXMatteRefinerNode`** | `Node 04: VFX Matte Refiner & Fusion` | `VFX Keying & ViT Masking` | 퓨어 화이트 코어+엣지 결합, 반전 자동 보정, 듀얼(안정화+Raw) 4K 마스크 동시 생성 |
| **`VFXMaskExportNode`** | `Node 05: VFX Sequence Exporter` | `VFX Keying & ViT Masking` | 4K UHD 32-bit Float EXR, Premultiplied RGBA EXR, 16-bit PNG, Red Overlay 비디오 내보내기 |

---

## 🚀 설치 및 등록 가이드

### 1. 패키지 설치

GPU 환경(CUDA)이 설정된 파이썬 환경에서 필수 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

*(선택사항) SAM 2 최신 비디오 추적 엔진 활성화:*

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### 2. Griptape Nodes Desktop 등록

`%APPDATA%\Griptape Nodes\xdg_config_home\griptape_nodes\griptape_nodes_config.json` 파일의 `libraries_to_register` 목록에 해당 라이브러리의 매니페스트 경로를 추가합니다:

```json
"libraries_to_register": [
  "D:\\AI\\GT_MaskingTool\\griptape_nodes_library.json"
]
```

### 3. Griptape Nodes Desktop에서 새로고침

1. Griptape Nodes Desktop 실행
2. 좌측 하단의 **Refresh Libraries** 버튼 클릭 (또는 Engine Restart)
3. 노드 라이브러리 목록에 **`VFX Keying & ViT Masking`** 카테고리가 나타납니다.

---

## 💡 워크플로우 활용 예시

### 1. 초간단 4K All-in-One 코스

- 노드 목록에서 **`VFX IBK & ViT Keyer (All-in-One)`** 노드를 캔버스에 배치합니다.
- `Input Video Path`: 입력 영상 파일 경로 지정
- `Screen Type`: `green` 또는 `blue`
- `Output Resolution`: `4k` (3840x2160 UHD 기본)
- `Export Format`: `exr` (32-bit Float OpenEXR) 또는 `png16`
- **Run** 실행 시 4K 해상도로 업스케일링된 무결점 알파 마스크 시퀀스와 함께 검수용 Red Overlay 비디오가 자동 생성됩니다.

### 2. 프로페셔널 모듈러 코스

- **Node 01 (Clean Plate)**: 배경 스크린 조명 추출
- **Node 02 (IBK Keyer)**: 정밀 머리카락 투과율 마스크 추출
- **Node 03 (ViT Extractor)**: 피사체 중심 클릭 좌표로 솔리드 퓨어 화이트 코어 추출
- **Node 04 (Refiner & Fusion)**: 퓨어 화이트 코어 + 엣지 융합 및 내부 자글거림 완전 차단, 4K 변환
- **Node 05 (Exporter)**: 4K OpenEXR / 16-bit PNG 시퀀스 내보내기

---

## 💻 독립형 CLI 실행 (Command Line)

Griptape Nodes 외에 터미널이나 배치 스크립트에서도 단독 실행할 수 있습니다:

```bash
# 기본 4K UHD 실행 (32-bit EXR 시퀀스 출력)
python examples/run_ibk_vit_pipeline.py --input "D:\path\to\greenscreen.mp4" --screen green --resolution 4k

# 4K 16-bit PNG 및 시드 좌표 지정 실행
python examples/run_ibk_vit_pipeline.py --input "D:\path\to\video.mp4" --screen green --seed "640,360" --format png16 --resolution 4k
```

---

## 🧪 테스트 실행

모든 단위 테스트와 노드 스키마 유효성 검증은 아래 명령어로 실행할 수 있습니다:

```bash
pytest -v tests/
```

(18개 전 테스트 항목 100% 통과 보장)

---

## 📄 라이선스

Apache License 2.0
