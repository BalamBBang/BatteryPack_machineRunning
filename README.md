# 🔋 배터리팩 이상감지

## ⚠️ 실행 전 사항
- 메모리가 적어도 **32GB 이상**의 사양을 가지는 기기에서 실행해야 합니다.
- 후술할 데이터셋 출처에서 데이터를 그대로 풀어서 사용하면 됩니다.

## 📦 데이터셋 출처
- [KAMP 전자부품(배터리팩) 품질보증 AI 데이터셋](https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=&page=2&DATASET_SEQ=58&DISPLAY_MODE_SEL=CARD&EQUIP_SEL=&GUBUN_SEL=C004027&FILE_TYPE_SEL=C005002&WDATE_SEL=)

---

## 🔧 설정 세팅 (셀 3~5)

라이브러리 임포트 및 환경 설정을 수행합니다.

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go, plotly.express as px
```

- 한글 폰트: `NanumGothic` 설정
- `axes.unicode_minus = False`

---

## 1. 파일 불러오기 (셀 6~30)

```python
HOME = os.getcwd()
train_path = os.path.join(HOME, "data", "raw_data", "train")
test_path  = os.path.join(HOME, "data", "raw_data", "test")
```

- `train_path_list`, `test_path_list`에 경로를 저장해두고 사용
- `chg`, `dchg`는 각각 **충전**과 **방전**을 의미
- test 파일의 `OK`와 `NG`는 각각 이상이 없는 데이터셋과 이상이 있는 데이터셋
- **모든 train 파일은 이상이 없는 데이터셋**

### 데이터 탐색 결과 (셀 10~16)

| 항목 | 내용 |
|------|------|
| 총 column 수 | 231개 |
| 시계열 컬럼 | `Date`, `Time` |
| `Date` 처리 | 한 데이터셋 내부에서 동일하므로 삭제 권장 |

### 단일값 & 결측치 탐색 (셀 17~23)

- `check_columns_one(train_path_list, length=4)` 함수로 모든 값이 동일한 column이 일정 수(`length`) 이상인 데이터셋을 탐색
- `find_nan(train_path_list)` 함수로 결측치를 가진 데이터를 탐색
- 값이 변하지 않는 단일값이 많으면 PCA나 학습에서 좋지 않은 결과를 보여줄 수 있으므로 삭제

**결측치 보유 파일:**
`1012_dchg.csv`, `1013_chg.csv`, `1025_chg.csv`, `1026_chg.csv`, `1033_dchg.csv`, `1050_chg.csv`

**단일값 과다 파일:**
`1013_dchg.csv`, `1014_dchg.csv`, `1015_dchg.csv`, `1016_dchg.csv`, `1017_chg.csv`, `1017_dchg.csv`, `1019_chg.csv`, `1030_chg.csv`, `1032_chg.csv`, `1035_chg.csv`, `1036_chg.csv`, `1038_chg.csv`, `1043_chg.csv`

### EDA 시각화 (셀 24~30)

- `make_df_list()`로 train/test 데이터를 DataFrame 리스트로 로드
- `plot_comparison()` 함수로 각 모듈별 배터리셀 비교 시각화 (HTML 파일로 저장)
- 불량 정의:

| 불량 유형 | 설명 |
|-----------|------|
| 용량불량 | 특정 배터리셀에서 충전 중 급격한 전압 상승/하강 |
| 용접불량 | 특정 배터리셀에서 전압 미측정 또는 전체 전압 하락 |
| 센싱 와이어 불량 | 인접 배터리셀들의 전압 차이 발생 |
| 센서 불량 | 온도센서 측정값이 너무 높거나 낮게 출력 |

- test 파일의 파형: 중간에 갑자기 튀어오르는 경향, 기울기가 꺾이는 현상 발생
- test 파일에는 시계열 데이터와 해당 부분만 정의되어 있어, 파생변수를 만들거나 더 뛰어난 모델이 필요

> ⚡ **핵심 주의사항**: test 파일에 맞도록 하되, 절대 test 파일을 사용하지 않고, test 파일의 결과에만 과적합하지 않아야 한다.

---

## 전처리 (셀 31~46)

### 백업 (셀 32~33)

```python
backup_train_list = train_list.copy()
backup_test_list = test_list.copy()
```

### 3가지 처리: 단일값, 결측치, 이상치 (셀 34~38)

```python
def choose_columns(df_list):
    # Time(iloc[:, 1]) + 유효 전압 컬럼(iloc[:, 23:]) 만 선택
```

- `handle_single(df)`: 단일값 존재 여부 확인
- `handle_nan(df)`: 결측치 존재 여부 확인
- `preprocessing(df_list, single=True, nan=True)`: 단일값/결측값이 있는 데이터를 리스트에서 제거
- test에는 single과 nan이 존재하지 않는 것을 확인함

### 파생변수 생성 (셀 41)

`add_derived_features(df)` 함수로 3종류 48개의 파생변수를 생성합니다.

| 파생변수 | 설명 | 개수 |
|----------|------|------|
| `{M}_cell_dev` | 모듈 내 셀간 전압 편차 (std) | 16개 |
| `{M}_mod_dev` | 모듈간 전압 편차 (모듈 평균 - 전체 평균) | 16개 |
| `{M}_temp_dev` | 모듈간 온도 편차 | 16개 |
| **합계** | | **48개** |

- 모듈(M01~M16)별로 셀 전압(CV), 온도(T) 컬럼을 사용

### 파생변수 시각화 (셀 42~46)

- `visualize_derived_features_interactive()`: test 파일의 파생변수 인터랙티브 시각화 (GT 이상구간 빨간 음영)
  - 3행 subplot: 셀간 편차(cell_dev), 모듈간 전압 편차(mod_dev), 모듈간 온도 편차(temp_dev)
  - 롤링 스무딩 지원 (`smooth`, `smooth_mode='median'` 또는 `'mean'`)
- `visualize_derived_features_train()`: train 파일의 파생변수 시각화

---

## 모델링 1. LSTM 기반 이상탐지 (셀 47~58)

### LSTM Autoencoder 구조 (셀 52)

```
Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2)
→ Dense(LATENT_DIM, relu)
→ RepeatVector(WIN_SIZE) → LSTM(32) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
→ TimeDistributed(Dense(features_dim))
```

### 하이퍼파라미터 (셀 48)

| 파라미터 | 값 |
|----------|-----|
| `WIN_SIZE` | 100 |
| `FEATURES_DIM` | 48 |
| `STEP_SIZE` | 5 |
| `BATCH_SIZE` | 256 |
| `EPOCHS` | 50 |
| `LATENT_DIM` | 32 |
| `DERIVED_SUFFIXES` | `('_cell_dev', '_mod_dev', '_temp_dev')` |

### 데이터 준비 (셀 49)

- `get_derived_cols(df)`: `DERIVED_SUFFIXES`로 끝나는 컬럼 추출
- `make_sequences(X, window_size, step_size)`: 롤링 윈도우 시퀀스 생성
- `SimpleImputer(strategy='mean')` → `MinMaxScaler(feature_range=(0, 1))`로 정규화

### 전압/온도 따로 학습 (셀 50~53)

| 모델 | 입력 차원 | 파생변수 접미사 |
|------|-----------|----------------|
| `model_volt` | 32 (cell_dev 16 + mod_dev 16) | `_cell_dev`, `_mod_dev` |
| `model_temp` | 16 (temp_dev 16) | `_temp_dev` |

- **Optimizer**: `Adam(lr=0.0001)`, **Loss**: `mse`
- **Callbacks**: `EarlyStopping(patience=10, min_delta=1e-6)`, `ModelCheckpoint`, `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`
- 각각 별도의 `imputer`, `scaler` 사용 (`imputer_v`/`scaler_v`, `imputer_t`/`scaler_t`)

### 임계값 설정 (셀 54)

- 전압/온도 각각 train 재구성 오차의 **99.5 percentile**을 임계값으로 설정
- `val_loss`보다 임계값이 낮으면 `val_loss * 1.5`로 조정

### 이상탐지 추론 (셀 55~56)

- `detect_anomalies_dual(df)` 함수로 전압/온도 각각 재구성 오차를 계산
- Huber loss로 오차 계산 → Z-score 정규화 후 합산 (`z_v + z_t`)
- `err_v > threshold_v` OR `err_t > threshold_t` 조건으로 이상 구간 판정
- 시각화: 전압 재구성 오차, 온도 재구성 오차, 합산 Z-score를 3행 subplot으로 표시
  - GT 이상구간: 빨간 음영, 예측 이상구간: 파란 음영

### 모델 저장 (셀 57)

```
checkpoints_lstm/volt_autoencoder_final.keras
checkpoints_lstm/temp_autoencoder_final.keras
```

### 🔍 LSTM 한계 (셀 58)
> LSTM은 **추세**를 반영하는 모델이다보니 추세에 대해서만 반영하고 **크기**에 대해서는 고려되지 않는다. 따라서 크기에 대한 보강, 그리고 적은 특징성에 대해서 보완이 필요하다.

---

## 모델링 2. Hybrid 모델링 (셀 59~67)

### 설정 (셀 60)

```python
INPUT_DIR      = "./data/raw_data/train"
TEST_RAW_DIR   = "./data/raw_data/test"
TEST_LABEL_DIR = "./data/preprocessed/test"
OUTPUT_DIR     = './hybrid_outputs'
```

### HybridConfig (셀 61)

| 파라미터 | 기본값 |
|----------|--------|
| `scaler_type` | `robust` |
| `pca_components` | 12 |
| `window` | 100 |
| `step` | 5 |
| `contamination` | 0.01 |
| `include_ok_as_normal` | True |
| `use_defect_features` | True |
| `defect_type_threshold` | 2.5 |

- PCA + IsolationForest 기반
- 학습 제외 ID (`DEFAULT_BAD_IDS`):

| 모드 | 제외 ID |
|------|---------|
| chg | 1009, 1017, 1019, 1026, 1030, 1035, 1036, 1038, 1043 |
| dchg | 1013, 1014, 1015, 1016, 1017, 1036, 1043 |

### 학습 흐름 (셀 62~67)

1. **charge 모델** 학습 → `train_hybrid_model(chg_cfg)`
2. **discharge 모델** 학습 → `train_hybrid_model(dchg_cfg)`
3. 모델 저장:
   - `hybrid_outputs/hybrid_charge_model.joblib`
   - `hybrid_outputs/hybrid_discharge_model.joblib`

---

## 모델링 3. One-Class SVM 다변량 시계열 모델링 (셀 68~104)

### 환경 세팅 (셀 69)

```python
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
SEED = 0
```

- `PROJECT_DIR` 자동 탐색으로 `data/raw_data/train` 디렉토리 위치 결정

### 전처리 함수 (셀 70)

| 함수 | 설명 |
|------|------|
| `trim_all_null_tail(df)` | 끝부분 전체 결측 행 제거 |
| `handle_missing_numeric(df)` | 보간(`interpolate`) → `ffill`/`bfill` → 0 채움 |
| `prepare_raw_frame(csv_path)` | Date/Time 파싱 + 결측 처리 통합 |
| `extract_signal_groups(df)` | CV 컬럼과 온도(T) 컬럼 분리 |

### 행 단위 피처 생성 (셀 71): `build_row_feature_frame`

| 피처 | 설명 |
|------|------|
| `cell_v_spread` | 셀 전압 max - min |
| `cell_v_std` | 셀 전압 표준편차 |
| `cell_adj_gap_max` | 인접 셀 전압 차이 최대 |
| `cell_adj_gap_mean` | 인접 셀 전압 차이 평균 |
| `cell_jump_max` | 시점간 셀 전압 급변 최대 |
| `temp_spread` | 온도 max - min |
| `temp_jump_max` | 시점간 온도 급변 최대 |
| `current`, `power` | 전류, 전력 |
| `voltage_jump` | 전체 전압 시간차 절대값 |
| `active` | 활성 구간 여부 |

### 활성 구간 감지 (셀 72)

- `_quantile_activity_mask()`: 변화량의 10% 분위수 이상인 구간을 활성으로 판정
- `dilate_binary(values, padding)`: 활성 마스크를 패딩하여 확장
- `build_active_mask()`: metadata(`active`) + structural(voltage_jump, cell_jump_max, temp_jump_max) 마스크를 결합 (hybrid 전략)

### 학습 데이터 품질 감사 (셀 73~76)

- `compute_train_file_metrics()`: 결측, 중복 시간, 상수 컬럼 비율 등 파일별 품질 지표 계산
- `build_flags()`: `schema_mismatch`, `heavy_missing_rows`, `heavy_duplicate_datetime`, `too_many_constant_cell_voltages` 등 기준으로 drop/review 판정
- `build_train_pair_inventory()`: battery_id별 chg/dchg 쌍 존재 여부와 제외 사유 기록
- `SELECTED_TRAIN_PAIR_IDS`: `pair_status == 'use'`인 배터리 ID만 선정

### 전처리 효과 확인 (셀 77)

- `summarize_preprocess_effect()`: raw → tail trim → datetime 제거 → 최종 행 수, 결측 셀 수 변화 요약

### 채점 함수 (셀 78): `score_file`

```python
windows → scaler.transform → reducer.transform → model.decision_function
→ window_scores_to_point_scores → active_mask 적용 → pred_bin
```

### 고장 단서 피처 (셀 79): `compute_fault_features`

| 피처 | 설명 |
|------|------|
| `voltage_spread_p95` | 셀 전압 편차 95th percentile |
| `tail_voltage_spread_mean` | 후반 20% 구간 전압 편차 평균 |
| `voltage_jump_p99` | 전압 급변 99th percentile |
| `adjacent_gap_p99` | 인접 셀 전압 차이 99th percentile |

### ModeSuite 데이터클래스 (셀 80)

| 필드 | 설명 |
|------|------|
| `mode` | `chg` 또는 `dchg` |
| `scaler` | `StandardScaler` |
| `reducer` | `PCA` |
| `model` | `OneClassSVM(kernel='rbf', gamma='scale')` |
| `threshold` | 검증 데이터 기반 결정 임계값 |
| `win_size`, `step_size` | 윈도우/스텝 크기 |
| `active_padding` | 활성 마스크 패딩 |
| `pca_components` | PCA 차원 수 |
| `nu` | OCSVM nu 파라미터 |
| `threshold_quantile` | 임계값 분위수 |

### 고장 주입 - 합성 데이터 (셀 82): `inject_fault_case`

정상 데이터에 4종 고장을 주입하여 검증용 합성 케이스를 생성합니다.

| 고장 유형 | 주입 방법 |
|-----------|-----------|
| `capacity_fault` | 후반부(70~95%) 특정 셀에 전압 ramp 주입 |
| `welding_fault` | 특정 셀 전압을 delta_v만큼 하락 |
| `sensing_wire_fault` | 인접 셀 간 전압 차이 주입 |
| `sensor_fault` | 온도 센서 값을 delta_t만큼 변경 |

### 모델 학습 (셀 83): `train_mode_suite_from_records`

```python
X_train = np.concatenate([record['windows'] for record in fit_records])
scaler = StandardScaler()
reducer = PCA(n_components=pca_components)
model = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)

scaled_fit → reduced_fit → model.fit
```

- `max_fit_windows` 이상이면 랜덤 샘플링

### 기본 후처리 (셀 85~86): `generalized_postfilter`

- 입력: raw 예측 마스크
- 처리: 짧은 잡음 제거, 가까운 구간 병합, 너무 약한 구간 제거
- 6개 파라미터: `low_ratio`, `merge_gap_rows`, `min_interval_len`, `persistent_ratio_threshold`, `head_interval_start_ratio_max`, `tail_interval_end_ratio_min`
- `low_ratio`로 하한/상한 이중 임계값 적용 후 구간 병합

### 고장 인지 후처리 (셀 87~88): `fault_aware_postfilter`

- 입력: raw 예측 마스크 + 고장 단서 점수 (`fault_scores`, `top_fault_name`, `top_fault_score`)
- `top_fault_score < fault_reject_threshold` 이고 예측 비율이 낮으면 → 경보 제거 (`fault_gate_reject`)
- `top_fault_score >= fault_boost_threshold` 이면 → fragment 병합, tail 패딩으로 구간 보강

### 최종 설정 선택 (셀 89~90): `auto_calibrate_mode`

탐색 순서:

1. **PCA 차원** 선정 → `evaluate_pca_stage`
2. **OCSVM nu + 임계값** 선정 → `search_detector_stage`
3. **기본 후처리 6종 파라미터** 전수 탐색 (최대 4,800가지)
4. **고장 인지 파라미터** 탐색
5. 전체 데이터 최종 재학습

- 학습용(`fit_ids`) / 검증용(`calib_ids`) 분할 후 합성 고장 케이스와 함께 평가

### 평가와 시각화 함수 (셀 91~94)

- `fit_train_derived_bundle(cfg)`: 충전/방전 모두 `auto_calibrate_mode` 실행
- `evaluate_train_derived_bundle(bundle, postfilter_type)`: test 파일별 점수 계산, 성능 요약
  - `postfilter_type`: `'generalized'` 또는 `'fault_aware'`
- `_plot_detail_on_axis()`: raw 점수(회색), 활성 마스크 적용 점수(주황), 임계값(검정 점선), GT 이상구간(빨간 음영), raw 예측(보라 테두리), 최종 예측(파란 음영)

---

## 설정 탐색 실행 (셀 95~97)

```python
derived_bundle = fit_train_derived_bundle(AUTO_DERIVED_CFG)
```

- PCA 차원 선정 → OCSVM(nu·임계값) 선정 → 기본 후처리 전수 탐색 → 고장 인지 파라미터 탐색 → 전체 데이터 최종 재학습을 순서대로 수행
- 기본 후처리는 6개 파라미터 조합을 전수 탐색 (최대 4,800가지)
- 실행 시간: 약 5분 내외
- 모델 저장: `Battery_THK.pkl` (pickle 형식)
- 이미 `Battery_THK.pkl` 파일이 있으면 이 셀을 건너뛰고 진행 가능

### 결과 요약 테이블 (`derived_mode_view`)

| 컬럼 | 설명 |
|------|------|
| 모드 | 충전 / 방전 |
| PCA 차원 수 | 선택된 PCA 차원 |
| OCSVM nu | 선택된 nu 값 |
| 임계값 분위수 | 선택된 threshold quantile |
| 정상 경보율 | 검증 구간 정상 경보율 |
| OCSVM 후보 수 | 탐색된 OCSVM 후보 수 |
| 후처리 후보 수 | 탐색된 후처리 후보 수 |
| 고장 인지 후보 수 | 탐색된 고장 인지 후보 수 |

---

## 탐색 과정 확인 (셀 98~99)

- 최종 선택값을 정한 뒤 어떤 후보들을 비교했는지 역으로 확인
- 모드별: 전체 학습 ID 수, 학습용/검증용 ID 수, 합성 고장 케이스 수
- 단계별: PCA 후보 수 → 선택 PCA, OCSVM 후보 수 → 선택 nu/임계값 분위수, 후처리 후보 수, 고장 인지 후보 수
- 후처리 후보 수는 6개 파라미터 전수 탐색 기준으로 최대 4,800가지

---

## 테스트 평가 (셀 100~104)

### 기본 후처리 vs 고장 인지 후처리 비교 (셀 100)

`summary_compare_df`로 두 방식을 비교:

| 비교 항목 | 설명 |
|-----------|------|
| 정상 무경보 | 정상 파일 중 경보가 없는 파일 수 |
| 고장 검출 | 고장 파일 중 경보가 발생한 파일 수 |
| Precision | 평균 정밀도 |
| Recall | 평균 재현율 |
| F1 | 평균 F1 점수 |

### 테스트 파일 기본 특성 (셀 101~102)

- 설정 선택에 사용한 정보가 아닌, 최종 성능을 해석하기 위한 참고 자료
- 각 테스트 파일의 셀 전압/온도 변화량이 얼마나 큰지 확인

| 특성 | 설명 |
|------|------|
| 셀 전압 편차 95% | `cell_v_spread` 95th percentile |
| 셀 전압 급변 95% | `cell_jump_max` 95th percentile |
| 온도 변화 95% | `temp_spread` 95th percentile |
| 온도 급변 95% | `temp_jump_max` 95th percentile |

### 파일별 결과 요약 (셀 103)

`file_summary_df`로 각 테스트 파일별 상세 결과:

| 항목 | 설명 |
|------|------|
| 실제 이상 비율 | `gt_ratio` |
| 기본/고장 인지 경보 구간 수 | `n_pred_intervals` |
| 기본/고장 인지 경보 길이 비율 | `final_pred_ratio` |
| 기본/고장 인지 Precision, Recall, F1 | 각각의 성능 지표 |
| F1 변화량 | 고장 인지 F1 - 기본 F1 |
| 최종 판정 근거 | `accept_reason` |
| 추정 고장 유형 | `top_fault_name` (예: 센싱와이어불량, 센서불량 등) |

### 파일별 좌우 비교 시각화 (셀 104)

- 같은 파일에 대해 **기본 후처리 결과**(좌)와 **고장 인지 후처리 결과**(우)를 한 figure 안에 좌우로 배치
- 정상 파일: 경보 비율 변화, 판정 사유 표시
- 고장 파일: Recall, F1 변화, 추정 고장 유형 표시

---

## 📂 디렉토리 구조

```
프로젝트/
├── data/
│   ├── raw_data/
│   │   ├── train/          # 학습 데이터 (정상, *_chg.csv / *_dchg.csv)
│   │   └── test/           # 테스트 데이터 (Test*_OK_*.csv / Test*_NG_*.csv)
│   └── preprocessed/
│       └── test/           # 라벨 파일 (*_Label.csv)
├── checkpoints_lstm/       # LSTM 모델 저장
├── hybrid_outputs/         # Hybrid 모델 저장
└── Battery_THK.pkl         # One-Class SVM 최종 모델
```

---

## 🏗️ 전체 파이프라인 요약

```
데이터 로드 → EDA (컬럼 분석, 단일값/결측치 탐색, HTML 시각화)
    → 전처리 (컬럼 선택, 단일값/결측 데이터 제거)
    → 파생변수 생성 (cell_dev 16 + mod_dev 16 + temp_dev 16 = 48개)
    → 파생변수 시각화 (인터랙티브)
    → 모델링 1: LSTM Autoencoder (전압/온도 분리 학습) → 한계: 크기 미반영
    → 모델링 2: Hybrid (PCA + IsolationForest, charge/discharge 분리)
    → 모델링 3: One-Class SVM (행단위 피처 + PCA + OCSVM)
        → 학습 데이터 품질 감사 (drop/review 판정)
        → 합성 고장 주입 (4종)
        → 자동 설정 탐색 (PCA → nu → 기본 후처리 6종 → 고장 인지 후처리)
        → 기본 후처리 (잡음 제거, 구간 병합, 약한 구간 제거)
        → 고장 인지 후처리 (고장 단서 기반 보강/기각)
    → 평가 및 비교 시각화 (기본 vs 고장 인지, 파일별 좌우 비교)
```

---

## 📋 의존성

| 패키지 | 용도 |
|--------|------|
| `numpy`, `pandas` | 데이터 처리 |
| `scikit-learn` | PCA, StandardScaler, MinMaxScaler, OneClassSVM, IsolationForest, 평가 지표 |
| `matplotlib`, `seaborn` | 정적 시각화 |
| `plotly` | 인터랙티브 시각화 |
| `tensorflow` | LSTM Autoencoder (모델링 1) |
| `joblib` | Hybrid 모델 저장 (모델링 2) |
| `pickle` | One-Class SVM 모델 저장 (모델링 3) |
