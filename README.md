# 한국어 조사 Activation Sink 분석

> [Research] 한국어 조사 중심 Activation Sink 및 양자화 민감도 분석

본 레포지토리는 한국어 LLM 내부에서 **조사(postposition) 토큰이 문장 구조 정보를 과도하게 떠안는 activation sink로 작동하는지**를 정량적으로 분석하기 위한 연구 코드입니다. 단순 heatmap 기반 정성 분석을 넘어서, token-level activation, attention mass, layer-wise statistical test, matched fake activation quantization을 함께 측정합니다.

## 연구 요약

한국어 조사는 주어, 목적어, 부사어 등 문장 성분 간 관계를 표시하는 기능어입니다. 본 연구의 핵심 질문은 다음과 같습니다.

**한국어 LLM은 조사 위치를 문맥 구조 정보의 압축 지점으로 활용하는가?**

이를 검증하기 위해 `Bllossom/llama-3.2-Korean-Bllossom-3B` 모델을 대상으로 KoNLPy 기반 형태소 분석 결과와 Hugging Face subword token을 정렬하고, 조사 중첩 token과 비조사 token의 내부 표현 차이를 비교했습니다.

## 핵심 결론

| 가설 | 현재 실험 결과 | 해석 |
|---|---:|---|
| 조사 token의 activation peak가 더 큰가? | `True` | activation sink 가설은 비교적 강하게 지지됨 |
| 조사 token이 attention sink인가? | `False` | attention mass 집중 근거는 약하거나 제한적임 |
| 조사 token이 양자화에 더 민감한가? | `True` | selective fake quantization 기준에서 조사 token의 손실 민감도가 더 큼 |

보수적인 결론은 다음과 같습니다.

> 한국어 조사는 일반적인 attention sink라기보다, **sparse activation sink 또는 quantization-sensitive structural carrier** 후보로 해석하는 것이 더 타당합니다.

## 실험 설정

| 항목 | 값 |
|---|---:|
| 모델 | `Bllossom/llama-3.2-Korean-Bllossom-3B` |
| Decoder layers | `28` |
| Hidden size | `3072` |
| Attention heads | `24` |
| 문장 수 | `180` |
| 전체 token 수 | `2136` |
| 조사 중첩 token 수 | `304` |
| 형태소 분석기 | KoNLPy Okt |
| 주요 비교 | JOSA-overlapping token vs NON-JOSA token |

## 방법론

전체 파이프라인은 다음 순서로 구성됩니다.

1. 한국어 문장을 KoNLPy Okt로 형태소 분석합니다.
2. LLM tokenizer의 subword token과 형태소 span을 character offset 기반으로 정렬합니다.
3. 각 token의 hidden state에서 activation outlier 지표를 수집합니다.
4. 각 token이 key position으로서 받는 attention mass를 future-query-normalized 방식으로 계산합니다.
5. layer별로 JOSA와 NON-JOSA의 평균 차이를 paired test로 검정합니다.
6. 같은 수의 조사 token과 비조사 token에 selective fake activation quantization을 적용해 loss 증가량을 비교합니다.

## 주요 지표

### Activation sink 지표

| 지표 | 의미 |
|---|---|
| `max_abs` | token hidden state에서 가장 큰 절댓값 |
| `zmax` | layer 내부 scalar activation 분포 기준의 outlier score |
| `norm` | token hidden state의 L2 norm |
| `outlier_ratio` | calibrated threshold를 넘는 channel 비율 |
| `topk_share` | 상위 k개 channel이 차지하는 activation mass 비율 |

### Attention sink 지표

| 지표 | 의미 |
|---|---|
| `received_attn_future_mean` | 해당 token이 key position으로서 미래 query token들로부터 받은 평균 attention mass |
| `attn_sink_z_future` | sentence-layer 단위로 정규화한 attention sink score |

### Quantization sensitivity 지표

| 지표 | 의미 |
|---|---|
| `delta_loss` | selective fake quantization 후 loss 증가량 |
| `delta_loss_per_quantized_token` | 양자화된 token 1개당 평균 loss 증가량 |

## 정량 결과

### Layer-wise 유의성

| 분석 지표 | FDR<0.05 유의 layer 수 |
|---|---:|
| Activation `max_abs` | 14/28 |
| Activation `zmax` | 14/28 |
| Attention `received_attn_future_mean` | 9/28 |

Activation 지표는 여러 layer에서 JOSA token이 NON-JOSA token보다 큰 값을 보이는 패턴이 나타났습니다. 반면, future-normalized attention 기준에서는 조사 token이 일관된 attention sink로 작동한다는 증거가 상대적으로 약했습니다.

### Last-layer activation summary

| metric        |   mean_josa |   mean_nonjosa |   diff_josa_minus_nonjosa |   ratio_josa_over_nonjosa |
|:--------------|------------:|---------------:|--------------------------:|--------------------------:|
| max_abs       | 22.8158     |    20.211      |                2.60481    |                  1.12888  |
| zmax          | 17.4452     |    15.3589     |                2.08631    |                  1.13584  |
| norm          | 88.5749     |    89.7427     |               -1.16773    |                  0.986988 |
| outlier_ratio |  0.00587008 |     0.00459177 |                0.00127831 |                  1.27839  |
| topk_share    |  0.11262    |     0.0887332  |                0.0238865  |                  1.26919  |

### Last-layer attention summary

| metric                    |   mean_josa |   mean_nonjosa |   diff_josa_minus_nonjosa |   ratio_josa_over_nonjosa |
|:--------------------------|------------:|---------------:|--------------------------:|--------------------------:|
| received_attn_future_mean |   0.0121006 |      0.0174    |              -0.00529939  |                  0.695437 |
| received_attn_total_share |   0.020122  |      0.0195045 |               0.000617534 |                  1.03166  |
| attn_sink_z_future        |  -0.328123  |     -0.302077  |              -0.0260465   |                  1.08622  |

### Matched fake activation quantization summary

| condition                  |   mean_delta_loss |   mean_delta_loss_per_token |   median_delta_loss_per_token |   mean_n_quantized_tokens |
|:---------------------------|------------------:|----------------------------:|------------------------------:|--------------------------:|
| matched_josa_only          |         0.116162  |                   0.0393621 |                     0.0174157 |                   4.10811 |
| matched_nonjosa_only       |         0.0993166 |                   0.0293878 |                     0.022597  |                   4.10811 |
| matched_random_non_special |         0.087442  |                   0.0251106 |                     0.0129963 |                   4.10811 |
| all_non_special            |         0.418957  |                   0.0242134 |                     0.0159535 |                  21.7333  |


선택적 activation fake quantization 결과, `matched_josa_only` 조건의 token당 평균 손실 증가는 `0.039362`이고, `matched_nonjosa_only` 조건은 `0.029388`입니다. 비율 기준으로는 조사 token이 약 `1.339x` 더 민감하게 나타났습니다.


## 핵심 Figure

### 1. Layer-wise activation peak 차이

JOSA token과 NON-JOSA token의 `max_abs` 차이를 layer별로 비교한 결과입니다.

![Layer-wise JOSA minus NON-JOSA activation max_abs](/content/korean-josa-activation-sink/assets/figures/layerwise_josa_minus_nonjosa_activation_max_abs.png)

### 2. Layer-wise normalized outlier score 차이

`zmax`는 layer별 activation scale 차이를 보정한 outlier score입니다.

![Layer-wise JOSA minus NON-JOSA activation zmax](/content/korean-josa-activation-sink/assets/figures/layerwise_josa_minus_nonjosa_activation_zmax.png)

### 3. Attention sink 비교

Causal attention의 위치 편향을 줄이기 위해 future query에서 받은 attention만 정규화해 비교했습니다.

![Layer-wise JOSA minus NON-JOSA future-normalized attention](/content/korean-josa-activation-sink/assets/figures/layerwise_josa_minus_nonjosa_attention_future_mean.png)

### 4. Selective fake activation quantization

동일 개수의 JOSA token과 NON-JOSA token을 선택해 hidden state만 fake quantization한 뒤 loss 증가량을 비교했습니다.

![Matched fake quantization delta loss per token](/content/korean-josa-activation-sink/assets/figures/fake_quant_delta_loss_per_token_by_layer.png)

### 5. Last-layer POS별 activation 분포

![Last layer activation distribution by POS](/content/korean-josa-activation-sink/assets/figures/last_layer_activation_max_abs_boxplot.png)

## 코드 구조

| 경로 | 설명 |
|---|---|
| `configs/default.yaml` | 실험 기본 설정 |
| `scripts/colab_setup.sh` | Colab 환경 설치 스크립트 |
| `scripts/run_full_experiment.py` | 전체 실험 실행 entry point |
| `src/josa_sink/morph_align.py` | KoNLPy 형태소 분석 및 subword alignment |
| `src/josa_sink/activation.py` | activation metric 수집 |
| `src/josa_sink/attention.py` | attention sink metric 수집 |
| `src/josa_sink/quant_probe.py` | matched fake activation quantization |
| `src/josa_sink/stats.py` | layer-wise 통계 검정 |
| `src/josa_sink/visualization.py` | figure 생성 |
| `src/josa_sink/reporting.py` | summary/report 저장 |

## 재현 방법

Colab A100 환경 기준:

    bash scripts/colab_setup.sh
    pip install -e .
    python scripts/run_full_experiment.py --config configs/default.yaml

또는 entry point를 사용할 수 있습니다.

    josa-sink-run --config configs/default.yaml

## 선별된 결과물 정책

본 레포지토리는 모든 raw result를 push하지 않습니다. GitHub에는 연구 내용을 빠르게 이해하는 데 필요한 핵심 산출물만 포함합니다.

| 포함 | 제외 |
|---|---|
| 핵심 layer-wise figure | token-level raw activation table |
| 핵심 quantization figure | token-level raw attention table |
| summary CSV | 전체 zip 결과물 |
| final summary JSON | model checkpoint / cache |
| README 내 핵심 표 | 대용량 중간 산출물 |

전체 결과가 필요한 경우 Colab 실행 후 생성되는 `/content/korean_josa_sink_results.zip`을 별도로 보관하는 것을 권장합니다.

## 해석상 주의점

1. KoNLPy Okt는 native character offset을 제공하지 않으므로 형태소 span은 순차 문자열 매칭으로 복원했습니다.
2. Hugging Face subword token은 명사와 조사를 하나의 token으로 병합할 수 있습니다. 따라서 본 연구의 JOSA label은 엄밀히 말해 `pure 조사 token`이 아니라 `조사와 character span이 겹치는 token`입니다.
3. Attention sink 여부는 raw attention이 아니라 future-query-normalized received attention 기준으로 판단해야 합니다.
4. Fake activation quantization은 분석용 perturbation이며, 실제 배포용 weight quantization과 동일하지 않습니다.
5. 본 결과는 단일 한국어 LLM과 제한된 문장 집합에 대한 분석이므로, 후속 연구에서는 더 큰 corpus와 다른 한국어 LLM에서 재현 검증이 필요합니다.

## Citation / Related Concepts

이 레포지토리는 다음 연구 흐름과 연결됩니다.

- Attention sink in streaming language models
- Massive activation outliers in LLMs
- SmoothQuant and activation-aware quantization
- Register token / internal computation carrier interpretation in Vision Transformers
- Korean morphological structure and subword tokenization analysis

## License

MIT License

---

Last updated: 2026-05-05 22:39:38
