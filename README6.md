````markdown
# Fixed-20 / Fixed-40 / Semantic Segment 실험 결과 분석

## 1. 실험 설정

LongMemEval oracle에서 50개 샘플을 사용해 다음 세 가지 memory grouping 전략을 비교했다.

- `fixed_20`: 20 exchanges 단위로 묶어서 memory extraction
- `fixed_40`: 40 exchanges 단위로 묶어서 memory extraction
- `segment`: Qwen3.5-9B로 semantic/topic boundary를 탐지한 뒤 segment별 memory extraction
- Memory model: `qwen/qwen3.5-35b-a3b`
- QA model: `qwen/qwen3.5-35b-a3b`
- Memory output: 최대 1000 tokens
- QA output: 최대 16 tokens
- Reasoning: OFF
- Structured JSON: ON
- 50 workers 사용 :contentReference[oaicite:0]{index=0}

샘플은 question type별로 거의 균등하게 구성되었다. :contentReference[oaicite:1]{index=1}

---

## 2. 전체 결과

| Method | EM | Correct | Avg Units | Avg Memories | Avg Calls | Avg Tokens | Avg Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Fixed-20** | **0.400** | **20/50** | 1.68 | 17.76 | 2.68 | 6,518.8 | **7.90s** |
| Fixed-40 | 0.380 | 19/50 | **1.68** | **16.96** | **2.68** | **6,468.0** | 8.08s |
| Segment | 0.340 | 17/50 | 3.48 | 28.24 | 6.14 | 13,323.3 | 21.76s |

:contentReference[oaicite:2]{index=2}

### 핵심 결과

성능은:

`Fixed-20 (0.40) > Fixed-40 (0.38) > Segment (0.34)`

이번 실험에서는 **semantic segmentation이 오히려 가장 낮은 EM**을 기록했다.

비용까지 보면 차이가 더 크다.

`Segment`는 `Fixed-20` 대비:

- 약 **2.04× tokens**
- 약 **2.29× API calls**
- 약 **2.75× latency**

를 사용했는데 EM은 `0.40 → 0.34`로 떨어졌다.

따라서 현재 setup에서는 semantic segmentation이 **성능·비용 모두 열세**다.

---

## 3. Fixed-20이 가장 좋은 sweet spot

`Fixed-20`과 `Fixed-40`은 비용이 거의 동일하다.

```text
Fixed-20: 6518.8 tokens / 7.90 sec
Fixed-40: 6468.0 tokens / 8.08 sec
````

하지만 EM은:

```text
Fixed-20 = 0.40
Fixed-40 = 0.38
```

으로 Fixed-20이 약간 높았다. 

즉 너무 작게 자를 필요도 없지만, **너무 크게 합치는 것도 약간의 정보 손실을 만들 가능성**이 있다.

현재 결과를 granularity 관점에서 정리하면 대략:

```text
너무 작은 fixed
    ↓
memory 수 / cost 증가

적당히 큰 fixed (~20)
    ↓
성능/비용 균형 가장 좋음

너무 큰 fixed (~40)
    ↓
세부 정보 또는 temporal relation 일부 손실 가능

semantic segment
    ↓
추가 segmentation 비용 + 과도한 memory 생성
하지만 성능 개선 없음
```

---

## 4. Paired 결과도 Fixed-20 우세

### Fixed-20 vs Fixed-40

```text
fixed_20_only = 3
fixed_40_only = 2
both_correct  = 17
both_wrong    = 28
```

즉 Fixed-20이 3문제를 추가로 맞히고 Fixed-40은 2문제를 추가로 맞혔다. 

차이는 작지만 Fixed-20 쪽이 미세하게 우세하다.

### Fixed-20 vs Segment

```text
fixed_20_only = 5
segment_only  = 2
both_correct  = 15
both_wrong    = 28
```

이건 더 의미가 있다.

Semantic segmentation으로 좋아진 문제는 2개뿐인데, 오히려 **Fixed-20에서는 맞고 Segment에서 틀린 문제가 5개**였다. 

즉 segment 방식이 단순히 같은 답을 더 비싸게 만드는 정도가 아니라, **일부 질문에서는 실제로 useful context를 깨뜨렸을 가능성**이 있다.

---

# 5. Question type별 차이

| Question type             | Fixed-20 | Fixed-40 |  Segment |
| ------------------------- | -------: | -------: | -------: |
| Knowledge Update          |     .222 | **.333** | **.333** |
| Multi-session             | **.667** |     .444 |     .444 |
| Single-session Assistant  |     .500 |     .500 |     .500 |
| Single-session Preference |     .000 |     .000 |     .000 |
| Single-session User       |     .750 | **.875** |     .750 |
| Temporal Reasoning        | **.250** |     .125 |     .000 |



여기서 가장 흥미로운 건 두 가지다.

## Multi-session: Fixed-20이 크게 우세

```text
Fixed-20 = 0.667
Fixed-40 = 0.444
Segment  = 0.444
```

여러 session의 정보를 연결해야 하는 문제에서는 **20 exchange 정도의 중간 granularity가 유리**했다. 

가능한 해석은:

> 너무 큰 chunk에서는 서로 다른 사건들이 한 memory extraction에 섞이고,
> semantic segment에서는 반대로 서로 연결되어야 할 정보가 다른 segment로 분리될 수 있다.

---

## Temporal reasoning: Segment가 가장 안 좋음

```text
Fixed-20 = 0.250
Fixed-40 = 0.125
Segment  = 0.000
```



이건 semantic segmentation의 중요한 약점일 수 있다.

Topic 기준 segmentation은:

```text
topic A ─────┐
             segment A

topic B ─────┐
             segment B
```

처럼 잘 자를 수 있지만, LongMemEval temporal question은 종종:

```text
예전 state
    ↓
시간 경과
    ↓
새로운 state
```

같은 **cross-segment temporal relation**을 요구한다.

그래서 topic coherence만 보고 자르면 오히려 update/history 연결을 깨뜨릴 가능성이 있다.

---

# 6. Fixed-40이 일부 문제에서는 좋은 이유

Fixed-40은 `single-session-user`에서 가장 높았다.

```text
Fixed-20 = .750
Fixed-40 = .875
Segment  = .750
```



즉 한 session 안의 user fact처럼 단순한 정보라면 큰 chunk를 한 번에 보는 것이 오히려 좋을 수 있다.

하지만 multi-session / temporal에서는 Fixed-20이 더 좋았기 때문에:

> **큰 chunk는 단순 fact QA에는 괜찮지만 관계/시간 정보에는 지나치게 coarse할 수 있다.**

라는 패턴이 보인다.

---

# 7. Segment 방식은 memory도 더 많이 만든다

평균 memory 수:

```text
Fixed-20 = 17.76
Fixed-40 = 16.96
Segment  = 28.24
```



Segment가 Fixed 대비 약 **60% 이상 많은 memory**를 생성했다.

그런데 EM은 더 낮았다.

다시 한번:

> **memory를 더 많이 저장한다고 downstream QA가 좋아지는 것은 아니다.**

오히려 segment별로 따로 extraction하면서 비슷한 사실이 반복되거나, local context만 보고 redundant memory를 뽑을 가능성이 있다.

---

# 8. 실제 qualitative failure에서도 granularity 차이가 보임

예를 들어 `b6019101`에서는:

```text
Fixed-20 → 12  ✗
Fixed-40 → 5   ✓
Segment  → 5   ✓
```

처럼 큰 context가 도움이 되는 경우도 있다. 

반대로 grocery update 문제에서는:

```text
Fixed-20 → Thrive Market ✓
Fixed-40 → Walmart       ✗
Segment  → Publix        ✗
```

으로 Fixed-20만 최신/올바른 state를 보존했다.   

이런 사례는 **중간 granularity가 update relation을 보존하기 가장 좋을 수 있다**는 가설과 잘 맞는다.

---

# 9. 다만 EM 문제는 여전히 큼

이번 결과도 raw EM이 실제 semantic correctness를 과소평가한다.

예:

```text
pred = "Under the bed"
gold = "under my bed"
```

가 틀린 것으로 처리됐다. 

또:

```text
pred = "12"
gold = "The Chiefs played the Jaguars 12 times at Arrowhead Stadium."
```

도 EM=0이다. 

그리고:

```text
pred = "4"
gold = "4 days."
```

도 오답이다. 

따라서 `0.40 vs 0.38 vs 0.34`를 절대적인 QA accuracy로 해석하면 안 된다.

하지만 **모든 방법에 동일 evaluator를 적용했다는 점에서 상대 비교 자체는 어느 정도 의미가 있다.**

---

# 10. 이전 실험까지 합치면 보이는 패턴

지금까지 실험 결과는 대략 이런 형태다.

```text
fixed_2   ≈ 0.38
fixed_10  ≈ 0.38
fixed_20  = 0.40  ← 현재 최고
fixed_40  = 0.38

segment   = 0.34~0.38
```

즉 아주 재미있게도 **fixed size에 꽤 넓은 plateau가 있다.**

```text
2 ------ 10 ------ 20 ------ 40
.38      .38       .40       .38
```

그러므로 적어도 현재 데이터와 extraction 방식에서는:

> 정확한 chunk size가 극도로 중요하지 않고,
> `10~20 exchange` 정도로 적당히 묶으면 충분하다.

라는 결과에 가깝다.

Semantic segmentation은 이 plateau를 넘어서는 성능 향상을 보여주지 못했다.

---

# 결론

## 성능

`Fixed-20 > Fixed-40 > Segment`

## 비용

`Fixed-40 ≈ Fixed-20 << Segment`

## 현재 최적점

**Fixed-20**

현재 실험에서는 semantic segmentation을 별도 LLM call로 수행하는 것보다 **그냥 약 10~20 exchanges 정도를 fixed chunk로 묶어 memory를 추출하는 것이 훨씬 효율적**이다.

특히 Fixed-20은:

* 최고 EM
* segment의 절반 수준 token
* segment의 절반 이하 API call
* segment보다 약 2.7배 빠른 latency

를 달성했다.

### 현재 가장 중요한 결론

> **Semantic segmentation이 항상 memory construction을 개선하는 것은 아니다.**
>
> LongMemEval에서는 topic boundary를 찾는 추가 복잡성보다, 적당한 크기의 fixed context에서 한 번에 memory를 추출하는 방식이 더 단순하고 효과적이었다.

다만 다음에는 **EM 하나만 쓰지 말고 F1 또는 LLM judge를 같이 붙여서 `fixed_10 vs fixed_20 vs segment`만 다시 평가**하는 게 가장 가치가 크다. 현재 결과상 `fixed_2`나 `fixed_40`을 더 파고들 필요성은 상대적으로 낮다.

```
```
