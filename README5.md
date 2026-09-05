# Fixed-size vs Semantic Segment 실험 결과 분석

## 1. 실험 설정

LongMemEval oracle에서 총 50개 샘플을 question type별로 균형 있게 뽑아 비교했다. 비교 대상은 `fixed_2`, `fixed_10`, `segment` 세 가지이며, segmentation은 `Qwen3.5-9B`, memory extraction과 QA는 `Qwen3.5-35B-A3B`를 사용했다. Memory extraction은 모든 방법에서 동일한 untyped 방식이고, reasoning은 껐다. 

```text
fixed_2
→ 2 exchanges씩 묶어서 memory extraction

fixed_10
→ 10 exchanges씩 묶어서 memory extraction

segment
→ LLM이 semantic/topic boundary 탐지
→ segment별 memory extraction
```

샘플 분포는 `knowledge-update 9`, `multi-session 9`, 나머지 네 유형은 각각 8개로 총 50개였다. 

---

## 2. 전체 결과

| Method     |        EM | Correct | Avg Units | Avg Memories | Avg Calls | Avg Tokens |  Avg Time |
| ---------- | --------: | ------: | --------: | -----------: | --------: | ---------: | --------: |
| `fixed_2`  | **0.380** |   19/50 |      5.10 |        33.74 |      6.10 |      8,536 |    16.77s |
| `fixed_10` | **0.380** |   19/50 |  **1.70** |    **18.36** |  **2.70** |  **6,538** | **7.78s** |
| `segment`  | **0.380** |   19/50 |      3.46 |        27.44 |      6.10 | **13,225** |    15.47s |

세 방법 모두 raw EM은 정확히 **0.38**로 동일했다. 그런데 비용은 큰 차이가 났다. `fixed_10`이 가장 적은 memory unit, 가장 적은 memory 수, 가장 적은 API call, 가장 적은 token, 가장 낮은 latency를 기록했다. 

따라서 **이 실험의 raw EM만 기준으로 하면 semantic segmentation을 추가할 이유가 없다.**

```text
Quality: fixed_2 = fixed_10 = segment

Cost:
fixed_10 << fixed_2 <≈ segment
```

특히 segment는 `fixed_10` 대비 token을 약 **2.0배** 사용했다.

---

## 3. 그런데 세 방법이 완전히 같은 답을 맞춘 것은 아니다

Paired 결과가 중요하다.

```text
fixed_2 vs fixed_10
fixed_2_only  = 2
fixed_10_only = 2
both_correct  = 17
both_wrong    = 29

fixed_2 vs segment
fixed_2_only  = 1
segment_only  = 1
both_correct  = 18
both_wrong    = 30

fixed_10 vs segment
fixed_10_only = 1
segment_only  = 1
both_correct  = 18
both_wrong    = 30
```



즉 `fixed_2`와 `fixed_10`은 총 EM은 같지만 **4개 문제에서 서로 승패가 뒤집혔다.**

반면 `segment`와 각 fixed 방식은 거의 똑같다.

```text
fixed_2 ↔ segment
48/50 문제에서 correctness 동일

fixed_10 ↔ segment
48/50 문제에서 correctness 동일
```

그래서 현재 데이터에서는 semantic segmentation이 memory representation을 실질적으로 크게 바꾸지는 못한 것으로 보인다.

---

# 4. Question type별로 보면 조금 다르다

| Question type             |  fixed_2 | fixed_10 |  segment |
| ------------------------- | -------: | -------: | -------: |
| Knowledge Update          |     .333 |     .333 |     .333 |
| Multi-session             |     .556 |     .556 |     .556 |
| Single-session Assistant  | **.625** |     .500 |     .500 |
| Single-session Preference |     .000 |     .000 |     .000 |
| Single-session User       |     .750 |     .750 | **.875** |
| Temporal Reasoning        |     .000 | **.125** |     .000 |



여기서는 꽤 흥미로운 패턴이 있다.

### `fixed_2`: assistant 정보에 조금 유리

`single-session-assistant`에서:

```text
fixed_2  = 0.625
fixed_10 = 0.500
segment  = 0.500
```

작게 묶으면 특정 assistant 발화의 세부 정보가 덜 희석될 가능성이 있다.

### `segment`: single-session user에서 가장 좋음

```text
fixed_2  = 0.750
fixed_10 = 0.750
segment  = 0.875
```

semantic grouping이 사용자에 관한 사실을 보존하는 데 약간 도움이 됐을 가능성이 있다.

하지만 n=8이라 **1문제 차이**다. 강한 결론은 못 낸다.

### `fixed_10`: temporal reasoning에서 유일하게 1문제 성공

```text
fixed_2  = 0.000
fixed_10 = 0.125
segment  = 0.000
```

큰 context에 같이 넣는 것이 temporal relation을 보존하는 경우가 있을 수 있다.

---

# 5. 그런데 현재 EM에 상당히 큰 평가 문제가 있다

이게 이번 결과에서 **가장 중요합니다.**

현재 EM은 semantic correctness를 상당히 과소평가하고 있다.

예를 들어:

```text
pred = "Under the bed"
gold = "under my bed"
```

이건 의미상 거의 정확하지만 EM=0이다. 

또:

```text
pred = "Triple the amount you paid."
gold = "The painting is worth triple what I paid for it."
```

도 사실상 답은 맞았지만 EM=0이다. 

이런 경우도 있다.

```text
pred = "12"
gold = "The Chiefs played the Jaguars 12 times at Arrowhead Stadium."
```

숫자 답은 맞았는데 EM=0. 

그리고:

```text
pred = "Four"
gold = "4"
```

도 EM=0이다. 

List QA에서도 순서 때문에 틀릴 수 있다.

```text
pred:
JetBlue, American Airlines, United Airlines, Delta

gold:
JetBlue, Delta, United, American Airlines
```

내용 자체는 거의 동일하지만 EM=0이다. 

---

# 6. 특히 Preference 문제 EM=0은 memory 방식 문제가 아니다

세 방법 모두:

```text
single-session-preference = 0.000
```

이다. 

그런데 실제 prediction을 보면 완전 헛답만 하는 것이 아니다.

예를 들어 gold가 길게:

> 이전 Denver 경험, live music, Brandon Flowers 경험을 활용한 추천을 선호...

라고 되어 있는데 prediction은:

```text
Attend a concert at Red Rocks Amphitheater...
```

처럼 **실제 preference를 적용한 recommendation을 직접 생성**한다. 

또 dinner preference에서는 prediction이:

```text
Dishes featuring basil, mint, and cherry tomatoes.
```

이고 gold는:

```text
homegrown cherry tomatoes, basil, mint를 활용하는 저녁을 선호한다...
```

이다. 

즉 여기서는 모델이 질문에 **추천 답변**을 하고 있는데 gold는 **사용자의 preference profile 설명** 형태다.

그래서 이 유형의 `EM=0`은 segmentation 실패라기보다:

> **QA prompt와 LongMemEval preference gold의 expected answer style이 서로 맞지 않는 문제**

가 훨씬 크다.

이건 반드시 고쳐야 한다.

---

# 7. Temporal reasoning도 같은 문제가 있음

예를 들어:

```text
pred = "3"
gold = "3 weeks ago"
```



또:

```text
pred = "4"
gold = "4 days."
```



처럼 숫자 자체는 맞아도 unit을 생략해서 EM=0이 된다.

따라서 temporal reasoning 0.0도 memory construction 자체가 전부 실패했다고 해석하면 안 된다.

---

# 8. 그래도 실제로 틀린 문제도 존재한다

평가 artifact만 있는 것은 아니다.

예를 들어:

```text
gold = every week
pred = every two weeks
```

세 방식 모두 실제로 잘못 기억하거나 잘못 답했다.   

또:

```text
gold = Thrive Market
pred = Walmart
```

도 실제 memory/temporal update failure에 가깝다.   

즉:

```text
평가 포맷 문제
+
실제 memory extraction/reasoning failure
```

두 가지가 같이 섞여 있다.

---

# 9. 현재 실험에서 내릴 수 있는 결론

## 결론 1 — Semantic segmentation은 현재 이득 없음

현재 setup에서는:

```text
fixed_2 = 0.38
fixed_10 = 0.38
segment = 0.38
```

이고 paired 차이도 거의 없다. 따라서 **LLM으로 segmentation하는 추가 비용만큼 downstream QA 성능이 개선되지 않았다.**

---

## 결론 2 — 현재는 fixed_10이 가장 좋은 trade-off

성능이 동일한 상황에서:

```text
fixed_10
tokens = 6.5k
latency = 7.78s

segment
tokens = 13.2k
latency = 15.47s
```

이므로 fixed_10이 압도적으로 효율적이다. 

현재 결과만 놓으면:

> **“semantic boundary detection을 추가하기보다 그냥 적당히 큰 fixed window로 묶는 것이 낫다.”**

라고 볼 수 있다.

---

## 결론 3 — 너무 작은 chunk도 별 이득 없음

`fixed_2`는:

* memory 33.74개
* 8.5k tokens
* 16.77 sec

를 쓰고도 fixed_10과 동일한 EM이다. 

따라서 **더 atomic하게 memory를 많이 뽑는 것 자체가 QA를 개선하지 않았다.**

이건 앞서 네가 한 typed-memory 실험과도 비슷한 패턴이다.

```text
more memories
≠
better downstream QA
```

---

# 10. 다만 이 상태로 논문식 결론 내리면 안 됨

현재 제일 큰 confound는 **EM evaluator**다.

실제로 결과를 보면:

```text
"12" vs "...12 times..."
"Four" vs "4"
"Under the bed" vs "under my bed"
"Triple the amount..." vs "worth triple..."
```

까지 모두 오답 처리된다.

따라서 다음 실험에서는 최소한:

```text
Normalized EM
+ token F1
+ LLM judge
```

세 개를 같이 봐야 한다.

특히 LongMemEval은 preference처럼 **장문의 open-ended answer**가 있기 때문에 EM 하나만으로 비교하는 것은 상당히 부적절하다.

---

# 최종 요약

```text
Raw EM:
fixed_2 = fixed_10 = segment = 0.38

비용:
fixed_10 << fixed_2 ≈ segment

paired 차이:
매우 작음

현재 evidence:
semantic segmentation 이점 없음

가성비:
fixed_10이 가장 좋음

BUT:
현재 EM 평가가 semantic equivalence를 심각하게 놓침
→ preference / temporal / list QA 결과는 특히 신뢰하기 어려움
```

따라서 **현재 실험에서 제일 강하게 말할 수 있는 건 “semantic segmentation이 fixed grouping 대비 추가적인 이득을 보이지 않았고, 비용은 훨씬 더 들었다”**입니다.

다만 **`0.38 = 0.38 = 0.38` 자체는 진짜 품질 동률이라고 단정하면 안 됩니다.** 다음 실험은 segmentation 방식을 더 바꾸기 전에 **LongMemEval 공식 evaluator 또는 LLM-as-judge/F1을 붙이는 게 우선**입니다.
