# Append Typed Memory: 2 Chunks vs 5 Chunks 결과 요약

## 1. 실험 목적

LongMemEval 대화 history를 typed atomic memory로 추출할 때, history를 **2개의 큰 chunk로 나누는 방식**과 **5개의 큰 chunk로 나누는 방식** 중 어떤 것이 원래 LongMemEval QA 성능에 더 좋은지 비교했다.

두 방식 모두 recursive rewrite는 사용하지 않고:

```text
chunk → typed fact extraction → Python append
```

만 수행했다.

즉 이번 실험은 **chunk granularity의 효과만 비교하는 ablation**이다.

---

## 2. 실험 방식

### `python_append_typed_c2`

```text
LongMemEval history
        ↓
2개의 큰 contiguous chunk
        ↓
각 chunk별 typed fact extraction
        ↓
Python append
        ↓
LongMemEval 원래 question으로 QA
```

Memory write 호출 수:

```text
2회
```

### `python_append_typed_c5`

```text
LongMemEval history
        ↓
5개의 작은 contiguous chunk
        ↓
각 chunk별 typed fact extraction
        ↓
Python append
        ↓
LongMemEval 원래 question으로 QA
```

Memory write 호출 수:

```text
5회
```

---

# 3. 결과

| Method                       |         EM |         F1 | Relaxed EM | Avg Memory Tokens | Writes |
| ---------------------------- | ---------: | ---------: | ---------: | ----------------: | -----: |
| `python_append_typed_c2`     |     0.3400 |     0.3833 |     0.4000 |            1328.7 |      2 |
| **`python_append_typed_c5`** | **0.5800** | **0.6202** | **0.6400** |            3341.1 |      5 |

50개 질문 기준으로 정답 개수는:

```text
c2: 17 / 50
c5: 29 / 50
```

즉 EM 기준으로:

```text
0.34 → 0.58
```

로 **+24%p** 상승했다.

상대적으로는 약 **1.7배 높은 EM**이다.

---

# 4. Paired 결과

같은 50개 질문을 두 방식에 모두 평가한 paired comparison:

```text
c5 only correct : 17
c2 only correct : 5
both correct    : 12
both wrong      : 16
```

이를 표로 보면:

|           | c2 정답 |  c2 오답 |
| --------- | ----: | -----: |
| **c5 정답** |    12 | **17** |
| **c5 오답** |     5 |     16 |

가장 중요한 부분은:

```text
c5에서만 맞은 질문 = 17
c2에서만 맞은 질문 = 5
```

이다.

따라서 단순히 몇 문제의 우연한 차이라기보다, **5-chunk extraction이 더 많은 질문에서 필요한 memory를 보존한 방향성이 상당히 뚜렷하다.**

---

# 5. F1에서도 같은 결과

EM뿐 아니라 F1에서도:

```text
c2 F1 = 0.3833
c5 F1 = 0.6202
```

로 큰 차이가 나타났다.

즉 c5의 EM 상승이 단순한 answer-formatting 차이 때문이라고 보기 어렵다.

Relaxed EM 역시:

```text
c2 = 0.40
c5 = 0.64
```

로 같은 방향이다.

따라서 세 지표가 모두:

```text
c5 >> c2
```

를 지지한다.

---

# 6. 왜 5 chunks가 훨씬 좋았나?

가장 가능성이 높은 이유는 **한 번의 extraction task가 쉬워졌기 때문**이다.

## 2 chunks

전체 history의 절반을 한 번에 LLM에게 준다.

```text
매우 긴 chunk
↓
수많은 사건 / 인물 / 날짜 / preference / state change
↓
typed fact extraction
```

입력이 너무 길기 때문에 LLM이 모든 fact를 추출하지 못하고 일부를 생략할 가능성이 높다.

즉:

```text
large input
→ extraction recall 저하
```

가 발생할 수 있다.

---

## 5 chunks

각 extraction prompt에 들어가는 history 양이 훨씬 작아진다.

```text
history 1/5 → extract
history 1/5 → extract
history 1/5 → extract
...
```

각 호출에서 처리해야 할 사건 수가 줄기 때문에 중요한 fact를 놓칠 확률도 낮아진다.

그리고 이 실험은 recursive summary가 아니라 **Python append**이기 때문에:

```text
chunk 1에서 뽑은 fact
```

가 chunk 2~5 처리 과정에서 다시 rewrite되거나 삭제되지 않는다.

따라서:

```text
작은 chunk
+
independent extraction
+
lossless append
```

가 상당히 강하게 작동한 것으로 보인다.

---

# 7. 이전 recursive 결과와 연결

이전 실험에서는 recursive 방식에서 chunk 수가 많아질 경우 오히려 불리할 가능성이 있었다.

Recursive는:

```text
Memory 1 + chunk 2
→ rewrite

Memory 2 + chunk 3
→ rewrite

...
```

이기 때문에 chunk 수를 늘리면 **전체 memory rewrite 횟수도 증가**한다.

반면 append는:

```text
chunk 1 → facts
chunk 2 → facts
chunk 3 → facts
...
↓
Python concat
```

이므로 chunk를 늘려도 이전 memory를 다시 생성하지 않는다.

따라서 현재까지 결과를 종합하면:

```text
Append:
작은 chunk가 유리할 가능성 높음

Recursive:
작은 chunk의 extraction 이점
vs
rewrite 횟수 증가에 따른 forgetting
```

이라는 차이가 보인다.

---

# 8. 하지만 c5는 memory를 훨씬 많이 사용했다

중요한 trade-off가 있다.

```text
c2 memory = 1328.7 tokens
c5 memory = 3341.1 tokens
```

즉 c5가 약:

```text
3341 / 1329 ≈ 2.5배
```

더 많은 memory를 사용했다.

그리고 write call도:

```text
c2 = 2 calls
c5 = 5 calls
```

로 2.5배이다.

따라서:

> 5 chunks가 더 효율적이다

라고 말하기는 어렵다.

정확한 해석은:

> **5 chunks가 더 많은 extraction 호출과 더 큰 memory를 사용하는 대신 factual recall과 downstream QA 정확도를 크게 높였다.**

이다.

---

# 9. 왜 memory가 2.5배 커졌나?

Append 방식에서는 각 chunk가 독립적으로 memory를 생성한다.

따라서 5 chunks에서는:

```text
Fact A
Fact B
Fact C
```

가 여러 chunk에서 비슷하게 반복되거나, 동일 사건의 세부 정보가 더 많이 살아남을 수 있다.

예를 들어 c2에서는:

```text
[EVENT] User traveled to Japan in 2023.
```

정도로 합쳐질 것이 c5에서는:

```text
[EVENT] User traveled to Japan in 2023.
[LOCATION] User visited Tokyo.
[RELATION] User traveled with Alex.
[PREFERENCE] User enjoyed Japanese food.
```

처럼 더 세밀하게 추출될 가능성이 있다.

즉 memory 증가가 단순 중복뿐 아니라 **higher recall / finer granularity**의 결과일 수도 있다.

---

# 10. 이번 결과에서 가장 강한 결론

현재 50-example 결과에서는:

```text
python_append_typed_c5
```

가 `c2`보다 명확하게 우수하다.

핵심 수치는:

```text
EM
0.34 → 0.58

F1
0.383 → 0.620

Relaxed EM
0.40 → 0.64
```

이다.

그리고 paired comparison에서도:

```text
c5 only = 17
c2 only = 5
```

로 c5의 우위가 뚜렷하다.

따라서:

> **Append-only typed memory에서는 history를 소수의 매우 큰 chunk로 처리하기보다 더 작은 여러 chunk로 나누어 독립적으로 fact extraction하는 것이 memory recall에 매우 중요할 수 있다.**

는 가설을 상당히 강하게 지지한다.

---

# 11. 다만 아직 남은 confound

현재 c2와 c5는 chunk 수뿐 아니라 결과적으로:

```text
memory token 수
write call 수
```

도 다르다.

따라서 현재 비교는 정확히:

```text
2-chunk extraction system
vs
5-chunk extraction system
```

이지,

```text
동일 memory budget에서 순수 chunk size만 비교
```

한 것은 아니다.

---

# 12. 다음으로 가장 중요한 실험

이제 반드시 해볼 만한 비교는:

```text
c2 full memory
vs
c5 full memory
vs
c5 + retrieval/token budget
```

이다.

예를 들어 c5가 만든 3341-token memory에서 질문과 관련 있는 memory만 retrieval해서:

```text
1329 tokens
```

로 제한한다.

그러면:

```text
c2:
1329 tokens → EM 0.34

c5 + retrieval:
1329 tokens → ?
```

가 된다.

만약 c5 + retrieval이 동일 token budget에서도 여전히 크게 높다면:

> **성능 상승은 단순히 더 많은 token을 사용해서가 아니라, smaller-chunk extraction이 더 높은-quality memory bank를 만들었기 때문이다.**

라는 훨씬 강한 결론을 얻을 수 있다.

---

# 최종 결론

이번 실험에서는 **5개의 작은 chunk로 나누어 typed fact를 독립 추출한 뒤 Python append하는 방식이 2개의 큰 chunk 방식보다 크게 우수했다.**

```text
c2: EM 0.34
c5: EM 0.58
```

이는 append-only memory에서 chunk를 작게 만들면 각 LLM extraction task가 쉬워지고, 더 많은 factual detail을 보존할 수 있기 때문으로 해석할 수 있다.

다만 c5는:

```text
약 2.5배 많은 memory tokens
약 2.5배 많은 write calls
```

를 사용했다.

따라서 현재까지 가장 유망한 구조는:

```text
Conversation
      ↓
여러 개의 비교적 작은 chunk
      ↓
Typed atomic fact extraction
      ↓
Python append
      ↓
Retrieval / selection
      ↓
질문에 필요한 memory만 LLM에 제공
```

이다.

즉 다음 단계의 핵심은 **`5-chunk append + retrieval`이 동일한 read-time token budget에서도 2-chunk append보다 좋은지 확인하는 것**이다.
