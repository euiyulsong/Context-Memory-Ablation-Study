
# SQuAD New-Question + Memory Chunk Ablation 결과 분석

## 1. 실험 목적

이번 실험은 LongMemEval 대화로 만든 long-term memory가 있는 상태에서, 갑자기 전혀 다른 도메인의 **SQuAD context + question**이 들어왔을 때 성능이 얼마나 유지되는지 확인했다.

동시에 memory를 만들 때 기존 대화를:

* **2개의 큰 chunk**
* **5개의 큰 chunk**

로 나누는 것이 각각 어떤 영향을 주는지 비교했다.

비교한 memory 방식은:

```text
python_append_typed
recursive_typed_prompt_limit
```

이다.

---

# 2. 공통 실험 설정

```text
Dataset for memory    : LongMemEval-S cleaned
New QA dataset        : SQuAD v1.1 dev
Evaluation examples   : 50
Model                 : qwen/qwen3.5-35b-a3b
Endpoint              : OpenRouter
Workers               : 20
Writer max tokens     : 768
Recursive word limit  : 100 words
Primary metric        : normalized Exact Match
Secondary metric      : Token F1
```

SQuAD 질문을 평가할 때는 반드시 현재 context를 함께 제공했다.

입력 구조는:

```text
OLD LONG-TERM MEMORY
+
CURRENT SQuAD PASSAGE
+
CURRENT SQuAD QUESTION
→ answer
```

이다.

따라서 이번 실험은 long-term memory 자체의 recall이 아니라:

> **과거 memory가 unrelated한 새로운 current-context QA를 얼마나 방해하는가**

를 측정한다.

---

# 3. 실험 결과

| Method                     |        EM |        F1 | ΔEM vs SQuAD only |       ΔF1 | Avg Memory Tokens | Writes |
| -------------------------- | --------: | --------: | ----------------: | --------: | ----------------: | -----: |
| **squad_only**             | **0.980** | **0.988** |             0.000 |     0.000 |                 0 |      0 |
| **python_append_typed_c5** | **0.980** | **0.988** |         **0.000** | **0.000** |            3405.7 |      5 |
| python_append_typed_c2     |     0.960 |     0.968 |            -0.020 |    -0.020 |            1350.5 |      2 |
| recursive_typed_limit_c2   |     0.960 |     0.968 |            -0.020 |    -0.020 |             671.6 |      2 |
| recursive_typed_limit_c5   |     0.940 |     0.964 |            -0.040 |    -0.024 |             660.8 |      5 |

---

# 4. 가장 중요한 결과

## 4.1 SQuAD only baseline이 거의 perfect

```text
SQuAD only
EM = 0.98
F1 = 0.988
```

즉 이번 50문제는 현재 모델에게 상당히 쉬운 subset이었다.

정답 개수로 보면:

```text
49 / 50 correct
```

이다.

그래서 memory를 붙였을 때의 성능 감소는 곧바로 **memory interference / distraction**으로 볼 수 있다.

---

# 5. 가장 좋은 memory 방식은 `python_append_typed_c5`

의외로:

```text
python_append_typed_c5
EM = 0.98
F1 = 0.988
```

로 SQuAD-only baseline과 완전히 동일했다.

즉 평균 약:

```text
3406 tokens
```

의 unrelated old memory가 붙어 있었는데도 current SQuAD passage를 읽는 능력이 전혀 떨어지지 않았다.

이 결과는 꽤 흥미롭다.

```text
No memory:
EM 0.98

3.4k-token append memory:
EM 0.98
```

즉 단순히:

> old memory가 길수록 distraction이 증가한다

는 가설은 이번 결과에서는 지지되지 않는다.

오히려 가장 긴 memory를 가진 arm이 가장 높은 성능을 기록했다.

---

# 6. Append: 5 chunks가 2 chunks보다 좋았다

Append 방식 내부 비교:

```text
2 chunks:
EM = 0.96
F1 = 0.968
Memory = 1350 tokens

5 chunks:
EM = 0.98
F1 = 0.988
Memory = 3406 tokens
```

paired comparison도:

```text
5 chunks only correct = 1
2 chunks only correct = 0
both correct          = 48
```

이다.

즉 50개 중 아주 작은 차이이지만 방향은:

```text
append c5 >= append c2
```

였다.

---

# 7. 왜 append에서는 5 chunks가 더 좋을 수 있나

이건 이전 실험 결과와 연결하면 이해가 된다.

Append 방식은:

```text
chunk → typed fact extraction
```

만 수행하고 이전 memory를 다시 rewrite하지 않는다.

### 2 chunks

매우 큰 chunk 하나를 LLM에 넣는다.

```text
history 절반
→ extract facts
```

하나의 extraction prompt에 너무 많은 사건과 fact가 들어가기 때문에 LLM이 일부 정보를 누락할 수 있다.

### 5 chunks

각 입력이 더 작아진다.

```text
history 1/5 → extract
history 1/5 → extract
...
```

따라서 chunk별 extraction task가 쉬워진다.

그리고 결과는 Python으로 단순 append하기 때문에 이미 추출한 fact는 다시 손실되지 않는다.

즉:

```text
smaller extraction chunks
+
no recursive forgetting
```

조합이 잘 작동한 것으로 볼 수 있다.

---

# 8. 그러나 append c5가 더 많은 token을 사용함

주의해야 할 점은:

```text
append c2 = 1350 tokens
append c5 = 3406 tokens
```

이라는 것이다.

5 chunks가 약 **2.5배 많은 memory**를 생성했다.

왜냐하면 chunk마다 독립적으로 typed fact를 뽑아서 붙이기 때문에:

* 중복 fact
* 비슷한 정보
* 동일 사건의 다른 표현

이 누적되기 때문이다.

즉 현재 결과에서:

> 5 chunks가 2 chunks보다 좋다

라고 바로 결론 내리기보다는:

> 5 chunks는 extraction recall이 높아지는 대신 memory size가 크게 증가한다.

가 더 정확하다.

---

# 9. Recursive에서는 반대로 5 chunks가 더 나빴다

Recursive 방식:

```text
2 chunks:
EM = 0.96
F1 = 0.968

5 chunks:
EM = 0.94
F1 = 0.964
```

paired 결과:

```text
5 chunks only correct = 0
2 chunks only correct = 1
both correct          = 47
```

즉 작은 차이지만 방향은:

```text
recursive c2 > recursive c5
```

였다.

---

# 10. 왜 recursive에서는 chunk가 많아질수록 불리한가

Recursive 방식은 매 chunk마다:

```text
previous memory
+
new chunk
→ rewrite complete memory
```

한다.

따라서 5 chunks면:

```text
rewrite #1
rewrite #2
rewrite #3
rewrite #4
rewrite #5
```

가 발생한다.

반면 2 chunks면:

```text
rewrite #1
rewrite #2
```

뿐이다.

매 rewrite마다:

* 일부 fact 제거
* 표현 변경
* temporal relationship 단순화
* semantic drift

가 발생할 가능성이 있다.

따라서 recursive에서는:

```text
chunk를 작게 만들면 한 번의 summarization은 쉬워지지만,
rewrite 횟수가 많아져 누적 forgetting이 증가
```

하는 trade-off가 생긴다.

이번 결과는 후자의 비용이 조금 더 컸다는 방향이다.

---

# 11. Append vs Recursive: 2 chunks에서는 차이가 없었다

2-chunk 조건:

```text
append c2
EM = 0.96

recursive c2
EM = 0.96
```

paired comparison:

```text
append only    = 0
recursive only = 0
both correct   = 48
```

즉 SQuAD interference 측면에서는 완전히 같은 결과였다.

하지만 memory 크기는:

```text
append    = 1350 tokens
recursive = 672 tokens
```

이다.

즉 이 조건에서는 recursive가 약 절반의 token을 사용하면서 같은 성능을 냈다.

이건 recursive의 장점이다.

> 새로운 unrelated QA에 대한 robustness만 본다면 2-chunk recursive는 append보다 훨씬 압축 효율적이었다.

---

# 12. 5 chunks에서는 Append가 Recursive보다 좋았다

```text
append c5
EM = 0.98

recursive c5
EM = 0.94
```

paired comparison:

```text
append only correct    = 2
recursive only correct = 0
both                    = 47
```

즉 이번 실험에서 가장 큰 method 차이다.

이 결과는 이전 LongMemEval memory QA 실험과도 같은 방향이다.

이전에는:

```text
python append typed > recursive typed
```

였고,

이번 SQuAD new-question에서도 5-chunk 조건에서는:

```text
python append typed > recursive typed
```

였다.

따라서 반복적인 recursive rewrite가 어느 정도 불리하다는 신호가 누적되고 있다.

---

# 13. 하지만 이번 SQuAD 실험은 memory quality를 직접 평가하는 실험이 아님

중요한 구분이다.

이번 질문은 SQuAD context에서 답을 찾는다.

즉 old memory가:

```text
완벽하게 좋은 memory인지
많은 정보를 잃은 memory인지
```

자체는 SQuAD QA 정답에 별로 중요하지 않다.

우리가 측정한 것은:

```text
old memory가 current context QA를 방해하느냐?
```

이다.

따라서:

```text
append c5 = 0.98
```

이 나왔다고 해서 append c5가 최고의 **memory recall architecture**라고 결론 낼 수는 없다.

다만:

> memory가 크더라도 current-context instruction이 명확하면 새 질문을 거의 방해하지 않을 수 있다.

는 결과는 얻을 수 있다.

---

# 14. 가장 의외인 점: Memory token 수와 interference가 비례하지 않았다

각 method의 memory 크기:

```text
recursive c5 :  661 tokens → EM 0.94
recursive c2 :  672 tokens → EM 0.96
append c2    : 1351 tokens → EM 0.96
append c5    : 3406 tokens → EM 0.98
```

만약 단순 context pollution만 문제라면:

```text
memory가 길수록 SQuAD가 더 나빠져야 함
```

이라고 예상할 수 있다.

하지만 실제 결과는 정반대였다.

가장 긴:

```text
append c5 = 3406 tokens
```

가 baseline과 동일했다.

따라서 이번 범위에서는 **memory 길이 자체보다 memory의 내용/표현 구조나 sampling noise가 더 중요**할 가능성이 있다.

또한 system prompt에서:

```text
CURRENT PASSAGE is authoritative
```

라고 매우 명확하게 줬기 때문에 모델이 old memory와 current context를 잘 분리했을 가능성도 높다.

---

# 15. 100-word recursive 제한 해석 시 주의

이번 recursive prompt는:

```text
keep the complete updated memory at or below 100 words
```

라고 설정했다.

그런데 측정된 memory가:

```text
~660-670 approximate tokens
```

수준이다.

이는 몇 가지 가능성을 의미한다.

```text
1. 모델이 100-word constraint를 제대로 따르지 않음
2. approx_tokens가 문자 기반 추정이라 실제 tokenizer와 차이가 큼
3. 출력 형식/tag 등이 token 수를 많이 소비함
```

따라서 이번 결과를:

> recursive memory는 정확히 100 words였다

라고 해석하면 안 된다.

실제 output word count도 별도로 측정하는 게 좋다.

---

# 16. 통계적으로는 아직 차이가 작음

50 examples에서:

```text
0.98 = 49/50
0.96 = 48/50
0.94 = 47/50
```

이다.

즉 전체 성능 차이는 사실상:

```text
49 correct
48 correct
47 correct
```

수준이다.

따라서 이번 pilot에서:

> 5-chunk append가 확실히 superior하다

라고 강하게 말하기는 어렵다.

현재 결론은:

> 방향성은 보였으나 larger sample에서 검증이 필요하다.

가 적절하다.

---

# 17. 현재까지 두 실험을 합친 해석

기존 LongMemEval memory QA 실험:

```text
python_append_typed
EM = 0.28

recursive
EM = 0.16
```

이번 unrelated SQuAD QA:

```text
append c5
EM = 0.98

recursive c5
EM = 0.94
```

두 실험을 함께 보면 상당히 흥미로운 패턴이 나온다.

## Append

```text
장점:
- 기존 fact를 rewrite하지 않아 정보 보존성이 높음
- chunk를 작게 나누면 extraction recall 증가 가능
- large memory가 있어도 new-context QA를 크게 방해하지 않았음

단점:
- memory size가 매우 빠르게 증가
- 중복 / stale information 누적 가능
```

## Recursive

```text
장점:
- memory를 상당히 작게 유지
- 2 chunks에서는 append와 동일한 SQuAD robustness

단점:
- rewrite 횟수가 증가하면 누적 information loss 가능
- 5 chunks에서는 성능이 조금 하락
```

---

# 18. 현재로서는 2 chunks vs 5 chunks 중 어느 쪽이 좋은가?

하나의 답은 없다.

### Python append라면

현재 결과:

```text
5 chunks > 2 chunks
```

방향이다.

이유:

```text
더 작은 chunk
→ extraction task 쉬움
→ fact recall 증가
→ 기존 fact rewrite 없음
```

다만 memory 크기가 너무 증가한다.

### Recursive라면

현재 결과:

```text
2 chunks > 5 chunks
```

방향이다.

이유:

```text
5 chunks
→ rewrite 횟수 증가
→ cumulative forgetting / drift 가능성
```

따라서 현재 실험이 시사하는 것은:

> **최적 chunk 수는 memory update strategy에 따라 달라질 수 있다.**

이다.

---

# 19. 실서비스 관점에서 가장 유망한 구조

현재까지의 실험만 보면 가장 유망한 architecture는:

```text
Conversation
    ↓
moderately small chunks
    ↓
typed atomic fact extraction
    ↓
append-only store
    ↓
dedup / retrieval
    ↓
current context + relevant memories
    ↓
LLM
```

이다.

즉:

```text
write:
append 중심

read:
retrieve 중심

consolidation:
필요할 때만 수행
```

구조다.

매 chunk마다 전체 memory를 recursive rewrite하는 것보다, atomic memory를 보존하고 read-time에 필요한 것만 가져오는 방식이 현재 결과에는 더 잘 맞는다.

---

# 20. 다음 실험

이번 결과 다음으로 가장 가치 있는 실험은:

```text
append c2
append c5

→ 둘 다 retrieval해서 최종 700 tokens만 QA에 제공
```

vs

```text
recursive c2
recursive c5
→ 약 700 tokens
```

이다.

즉:

```text
Append + Retrieval @ same token budget
vs
Recursive summary @ same token budget
```

을 비교해야 한다.

현재 append c5는:

```text
3406 tokens
```

를 사용하므로 recursive의:

```text
~670 tokens
```

와 직접적인 memory efficiency 비교는 어렵다.

만약 append memory에서 relevant facts만 골라 **670 tokens로 줄인 뒤에도 성능이 유지**된다면:

> recursive summarization보다 atomic append + retrieval이 더 좋은 long-term memory architecture다.

라는 훨씬 강한 결론을 얻을 수 있다.

---

# 최종 결론

이번 50-example pilot에서는:

```text
New SQuAD QA robustness:

squad_only              0.98
append 5 chunks         0.98
append 2 chunks         0.96
recursive 2 chunks      0.96
recursive 5 chunks      0.94
```

였다.

가장 중요한 관찰은 세 가지다.

1. **3.4k-token의 unrelated append memory가 있어도 SQuAD 성능은 baseline과 동일했다.**
2. **Append에서는 5 chunks가 2 chunks보다 약간 좋았지만 memory 크기가 크게 증가했다.**
3. **Recursive에서는 5 chunks가 오히려 2 chunks보다 나빴으며, 반복적인 rewrite에 의한 cumulative forgetting 가능성과 일치한다.**

따라서 현재까지의 결과는:

> **작은 chunk로 atomic facts를 추출한 뒤 append하고, retrieval을 통해 read-time budget을 제어하는 방식**

이 가장 유망하다는 방향을 보여준다.

다만 50개에서는 0.98/0.96/0.94가 각각 49/48/47개 정답 차이에 불과하므로, chunk 수 자체에 대한 최종 결론은 최소 200 examples 이상에서 다시 확인하는 것이 좋다.
