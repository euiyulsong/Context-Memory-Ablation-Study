# LongMemEval Memory Ablation 2차 실험 결과

## 1. 실험 목적

이번 실험은 `recursive_typed_list`를 기준으로 다음 3가지를 분리해서 검증했다.

1. **Recursive rewrite vs Python append**
2. **Prompt-level length constraint의 영향**
3. **생성 후 mechanical truncation의 영향**

### 공통 설정

```text
Dataset              : LongMemEval-S cleaned
Examples             : 50
Model                : qwen/qwen3.5-35b-a3b
Endpoint             : OpenRouter
Workers              : 50
Summary update cap   : 2
Writer hard max      : 768 tokens
Prompt length limit  : 200 words
Post truncate budget : ~128 tokens
Primary metric       : normalized Exact Match
Diagnostics          : Token F1, Relaxed EM
```

---

# 2. 결과

| Method                            |         EM |         F1 | Relaxed EM | Generated Tokens | QA Tokens |
| --------------------------------- | ---------: | ---------: | ---------: | ---------------: | --------: |
| **python_append_typed**           | **0.2800** | **0.3114** | **0.3200** |           1217.1 |    1217.1 |
| recursive_typed_no_limit_full     |     0.1600 |     0.1916 |     0.2000 |            778.2 |     778.2 |
| recursive_typed_prompt_limit_full |     0.1600 |     0.2266 |     0.2400 |            760.5 |     760.5 |
| recursive_typed_no_limit_truncate |     0.0200 |     0.0345 |     0.0400 |            778.2 |     128.0 |

Paired comparison:

```text
Recursive vs Python append
Recursive only correct : 3
Append only correct    : 9
Both correct           : 5

Prompt limit vs No limit
Prompt-limit only      : 4
No-limit only          : 4
Both correct           : 4

Truncate vs Full
Truncate only          : 0
Full only              : 7
Both correct           : 1
```

---

# 3. 가장 큰 결과: 단순 Python append가 가장 좋았다

이번 결과에서 가장 눈에 띄는 부분은:

```text
python_append_typed
EM = 0.28

recursive_typed_no_limit_full
EM = 0.16
```

이다.

즉 현재 50-example pilot에서는 **recursive consolidation보다 append-only memory가 훨씬 좋은 결과**를 보였다.

정답 개수로 보면:

```text
Append    : 14 / 50
Recursive :  8 / 50
```

이다.

그리고 paired 결과에서도:

```text
Append만 맞춤    = 9
Recursive만 맞춤 = 3
```

이므로 단순히 동일한 문제를 우연히 맞춘 게 아니라, **append 방식에서만 recover되는 질문이 더 많았다.**

---

# 4. 왜 append가 더 좋았나?

이 결과는 이상하다기보다는 사실 **현재 실험 설정에서는 충분히 나올 수 있는 결과**다.

가장 큰 이유는 **information retention**이다.

## Recursive 방식

Recursive memory는:

```text
Chunk 1
   ↓
Summary 1

Summary 1 + Chunk 2
   ↓
Summary 2
```

처럼 두 번째 단계에서 이전 memory를 다시 생성한다.

따라서 모델이 중요하지 않다고 판단한 정보는 사라질 수 있다.

예를 들어 첫 summary에:

```text
[EVENT] Alice visited Italy in 2019.
[PREFERENCE] Alice likes Italian food.
[RELATION] Alice traveled with Sarah.
```

가 있었다고 하자.

두 번째 chunk에 새로운 정보가 대량으로 들어오면 최종 summary가:

```text
[PREFERENCE] Alice likes Italian food.
[RELATION] Alice often travels with Sarah.
```

정도로 압축되면서:

```text
Italy
2019
```

같은 세부 정보가 없어질 수 있다.

LongMemEval은 이런 세부 factual recall을 많이 요구하기 때문에 이 손실이 바로 EM 하락으로 연결된다.

---

## Python append 방식

반면 append 방식은:

```text
Chunk 1 → fact extraction 1
Chunk 2 → fact extraction 2

Python:
fact1 + fact2
```

이다.

이미 추출된 memory를 다시 LLM에게 rewrite시키지 않는다.

따라서:

```text
한 번 추출된 fact가 이후 summary 단계에서 사라지는 현상
```

이 없다.

즉 이번 결과는 상당히 자연스럽게:

> **Long-term memory에서는 consolidation 품질보다 fact retention이 더 중요할 수 있다.**

는 가설을 지지한다.

---

# 5. 그런데 append가 더 많은 token을 썼다

이 부분이 매우 중요하다.

```text
python_append_typed       : 1217 tokens
recursive_no_limit        : 778 tokens
```

Append는 recursive보다 약 **56% 더 많은 memory**를 사용했다.

즉 현재 결과만 가지고:

> append 알고리즘이 recursive보다 본질적으로 우수하다.

라고 결론 내리면 안 된다.

현재 비교에는 두 효과가 동시에 들어 있다.

```text
1. Append라서 정보를 덜 잃음
2. Append가 더 많은 token을 QA에 제공함
```

그래서 append가 높은 성능을 기록한 이유 중 일부는 단순히 **더 많은 정보를 보존했기 때문**일 수 있다.

사실 long-term memory 관점에서는 이것도 append의 실질적 장점이기는 하지만, 순수 algorithm ablation은 아니다.

---

# 6. 다음에 반드시 해야 할 공정한 실험

이제 가장 중요한 실험은:

```text
Recursive : QA memory 778 tokens
Append    : QA memory 778 tokens
```

처럼 **동일 token budget**으로 비교하는 것이다.

예:

```text
append_full
vs
append_budget_778

recursive_full
```

이렇게 보면:

```text
Append 우위가
"rewrite를 안 해서" 생긴 것인지

아니면
"더 많은 tokens를 넣어서" 생긴 것인지
```

를 분리할 수 있다.

가장 깔끔하게는 모든 method를:

```text
512
768
1024
```

세 가지 memory budget에서 비교하면 좋다.

---

# 7. Prompt에 길이 제한을 주는 것은 거의 영향이 없었다

결과:

```text
No prompt limit
EM = 0.16
F1 = 0.1916

200-word prompt limit
EM = 0.16
F1 = 0.2266
```

EM은 완전히 동일했다.

Paired 결과도:

```text
prompt-limit만 맞음 = 4
no-limit만 맞음     = 4
both                 = 4
```

이다.

즉 50개 기준으로는:

> **Prompt에 "200 words 이하로 써라"라고 명시하는 것 자체는 성능을 떨어뜨리지 않았다.**

오히려 F1은:

```text
0.1916 → 0.2266
```

으로 조금 높아졌다.

하지만 memory 길이를 보면:

```text
No limit     : 778.2 tokens
Prompt limit : 760.5 tokens
```

차이가 거의 없다.

이게 중요하다.

즉 현재 `200 words` instruction이 실제로 강한 compression을 만든 것은 아니다.

따라서 이번 실험에서 확인한 것은 엄밀히 말하면:

> **약한 prompt-level brevity instruction은 성능에 큰 영향을 주지 않았다.**

정도다.

---

# 8. Prompt length constraint 자체가 나쁜 것은 아니다

이번 결과를 보면:

```text
EM:
no-limit = 0.16
limited  = 0.16
```

이므로 적어도 현재 설정에서는:

```text
"memory를 짧게 유지하라"
```

는 instruction을 넣었다고 정확도가 떨어지지는 않았다.

오히려 semantic compression을 LLM에게 맡기는 방식은 비교적 안전할 가능성이 있다.

왜냐하면 모델은 어떤 내용을 남길지 의미적으로 판단할 수 있기 때문이다.

---

# 9. 반면 생성 후 truncate는 매우 나빴다

이 결과가 이번 실험에서 가장 명확하다.

```text
Full recursive memory
EM = 0.16

128-token truncate
EM = 0.02
```

즉:

```text
8 / 50 correct
→
1 / 50 correct
```

으로 떨어졌다.

F1도:

```text
0.1916 → 0.0345
```

로 크게 하락했다.

Paired 결과는 더 명확하다.

```text
truncate만 맞음 = 0
full만 맞음     = 7
both            = 1
```

즉 truncate해서 새롭게 맞힌 질문은 단 하나도 없고, 기존에 맞히던 7개를 잃었다.

---

# 10. 왜 truncate가 이렇게 나쁜가?

현재 truncation은 semantic information을 보지 않고:

```text
head 35%
+
tail 65%
```

만 유지한다.

예를 들어 memory가:

```text
Fact A
Fact B
Fact C
Fact D
Fact E
Fact F
Fact G
```

이고 정답 근거가:

```text
Fact C
```

라면 mechanical truncation은 그냥 제거할 수 있다.

LLM summarization은 적어도:

```text
이 정보가 나중에 중요할 것 같은가?
```

를 판단하지만 Python substring truncation은 그런 판단 자체가 없다.

따라서:

```text
Semantic compression
≠
Mechanical truncation
```

이라는 결과가 매우 명확하게 나왔다.

---

# 11. 이번 실험에서 가장 강한 결론

현재 50-example 결과에서는 다음 세 가지가 가장 중요하다.

## 결론 1. Append-only memory가 recursive rewrite보다 강했다

```text
Append    EM 0.28
Recursive EM 0.16
```

이는 **recursive summary 과정에서 factual detail이 손실되는 현상**과 잘 맞는다.

특히 LongMemEval과 같은 factual long-term memory benchmark에서는:

```text
좋은 문장으로 다시 쓰는 것
```

보다:

```text
이미 추출한 fact를 잃지 않는 것
```

이 더 중요할 수 있다.

---

## 결론 2. Prompt-level length constraint는 크게 해롭지 않았다

```text
No limit     EM 0.16
Prompt limit EM 0.16
```

따라서 제한된 memory budget을 맞춰야 한다면:

```text
LLM에게 semantic compression을 요청
```

하는 것은 가능해 보인다.

다만 이번에는 실제 token 감소가 작았기 때문에 더 강한 compression ratio에서도 확인해야 한다.

---

## 결론 3. 생성 후 naive truncation은 매우 위험하다

```text
Full     EM 0.16
Truncate EM 0.02
```

현재 결과에서는 거의 붕괴 수준이다.

따라서:

```text
memory 너무 길다
→ 뒤에서 그냥 자른다
```

는 전략은 피하는 것이 좋다.

---

# 12. 이전 실험과 합치면 더 재미있는 결론

이전 실험에서는:

```text
recursive_typed_list
>
recursive paragraph / flat list
```

가 나왔다.

이번에는:

```text
python append typed
>
recursive typed list
```

가 나왔다.

두 결과를 합치면 현재까지 가장 좋은 가설은:

```text
typed atomic representation은 좋다
+
하지만 계속 recursive rewrite하는 것은 오히려 fact loss를 만든다
```

이다.

즉 현재 실험이 지지하는 architecture는 오히려:

```text
Conversation
    ↓
Typed atomic fact extraction
    ↓
Append-only memory store
    ↓
Retrieval / selection
    ↓
QA
```

에 가깝다.

반대로:

```text
Conversation
    ↓
Summary
    ↓
Summary + new context
    ↓
Summary rewrite
    ↓
Summary rewrite ...
```

처럼 repeatedly rewriting memory는 factual long-term memory에는 손실이 누적될 수 있다.

---

# 13. 그렇다고 무한 append가 답은 아니다

Append-only에는 명확한 문제가 있다.

예:

```text
[LOCATION] User lives in Seoul.
[LOCATION] User moved to Busan.
[LOCATION] User moved to Tokyo.
```

처럼 contradiction / stale information이 계속 쌓일 수 있다.

또 memory size도 이번 결과에서 이미 나타났다.

```text
Append    : 1217 tokens
Recursive : 778 tokens
```

대화가 길어질수록 append memory는 계속 증가한다.

따라서 실서비스에서는 단순 infinite append보다는:

```text
Typed atomic append
       ↓
deduplication
       ↓
relevance retrieval
       ↓
필요할 때만 consolidation
```

이 더 현실적인 구조다.

즉 매 turn마다 전체 memory를 rewrite하는 게 아니라:

```text
write는 append 중심
read는 retrieval 중심
consolidation은 가끔
```

이 훨씬 유망해 보인다.

---

# 14. 다음 실험 추천

이번 결과에서 가장 먼저 해야 할 것은 **token-budget controlled append vs recursive**다.

예:

```text
python_append_512
recursive_512

python_append_768
recursive_768

python_append_1024
recursive_1024
```

그리고 가능하면 append memory를 무식하게 truncate하지 말고:

```text
BM25 / embedding retrieval
```

로 budget 안에 relevant facts만 선택해야 한다.

가장 중요한 비교는:

```text
Recursive typed summary, 768 tokens

vs

Append typed memory
→ retrieval
→ top relevant facts 768 tokens
```

이다.

만약 이 조건에서도 append + retrieval이 이기면 상당히 강한 결과다.

---

# 최종 결론

이번 pilot에서는 **Python append typed memory가 가장 높은 성능**을 기록했다.

```text
Append    EM = 0.28
Recursive EM = 0.16
```

이는 recursive summarization이 memory를 압축하는 대신 **factual details를 잃을 수 있다는 trade-off**를 보여준다.

다만 append가 약 1.56배 많은 memory tokens를 사용했기 때문에, 현재 결과만으로 recursive와 append의 순수한 알고리즘 우열을 확정할 수는 없다.

반면 post-generation truncation의 악영향은 매우 명확하다.

```text
Recursive full     EM = 0.16
Recursive truncate EM = 0.02
```

따라서 현재 결과가 가장 강하게 시사하는 설계는:

```text
Atomic typed memory
+ Append
+ Retrieval
+ Semantic compression when necessary
```

이며,

```text
Repeated recursive rewrite
```

나

```text
Naive post-hoc truncation
```

은 factual long-term memory에서는 조심해서 사용해야 한다.
