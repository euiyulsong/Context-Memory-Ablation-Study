# LongMemEval Context Memory 요약 방식 Ablation

## 1. 실험 목적

긴 대화의 이전 context를 어떤 형태로 memory로 압축할 때 downstream QA 성능이 가장 좋은지 비교했다.

비교한 핵심 요소는 다음과 같다.

* Memory 없음
* 최근 turn 유지
* Paragraph 형태의 recursive summary
* List 형태의 recursive summary
* Type이 명시된 structured list
* 기존 memory를 직접 갱신하는 typed memory

### 공통 설정

```text
Dataset         : LongMemEval-S cleaned
Evaluation      : first 50 examples
Model           : qwen/qwen3.5-35b-a3b
Endpoint        : OpenRouter
Temperature     : 0.0
Memory budget   : 1024 approx tokens
Summary updates : max 2
Workers         : 8
Primary metric  : normalized Exact Match
Diagnostics     : Token F1, Relaxed EM
```

---

# 2. 각 Memory 방식

동일한 대화가 아래와 같다고 가정한다.

```text
2025-01
User: I live in Seoul.
User: I usually drink iced Americano.
User: I'm preparing for a machine learning interview.

2025-03
User: I moved to Busan last month.
User: I don't drink coffee anymore.
User: My interview is next Friday.
```

## 2.1 `current_only`

이전 대화를 전혀 제공하지 않는다.

```text
Memory:
(no previous memory)
```

질문:

```text
Where does the user live?
```

모델은 이전 정보를 알 수 없기 때문에 정상적인 동작이라면:

```text
I don't know
```

을 출력한다.

즉 순수한 **no-memory baseline**이다.

---

## 2.2 `last_k`

최근 K개의 raw turn을 그대로 유지한다.

예:

```text
[2025-03]
USER: I moved to Busan last month.
USER: I don't drink coffee anymore.
USER: My interview is next Friday.
```

요약을 하지 않는다는 장점이 있다.

하지만 relevant fact가 오래전에 등장했다면 최근 K turn 밖으로 밀려나기 때문에 찾을 수 없다.

즉:

```text
장점
- summary hallucination 없음
- 원문 정보 유지

단점
- 오래된 정보 손실
- irrelevant recent context가 많음
```

---

# 2.3 `recursive_paragraph`

이전 memory와 새로운 context를 합쳐 **하나의 자연어 문단**으로 계속 다시 작성한다.

예:

```text
The user originally lived in Seoul but moved to Busan in
February 2025. They previously drank iced Americano but
no longer drink coffee. They are preparing for a machine
learning interview scheduled for next Friday.
```

업데이트 방식은 대략:

```text
Summary(t-1) + New Context
             ↓
      New Summary(t)
```

이다.

### 특징

장점:

* 자연스럽고 문맥 관계를 잘 보존할 수 있음
* 여러 사실 간 관계를 문장으로 표현 가능
* token compression이 쉬움

단점:

* rewrite할 때 기존 세부 정보가 사라질 수 있음
* 한 문장 안에 여러 fact가 합쳐짐
* 특정 fact retrieval에는 atomic representation보다 불리할 수 있음

---

# 2.4 `recursive_list`

같은 recursive rewrite를 수행하지만 결과를 flat list로 만든다.

```text
- The user originally lived in Seoul.
- The user moved to Busan in February 2025.
- The user previously drank iced Americano.
- The user no longer drinks coffee.
- The user is preparing for a machine learning interview.
- The interview is next Friday.
```

Paragraph보다 하나의 fact가 분리되어 있기 때문에 retrieval이나 factual lookup에는 유리할 수 있다.

하지만 category나 relation에 대한 별도의 structure는 없다.

예를 들어:

```text
- User moved to Busan.
- Interview is next Friday.
```

둘은 완전히 동일한 수준의 item으로 저장된다.

---

# 2.5 `recursive_typed_list`

각 atomic fact에 의미적인 type을 추가한다.

```text
[LOCATION] The user originally lived in Seoul.
[LOCATION] The user moved to Busan in February 2025.

[PREFERENCE] The user previously drank iced Americano.
[PREFERENCE] The user no longer drinks coffee.

[WORK] The user is preparing for a machine learning interview.
[TEMPORAL] The interview is next Friday.
```

사용한 type은 다음과 같은 형태이다.

```text
IDENTITY
PREFERENCE
WORK
RELATION
LOCATION
ACTIVITY
PLAN
EVENT
TEMPORAL
OTHER
```

여전히 recursive rewrite 방식이지만, 각각의 정보를 명시적인 semantic unit으로 유지한다.

### 기대 효과

질문이:

```text
Where does the user currently live?
```

이라면 모델은:

```text
[LOCATION]
```

관련 entry를 쉽게 찾을 수 있다.

질문이:

```text
When is the interview?
```

라면:

```text
[TEMPORAL]
```

entry가 명확하게 분리되어 있다.

---

# 2.6 `update_typed`

표현 자체는 typed list와 비슷하지만 업데이트 정책이 다르다.

기존:

```text
[LOCATION] User lives in Seoul.
[PREFERENCE] User drinks iced Americano.
```

새로운 대화:

```text
I moved to Busan.
I don't drink coffee anymore.
```

단순히 append하면:

```text
[LOCATION] User lives in Seoul.
[LOCATION] User lives in Busan.

[PREFERENCE] User drinks iced Americano.
[PREFERENCE] User doesn't drink coffee.
```

처럼 서로 충돌하는 memory가 남는다.

`update_typed`는 이를 적극적으로 갱신해서:

```text
[LOCATION] User currently lives in Busan.
[LOCATION] User previously lived in Seoul.

[PREFERENCE] User no longer drinks coffee.
```

처럼 만드는 것을 목표로 한다.

즉 핵심 차이는:

```text
recursive_typed_list
= 전체 memory를 다시 잘 요약

update_typed
= 기존 state를 찾아 ADD / UPDATE / REMOVE
```

이다.

---

# 3. 실험 결과

| Method                   |         EM |         F1 | Relaxed EM | Avg Memory Tokens | Avg Latency / Example |
| ------------------------ | ---------: | ---------: | ---------: | ----------------: | --------------------: |
| **recursive_typed_list** | **0.0800** | **0.1198** | **0.1400** |             186.5 |               37.33 s |
| recursive_paragraph      |     0.0400 |     0.0681 |     0.0400 |             218.7 |               38.82 s |
| update_typed             |     0.0400 |     0.0581 |     0.0600 |             195.8 |               46.43 s |
| recursive_list           |     0.0400 |     0.0431 |     0.0400 |             199.0 |               39.82 s |
| current_only             |     0.0000 |     0.0031 |     0.0000 |               5.0 |                0.97 s |
| last_k                   |     0.0000 |     0.0031 |     0.0000 |            1029.0 |                0.92 s |

---

# 4. 핵심 결과

## 4.1 Structured typed list가 가장 좋은 성능

가장 좋은 결과는:

```text
recursive_typed_list
EM          = 0.0800
F1          = 0.1198
Relaxed EM  = 0.1400
```

이다.

Paragraph와 flat list는 모두:

```text
EM = 0.04
```

였으므로 strict EM 기준으로 typed list가 약 **2배 높은 결과**를 기록했다.

절대적인 sample 수가 50개이기 때문에 통계적으로 강한 결론을 내리기는 어렵지만, F1에서도 같은 경향이 나타난다.

```text
recursive_typed_list  0.1198
recursive_paragraph   0.0681
update_typed          0.0581
recursive_list        0.0431
```

따라서 현재 결과에서는 단순 우연한 EM formatting 차이보다 **typed representation 자체가 relevant fact를 더 잘 보존하거나 읽게 만든 가능성**이 있다.

---

# 4.2 List로 바꾸기만 해서는 좋아지지 않았다

흥미로운 결과는:

```text
recursive_paragraph
EM = 0.04
F1 = 0.0681

recursive_list
EM = 0.04
F1 = 0.0431
```

이다.

즉:

```text
paragraph → flat bullet list
```

로 바꾼 것만으로는 성능이 개선되지 않았다.

오히려 F1은 paragraph가 더 높았다.

이는 단순히:

> 하나의 fact를 한 줄씩 나누면 memory가 좋아진다

는 가설을 현재 결과가 지지하지 않는다는 의미다.

### 가능한 해석

Paragraph는 여러 사건 사이의 관계를 자연스럽게 유지할 수 있다.

예를 들어:

```text
The user moved to Busan after accepting a new job.
```

은 하나의 문장 안에:

```text
move
location
temporal relation
causal relation
job
```

이 같이 존재한다.

Flat list로 바꾸면서:

```text
- User moved to Busan.
- User got a new job.
```

처럼 관계가 약해질 수 있다.

따라서 단순 listification 자체가 반드시 장점은 아니다.

---

# 4.3 핵심은 `List`보다 `Typed List`일 가능성이 높음

가장 중요한 비교는 다음이다.

```text
recursive_list       EM 0.04 / F1 0.0431
recursive_typed_list EM 0.08 / F1 0.1198
```

두 방식 모두 list 형태인데 type을 붙인 경우 성능이 크게 상승했다.

따라서 현재 결과는:

```text
Paragraph vs List
```

보다 오히려

```text
Unstructured vs Structured Memory
```

차이가 더 중요할 가능성을 보여준다.

즉:

```text
- User moved to Busan.
```

보다:

```text
[LOCATION] User moved to Busan.
```

이 downstream QA에서 훨씬 읽기 쉬운 representation이 될 수 있다.

---

# 4.4 Typed list는 더 짧으면서도 가장 정확했다

평균 memory token은:

```text
recursive_paragraph  : 218.7
recursive_list       : 199.0
recursive_typed_list : 186.5
update_typed         : 195.8
```

였다.

즉 typed list는 **가장 짧은 summary memory를 사용하면서 가장 높은 QA 성능**을 기록했다.

이는 상당히 중요한 결과다.

단순히 더 많은 정보를 넣어서 정확도가 증가한 것이 아니다.

오히려:

```text
더 적은 tokens
+
더 높은 EM/F1
```

이다.

따라서 현재 실험에서는 **memory organization / structure가 token quantity보다 중요했을 가능성**이 있다.

---

# 4.5 `update_typed`가 예상과 달리 typed recursive보다 낮음

처음 가설에서는 update 방식이 stale memory나 contradiction을 관리하기 때문에 강할 것으로 예상할 수 있었다.

하지만 결과는:

```text
recursive_typed_list
EM = 0.08
F1 = 0.1198

update_typed
EM = 0.04
F1 = 0.0581
```

이다.

즉 현재는 recursive typed가 명확하게 우세했다.

가능한 이유는 몇 가지가 있다.

### 이유 1. Update prompt가 지나치게 공격적으로 정보를 제거했을 가능성

`update_typed`는:

```text
ADD
UPDATE
REMOVE stale information
deduplicate
```

를 동시에 수행한다.

따라서 실제로는 update 대상이 아닌 historical fact까지 stale한 것으로 판단해서 지웠을 수 있다.

LongMemEval에서는 과거 정보 자체를 묻는 질문도 있기 때문에:

```text
현재 상태만 잘 저장
```

하는 것이 항상 좋은 memory는 아니다.

---

### 이유 2. update가 더 어려운 LLM task

Recursive summarization은:

```text
A + B를 다시 정리해줘
```

라는 비교적 단순한 생성 task이다.

반면 update memory는:

```text
기존 item과 새 item 비교
→ 같은 attribute인가?
→ contradiction인가?
→ history를 보존해야 하나?
→ delete 해야 하나?
```

를 판단해야 한다.

따라서 동일한 LLM이라도 update 정책에서 실수가 더 발생할 수 있다.

---

### 이유 3. 두 번의 update만 허용한 영향

이번 실험에서는:

```text
max summary updates = 2
```

로 제한했다.

전체 history가 매우 긴데 이를 두 개 큰 chunk로 나누기 때문에 `update_typed`에서는 한 번에 매우 많은 state change를 처리해야 한다.

따라서 원래 의도한:

```text
small new event
→ existing memory update
```

보다 훨씬 어려운 task가 됐다.

이 부분은 별도 ablation이 필요하다.

---

# 4.6 `last_k`가 사실상 실패

가장 놀라운 baseline 결과는:

```text
last_k

EM = 0
F1 = 0.0031
Memory = 1029 tokens
```

이다.

거의 memory budget 전체를 사용하면서도 memory 없는 `current_only`와 사실상 동일한 성능이다.

```text
current_only F1 = 0.0031
last_k       F1 = 0.0031
```

이는 LongMemEval에서 중요한 정보가 **최근 turn에 존재한다고 가정하면 안 된다**는 것을 보여준다.

즉:

```text
최근 context 많이 넣기
```

보다:

```text
오래된 history에서 중요 정보를 선택 / 압축해서 유지하기
```

가 훨씬 중요하다.

특히 비교하면:

```text
last_k
1029 tokens → EM 0.00

recursive_typed
186 tokens → EM 0.08
```

이다.

약 **1/5 수준의 memory token으로 더 좋은 정확도**를 기록했다.

이 결과는 memory compression의 필요성을 상당히 잘 보여준다.

---

# 4.7 Memory가 없는 것보다는 summary가 명확히 낫다

```text
current_only
EM = 0
F1 = 0.0031
```

반면 모든 recursive memory 방식에서는 일정 수준의 성능이 나타났다.

따라서 적어도 이 50개 sample에서는:

```text
long-term memory information 자체가 필요한 질문
```

이 많이 포함되어 있다는 것을 확인할 수 있다.

---

# 5. Latency

평균 latency:

```text
current_only             0.97 s
last_k                   0.92 s

recursive_typed_list    37.33 s
recursive_paragraph     38.82 s
recursive_list          39.82 s
update_typed            46.43 s
```

Memory 생성이 들어가면 latency가 약 40초/example 수준으로 증가했다.

하지만 progress bar에서 보이는 wall-clock은 worker=8 병렬 처리 덕분에:

```text
50 examples

recursive_paragraph : 4m 24s
recursive_list      : 4m 52s
recursive_typed     : 4m 10s
update_typed        : 5m 35s
```

수준으로 줄어들었다.

`update_typed`가 가장 느리면서 성능도 가장 높지 않았기 때문에 현재 결과에서는 **accuracy-latency tradeoff 측면에서도 불리하다.**

반면 `recursive_typed_list`는:

```text
가장 높은 EM
가장 높은 F1
가장 적은 summary tokens
summary 방식 중 상대적으로 가장 낮은 latency
```

를 동시에 기록했다.

현재 실험의 가장 명확한 winner이다.

---

# 6. 현재 결과의 결론

50 examples 기준으로는 다음 순서로 해석할 수 있다.

```text
recursive_typed_list
        ↓
recursive_paragraph
        ≈
update_typed
        ≈
recursive_list
        ↓
last_k
        ≈
current_only
```

가장 중요한 발견은 세 가지다.

### 1. 단순 list는 paragraph보다 좋지 않았다.

```text
Paragraph:
EM 0.04 / F1 0.068

List:
EM 0.04 / F1 0.043
```

따라서 formatting만 bullet로 바꾸는 것은 도움이 되지 않았다.

### 2. Typed structure를 추가하면 큰 개선이 있었다.

```text
Flat List:
EM 0.04 / F1 0.043

Typed List:
EM 0.08 / F1 0.120
```

현재까지는 **memory의 구조화가 가장 큰 효과를 보였다.**

### 3. 많은 raw context보다 작은 structured memory가 훨씬 효율적이었다.

```text
last_k:
1029 tokens
EM 0

typed summary:
186 tokens
EM 0.08
```

따라서 long-term memory에서는 단순 context retention보다 **selective compression + structure**가 중요하다는 결과가 나왔다.

---

# 7. 아직 조심해야 하는 부분

현재 표본은 50개다.

EM 기준으로 보면:

```text
recursive_typed_list = 4 / 50 correct
recursive_paragraph  = 2 / 50 correct
recursive_list       = 2 / 50 correct
update_typed         = 2 / 50 correct
```

이다.

즉 실제 차이는 현재 **정답 2개 차이**에 불과하다.

따라서:

> Typed list가 paragraph보다 2배 좋다

라고 바로 일반화하면 안 된다.

정확한 표현은:

> 50-example pilot에서는 recursive typed-list가 가장 높은 EM/F1을 보였으며, larger evaluation에서 확인할 가치가 있는 명확한 경향이 나타났다.

정도가 적절하다.

---

# 8. 다음 실험

이 결과라면 모든 방법을 500개까지 돌릴 필요는 없다.

다음은 우선:

```text
recursive_paragraph
recursive_list
recursive_typed_list
update_typed
```

네 가지 중 특히:

```text
recursive_paragraph
recursive_typed_list
update_typed
```

를 200개까지 확대하는 것이 좋다.

추천:

```bash
python3 memory_ablation.py \
  --limit 200 \
  --workers 8 \
  --max-summary-updates 2 \
  --methods recursive_paragraph,recursive_typed_list,update_typed
```

그리고 결과가 유지되면 최종으로:

```text
recursive_paragraph
recursive_typed_list
```

두 개를 전체 500에서 비교한다.

---

# 9. 추가로 꼭 해볼 Ablation

현재 결과에서 가장 중요한 추가 질문은:

> Typed list가 왜 좋은가?

이다.

이를 분리하려면 다음 3개를 비교해야 한다.

```text
recursive_list

recursive_typed_list

recursive_typed_list_without_category_description
```

또는:

```text
- User moved to Busan.

[LOCATION] User moved to Busan.

location: User moved to Busan.
```

를 비교하면 된다.

그러면 성능 개선이:

```text
semantic category 자체 때문인지
structured delimiter 때문인지
atomic fact preservation 때문인지
```

를 좀 더 분리할 수 있다.

---

# 최종 요약

현재 pilot 결과에서 가장 추천되는 memory representation은:

```text
Recursive Typed Atomic Memory
```

이다.

형태는:

```text
[LOCATION] ...
[PREFERENCE] ...
[WORK] ...
[RELATION] ...
[TEMPORAL] ...
```

처럼 **한 fact를 하나의 semantic type과 함께 저장하고, 새로운 context가 들어올 때 전체 memory를 최대한 보존하면서 recursive하게 rewrite하는 방식**이다.

현재 결과에서는 이 방식이:

```text
가장 높은 EM
가장 높은 F1
가장 높은 Relaxed EM
가장 적은 memory tokens
낮은 편의 summary latency
```

를 동시에 보였다.

따라서 현재 단계에서는 단순 paragraph나 flat list보다 **typed structured summary를 우선 후보로 두는 것이 가장 합리적이다.**
