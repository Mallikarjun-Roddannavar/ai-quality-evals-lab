# Read a result before reading the framework

The application answers or acts. An evaluator checks a declared criterion.
An execution error means the measurement failed; a quality failure means the
measurement ran and found bad behavior. A trace helps explain either result.

## Foundations and judges

- Exact match: `800 INR` vs `800 INR` gives 1. `800 INR` vs `800 inr` gives 0.
- Normalized match case-folds and collapses whitespace, so the second pair gives
  1. It also accepts a lowercased case-sensitive shop ID: a false pass.
- A supported paraphrase may fail both string checks. A deterministic price
  parser measures a narrower fact, but unfamiliar grammar needs review.
- The Python judge asks for support and completeness in one call. DeepEval's
  faithfulness metric extracts and checks claims; its score measures supported
  claims, not task completeness. Neither is a verified human oracle.
- Swapping A/B answer order can expose judge inconsistency. One swapped pair
  cannot establish a population bias rate. Verbosity must not be treated as truth.
- Human labels start empty. Export the blind review file, have two people label
  independently, import their records and adjudicate disagreements. Code-generated
  control labels cannot stand in for this process.

## Reading the numbers

One success in two independent Bernoulli trials is 50%; the 95% Wilson interval
is approximately [9.45%, 90.55%]. Two cases provide little precision. Repeating
the same case does not create two independent task samples. Paired comparisons
must match case IDs, starting state and configuration, and include errors and
ungraded coverage. Cost remains unknown when it has not been measured.

Traditional tests establish contracts and invariants. AI evaluation adds sampled
behavior, contextual quality, uncertain judgments and resource trade-offs. Keep
the deterministic tests: they independently enforce ownership and action rules.

## Follow the execution

`CLI -> scenario -> application -> permitted tool -> sandbox -> evaluator -> report`

The model receives public facts or authorized tool results. It never receives
the scenario's expected result. Evaluators inspect actual outputs and state.
OpenTelemetry connects the execution steps; Phoenix stores and displays traces.
A successful trace export says nothing about answer correctness.

| Topic | Tiny worked example / question to ask |
|---|---|
| 01 Foundations | One unchanged answer out of two gives exact match 1/2; normalization can change the denominator's passes, not the task truth. |
| 02 Deterministic | Catalog 800 INR vs parsed 900 INR fails; an unfamiliar correct sentence needs review. |
| 03 Human | Controlled reviewer labels `[correct, incorrect]` vs `[correct, correct]` give agreement 1/2, chance agreement 1/2, kappa 0. These are unit-test labels, not real reviewers. |
| 04 Judges | A plausible wrong price may sound fluent. Check support separately from style, completeness and order bias. |
| 05 Benchmarking | Score the same three products on two models. Inspect exact-format and Python value checks can disagree without either arithmetic being wrong. |
| 06 Chat | “The backpack” then “What does it cost?” tests a reference across turns. Grade each turn and the whole task separately. |
| 07 RAG | Ranked `[irrelevant, returns]` with one relevant document gives recall 1, precision 1/2, MRR 1/2 and binary nDCG about 0.631. A retrieved document can still be misused. |
| 08 Tools | A native `get_product` call with a valid ID is action evidence. Text saying “I looked it up” is insufficient. |
| 09 MCP | Initialization/discovery/schema/call success establishes protocol behavior; the domain must still reject another customer's order. |
| 10 Skills | Two correct selections out of two selected gives precision 1; finding both required selections gives recall 1. Five routing controls do not establish robust language understanding. |
| 11 Single agent | Inspect -> cancel -> verify. Grade final state, permission decisions and response claims independently. |
| 12 Multi-agent | If single uses 2 calls and multi uses 3 with the same successful outcome, the extra call is 50% overhead for that case; no quality benefit was observed there. |
| 13 Security | Report blocked attacks alongside permitted work. Refusing everything can be safe but useless. Authored tool requests do not measure general jailbreak resistance. |
| 14 Reliability | Keep repeated trials in the same task cluster when bootstrapping paired differences. Curated synthetic controls do not justify population accuracy claims. |
| 15 Operations | Attempts `[10 ms failed, 30 ms succeeded]` total 40 ms; nearest-rank p50 is 10 and p95 is 30. Missing token usage and energy cost stay unknown. |
| 16 Regression | A controlled 900-INR variant should fail the 800-INR catalog gate. Gate failure is the desired outcome of this test. |

## Framework comparison boundaries

- DeepEval's real faithfulness metric uses a custom bounded local judge. It checks
  support through multiple requests; the Python judge asks a different, combined
  support/completeness question. Compare diagnosed cases, not raw-score equivalence.
- Ragas ID recall checks the same document-ID overlap as Python recall. It does
  not evaluate generated claims. MRR/nDCG examine rank; citation IDs examine links.
- LangChain constructs the matched chat/RAG pipeline. LangGraph runs the state
  machine and conditional transitions. Neither supplies the correctness oracle.
- Inspect supplies a real task/solver/scorer/log runner through the approved
  provider adapter. Its exact-format grade is narrower than semantic correctness.
- promptfoo runs the local security assertions against actual Python domain
  operations. No hosted red-team generation or unknown external target is used.

See the official [DeepEval custom-model contract](https://deepeval.com/guides/guides-using-custom-llms),
[Ragas recall definitions](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/),
[Inspect provider API](https://inspect.aisi.org.uk/extensions-model-api.html) and
[promptfoo custom providers](https://www.promptfoo.dev/docs/providers/custom-api/).

## Before saying you understand a topic

Predict one unseen shop result, run it, explain one false pass or failure, and
write a case that exposes a limitation. Keep your reasoning separate from the
assistant's generated explanation. Implementation evidence does not certify
your understanding; review it at your own pace.
