# Source-Fidelity Quality Testing

This document describes the quality benchmark used to compare the fresh Qwen3.8-27B GPTQ checkpoints and sampling profiles on Intel Arc Pro B70.

## Benchmark scope

The suite contains 30 source-grounded prompts covering:

- conflicting documents
- missing information
- timeline ambiguity
- document hierarchy
- source provenance
- unsupported explanations

The goal is not creative answer quality. The goal is whether the model stays inside the supplied evidence and refuses to invent unsupported facts, causes, rules, dates, or provenance.

## Base prompt

The preferred prompt variant is brief-think v2:

```text
Think briefly and directly.

Do not repeatedly reconsider the same conclusion.
Do not explore hypothetical interpretations once the supplied text is sufficient.
Do not suggest examples of missing facts, causes, rules, procedures, or document effects unless those examples appear in the supplied text.
Once the supplied text establishes or fails to establish the answer, stop reasoning and give the final answer.
```

The source-fidelity instructions then require the model to:

- use only supplied text
- avoid unsupported claims about source authority or provenance
- avoid inventing reasons for conflicting figures
- avoid importing outside accounting, tax, legal, or procedural rules
- state unresolved facts when the evidence is insufficient

## Sampling comparison

The original thinking profile used:

```text
temperature=1.0
top_p=0.95
top_k=20
```

Repeated runs showed substantial variation in reasoning length and occasional 2048-token truncation.

The final deterministic profile used:

```text
temperature=0.0
top_p=1.0
top_k=-1
max_tokens=4096
```

The 4096-token value is an output ceiling, not a required generation length.

## Deterministic repeatability

Two back-to-back deterministic G128/MTP4 runs completed:

```text
30/30 prompts
0 errors
0 truncations
```

Run 1:

```text
Generated tokens: 17,936
Total latency:    ~310.5 s
Aggregate rate:   ~57.76 tok/s
```

Run 2:

```text
Generated tokens: 17,720
Total latency:    ~306.8 s
Aggregate rate:   ~57.75 tok/s
```

29 of 30 prompts produced the same output-token count across the two deterministic runs.

One prompt differed in reasoning length.

This was substantially more repeatable than the sampled thinking profile.

## Important throughput note

The approximately 57.75 tok/s result comes from a thinking/source-fidelity workload.

It must not be directly compared with the separate 84.65 tok/s short-context decode benchmark because the workloads, prompts, reasoning behavior, and output patterns differ.

## Known recurring edge case

One conflicting-document prompt contained quantities approximately equivalent to:

```text
Invoice A:       240 units
Receiving Log B: 238 units
Document C:      no quantity
```

The benchmark expected the conflict to remain unresolved.

The model repeatedly answered:

```text
238 units were definitively received.
```

This behavior appeared across:

- G128
- G32
- multiple prompt variants
- sampled decoding
- deterministic decoding

This makes a simple GPTQ group-size explanation unlikely.

## Benchmark-design caveat

The edge case may itself contain semantic ambiguity.

An invoice quantity and a receiving-log quantity do not necessarily represent the same real-world field.

For example:

```text
240 units shipped
238 units received
```

can both be true.

A stronger future benchmark should separate these cases.

### Same-field contradiction

```text
Document A: units received = 240
Document B: units received = 238
```

Expected behavior:

```text
The received quantity is unresolved.
```

### Different semantic facts

```text
Shipping record: units shipped = 240
Receiving log:   units received = 238
```

Expected behavior:

```text
240 units were shipped and 238 units were received.
The supplied evidence does not establish the cause of the difference.
```

## Production recommendation

For high-reliability RAG and document QA, use a deterministic evidence-processing layer before generation.

Recommended flow:

```text
source documents
    |
    v
extract facts
    |
    v
normalize fields and units
    |
    v
detect conflicts
    |
    v
classify evidence
    |
    +--> CONSISTENT
    +--> CONFLICTING
    +--> MISSING
    +--> AMBIGUOUS
    |
    v
construct grounded prompt
    |
    v
LLM generation
```

Prompt engineering alone should not be treated as the only conflict-detection mechanism for high-stakes workflows.
