# Design

## Architecture Overview

**Tool Registry**  
The registry loads ToolBench `data/toolenv/tools/{Category}/{tool}.json` files, normalizes inconsistent fields, and exposes typed `Tool`, `Endpoint`, and `Parameter` objects. It is the source of truth for tool metadata, endpoint schemas, parameter defaults, and response schemas when available.

**Tool Graph + Sampler**  
The graph is built from the registry with NetworkX and contains `Tool`, `Endpoint`, `Parameter`, `ResponseField`, and `Concept/Tag` nodes. Category directory names become concept nodes, and additional keyword concepts are extracted from `tool_description`. Sampling traverses this graph directly to produce multi-step or parallel endpoint sets; no hardcoded endpoint lists are used.

**Offline Execution Model**  
The executor validates arguments against the endpoint schema and produces deterministic mock outputs. In `template` mode it generates minimal valid JSON from the response schema; in `llm` mode it calls the LLM with `temperature=0` and a fixed seed. `SessionState` stores prior outputs so later steps can reference IDs, handles, or selected items.

**Multi-Agent Generator**  
Generation is split into five agents: sampler, planner, user-proxy, assistant, and validator. The sampler proposes a tool chain, the planner turns it into a conversation plan, the user-proxy emits user-side turns, the assistant alternates between clarification and tool-call proposals, and the validator checks the final record against the assignment rules before it is written.

**Memory Layer**  
All memory access goes through the `MemoryStore` interface. `Mem0MemoryStore` is the real implementation and uses mem0 with in-process Qdrant; `FakeMemoryStore` supports deterministic tests. Session memory is written after every tool call and read before non-first argument filling. Corpus memory is written after each completed conversation and read before planning the next one.

**Dataset + CLI**  
The dataset contract is defined in Pydantic v2 as `ConversationRecord`, and JSONL is used for append-safe writes. The CLI is a thin Typer layer over the existing modules: `build`, `generate`, `validate`, and `metrics`. `append_jsonl` is the primary write path so generation progress is preserved if the process stops mid-run.

## Key Design Decisions

**Interface / implementation split**  
The main explicit split is `MemoryStore` vs `Mem0MemoryStore`. Other components depend only on the interface, which keeps tests fast and isolates mem0-specific behavior to one module.

**Dependency rules by layer**  
Lower layers do not import higher ones. The executor does not import agents or generator modules, `SessionState` has no toolgenerator dependencies, and agent-shared types live in `agents/types.py` so agents do not form circular imports.

**Crash-safe writes**  
`append_jsonl()` is used during generation so each validated conversation is persisted immediately. This avoids losing the full dataset if generation fails halfway through a long run.

**Graph-based sampling instead of hardcoded chains**  
The generator uses `ToolGraphSampler` through the sampler agent, satisfying the hard requirement that sampling must traverse the graph. This also makes tool-chain proposals reproducible from a fixed seed and flexible across different tool corpora.

**Determinism**  
The pipeline sets `random.seed(seed)` and passes the same seed to the graph sampler, executor, and all LLM-backed agents. Template-mode execution is fully deterministic, and LLM-mode calls use `temperature=0` with the global seed.

## Corpus Memory & Diversity Analysis

**Metrics chosen and why**  
I used `unique_tool_chain_ratio` as the primary diversity metric because the assignment focuses on multi-tool and multi-step tool-use traces, so measuring the fraction of unique endpoint chains directly reflects structural diversity in the generated dataset. I used `distinct_2_gram` on assistant utterances as a secondary metric because it captures surface-form diversity in the language the assistant produces.

**Run configuration**  
Both runs used the same seed (`42`) and generated 50 conversations in deterministic template mode.

**Numeric results**

| Run | Corpus Memory | Unique Tool-Chain Ratio | Distinct-2 |
|---|---|---:|---:|
| A | Disabled | 1.0000 | 0.0200 |
| B | Enabled | 1.0000 | 0.0200 |

**Analysis**  
In this deterministic template-mode setup, enabling corpus memory did not improve either structural or lexical diversity. The primary reason is that the planner/user/assistant behavior was intentionally constrained for reproducible offline testing, so the same seed produced the same overall planning and language patterns in both runs. The graph sampler already generated highly diverse tool chains in this experiment, which is why `unique_tool_chain_ratio` was already saturated at `1.0000` without corpus memory. In a richer LLM-driven setup where planner prompts can react more strongly to retrieved corpus summaries, I would expect corpus memory to have more visible effect on topical and linguistic diversity.
