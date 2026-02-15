# Nous

**AI-native communication protocol — beyond human language.**

Nous is a protocol for AI agents that need to communicate with semantic precision. Instead of passing plain text or JSON between agents, Nous messages carry embeddings, multidimensional confidence maps, logical atoms, and executable contracts with pre/post-conditions and fallback strategies.

The name comes from the Greek *nous* (νοῦς) — the faculty of intellectual understanding.

## The Problem

When AI agents communicate via plain text or JSON, they share strings. Strings are ambiguous.

Consider two agents coordinating a database cleanup:

- Agent A says: *"Delete all inactive users older than 30 days"*
- Agent B interprets: *"Delete all users that are inactive"*

An embedding model would give these ~0.99 cosine similarity. They look almost identical. But they mean different things — Agent A specified a constraint (30 days), Agent B dropped it.

Or worse:

- Agent A: *"Delete all user records"*
- Agent B: *"Remove some user records"*

Embedding similarity: 0.998. But "all" vs "some" is a fundamental logical conflict that pure embeddings cannot detect.

This is **embedding blindness** — the inability of vector representations to capture logical structure like quantifiers (all/some/none), negations (do/don't), numerical constraints, and ordering requirements.

## How Nous Solves This

Nous uses a **hybrid representation** that combines three layers:

### 1. Semantic Layer (Embeddings)

Every Nous message carries an embedding vector — the standard dense representation of meaning. This handles the "what is this about?" question well: similar topics cluster together, unrelated topics are far apart.

```rust
use nous::core::embeddings::math::cosine_similarity;

let sim = cosine_similarity(&embedding_a, &embedding_b)?;
// 0.99 — these are about the same topic
```

### 2. Logical Layer (Atoms)

Nous extracts **logical atoms** from natural language — quantifiers, negations, numbers, constraints, and orderings. These are the structural components that embeddings miss.

```rust
use nous::core::logical_atoms::extractor::extract_logical_atoms_sync;
use nous::core::logical_atoms::comparison::compare_logical_atoms;

let atoms_a = extract_logical_atoms_sync("Delete all inactive users");
let atoms_b = extract_logical_atoms_sync("Delete some inactive users");

let result = compare_logical_atoms(&atoms_a, &atoms_b);
// result.compatible = false
// result.conflicts = ["Quantifier conflict: All vs Some on 'inactive users'"]
```

### 3. Hybrid Validator

The hybrid validator combines both layers. Embeddings say "these are similar" — atoms say "but they logically conflict." The conflict wins.

```rust
use nous::core::hybrid_validator::validate_hybrid;

let result = validate_hybrid(&emb_a, &emb_b, Some(&atoms_a), Some(&atoms_b), 0.8)?;
// result.embedding_similarity = 0.998  (high — embeddings think they match)
// result.compatible = false             (atoms caught the quantifier conflict)
```

This is the core insight of Nous: **semantic similarity is necessary but not sufficient for communication correctness.**

## Multidimensional Confidence

Traditional AI confidence is a single number: "I'm 0.85 confident." But 0.85 confident about *what*?

Nous confidence maps break this into dimensions:

```rust
use nous::core::confidence::builder::create_confidence;

let confidence = create_confidence(0.85)
    .factual(0.95, Some("verified against primary source"))
    .completeness(0.60, Some("3 of 5 fields populated"))
    .relevance(0.90, None)
    .recalculate_overall()
    .build();
```

An agent receiving this message can make nuanced decisions:
- **High factual, low completeness** → the data is correct but partial. Safe to use, but flag as incomplete.
- **Low factual, high completeness** → all fields present, but source is unreliable. Needs verification.
- **Everything low** → don't trust this message.

### Confidence Propagation

When a message passes through a chain of agents (research → summarize → decide → execute), confidence degrades at each step. Nous tracks this:

```rust
use nous::core::confidence::propagation::propagate_confidence;

// Each transformation reduces confidence by its propagation factor
let propagated = propagate_confidence(&original_confidence, &transform, &config);
// Original: 0.90 → After summarization (factor 0.85): 0.765
```

An executor at the end of a long chain knows it's operating on degraded information and can escalate to a human instead of acting autonomously.

## Protocol Messages

A `NousProtocolMessage` is the unit of communication. It carries everything an agent needs to understand, validate, and act on a request:

```rust
use nous::protocol::builders::{IntentBuilder, NousProtocolBuilder};

let msg = NousProtocolBuilder::new()
    .embedding(embedding)
    .intent(
        IntentBuilder::new()
            .action("delete_account")
            .confidence(0.92)
            .description("Permanently delete a user account")
            .build()?
    )
    .param(TypedParam {
        name: "user_id".into(),
        param_type: ParamType::String,
        value: ParamValue::String("usr_42".into()),
        uncertainty: Confidence::new(0.05),
        alternatives: vec![],
        source: Some(ParamSource::User),
    })
    .contracts(contracts)
    .text("Please delete user usr_42")
    .build()?;
```

### Contracts

Every message can carry contracts — preconditions that must hold before execution, postconditions that must hold after, and fallback strategies for when they don't:

- **Preconditions**: "The user must exist in the database" (verified via embedding similarity against state)
- **Postconditions**: "The result must have >0.8 confidence" (verified against handler output)
- **Fallbacks**: If precondition fails → retry. If execution errors → try alternative message. If postcondition fails → escalate to human.

This is **design-by-contract applied to agent communication.**

## Executor

The `NousExecutor` orchestrates the full lifecycle:

1. Validate message structure
2. Validate contract consistency
3. Check preconditions against execution context
4. Find and invoke the registered handler
5. Verify postconditions against the result
6. Handle fallbacks when any step fails

```rust
use nous::runtime::executor::{NousExecutor, ExecutionContext, MessageHandler, HandlerResult};
use nous::runtime::store::InMemoryStore;

struct DeleteHandler;
impl MessageHandler for DeleteHandler {
    fn action(&self) -> &str { "delete_account" }
    fn handle(&self, msg: &NousProtocolMessage, ctx: &ExecutionContext)
        -> RuntimeResult<HandlerResult> {
        // ... your logic here
        Ok(HandlerResult {
            data: Some("done".into()),
            result_embedding: msg.embedding.clone(),
            confidence: 0.95,
        })
    }
}

let mut executor = NousExecutor::with_store(InMemoryStore::new());
executor.register_handler(Box::new(DeleteHandler));

let result = executor.execute(&msg, &ctx);
// result.success, result.data, result.metrics, result.preconditions, ...
```

## Quantization

For high-throughput systems, Nous supports embedding quantization:

- **Binary quantization**: 1536 floats (6,144 bytes) → 192 bytes. **32x compression**. Uses Hamming distance for fast similarity search.
- **Scalar quantization**: 1536 floats (6,144 bytes) → 1,536 bytes. **4x compression**. Better accuracy than binary.

```rust
use nous::core::quantization::{binary_quantize, hamming_distance};

let quantized = binary_quantize(&embedding, "text-embedding-3-small");
// 6144 bytes → 192 bytes

let dist = hamming_distance(&q1, &q2)?;
// Much faster than cosine similarity on full vectors
```

## Concept Graphs

The runtime can build concept graphs from message history using agglomerative clustering. Messages with similar embeddings cluster into concepts, and edges form between related concepts:

```rust
use nous::runtime::concepts::ConceptGraphBuilder;

let mut builder = ConceptGraphBuilder::new(options);
builder.add_message("msg-1", &embedding_1, "user-management", timestamp);
builder.add_message("msg-2", &embedding_2, "user-deletion", timestamp);

let graph = builder.build();
// graph.nodes: discovered concepts
// graph.edges: relationships between concepts
```

## Architecture

```
nous-rs/
  nous/              # Facade crate — `use nous::prelude::*`
  crates/
    nous-core/       # Types, confidence, embeddings, quantization, logical atoms
    nous-protocol/   # Messages, builders, validation, contracts
    nous-runtime/    # Executor, store, memory, concept graphs
```

| Crate | What | Tests |
|-------|------|-------|
| `nous-core` | Embedding/Confidence newtypes, math, quantization, logical atoms, hybrid validator | 24 |
| `nous-protocol` | NousProtocolMessage, builders, validation, contracts | 38 |
| `nous-runtime` | Executor, InMemoryStore, ReferenceResolver, memory manager, concept graphs | 30 |
| **Total** | | **92** |

## When to Use Nous

**Use Nous when:**

- Agents delegate **destructive or irreversible actions** to each other (delete data, send payments, modify infrastructure)
- You have **long agent chains** where information degrades through multiple transformations and you need to track confidence decay
- You need **semantic auditing** — not just "what happened" but "why did the agent think this was correct"
- Agents operate **autonomously without human-in-the-loop** and you need safety nets beyond retry logic
- You care about catching **logical conflicts** that embedding similarity alone misses (all vs some, do vs don't, 30 days vs 90 days)

**Don't use Nous when:**

- You're doing simple tool calling or RPC between agents — plain JSON works fine
- The cost of an error is low and retry resolves it
- You have a human reviewing every agent action anyway
- Your agents communicate structured commands, not natural language intentions

## When Does This Matter?

Today, most multi-agent systems are simple enough for JSON. An orchestrator calls tools, gets results, moves on. The cost of a misunderstanding is usually a retry or a user seeing an error message.

But the trajectory is clear: agents are getting more autonomous, chains are getting longer, and the actions they take are getting more consequential. When an agent autonomously manages your infrastructure, handles your finances, or coordinates a fleet of other agents — "the embeddings were similar" is not a sufficient safety guarantee.

Nous is designed for that future. It's the difference between a handshake and a signed contract.

## Quick Start

```bash
# Run the example
cargo run --example basic_flow -p nous

# Run all tests
cargo test --workspace

# Use as a dependency
# Cargo.toml:
# [dependencies]
# nous = { git = "https://github.com/Ebaoj/nous-rs" }
```

## License

MIT — Joabe Cornelio, 2026
