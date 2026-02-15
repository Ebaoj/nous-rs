//! Basic Nous flow: build a message, execute it, inspect the result.
//!
//! Run with: cargo run --example basic_flow

use nous::core::confidence::builder::create_confidence;
use nous::core::embeddings::math::cosine_similarity;
use nous::core::hybrid_validator::validate_hybrid;
use nous::core::logical_atoms::comparison::compare_logical_atoms;
use nous::core::logical_atoms::extractor::extract_logical_atoms_sync;
use nous::core::quantization::{binary_quantize, hamming_distance, hamming_to_cosine_similarity};
use nous::core::types::{Confidence, Embedding};

use nous::protocol::builders::{IntentBuilder, NousProtocolBuilder};
use nous::protocol::types::*;
use nous::protocol::validation::validate_message;

use nous::runtime::executor::{
    ExecutionContext, ExecutorOptions, HandlerResult, MessageHandler, NousExecutor,
};
use nous::runtime::store::{InMemoryStore, MessageStore};

// ---------------------------------------------------------------------------
// 1. A handler that "deletes a user account"
// ---------------------------------------------------------------------------
struct DeleteAccountHandler;

impl MessageHandler for DeleteAccountHandler {
    fn action(&self) -> &str {
        "delete_account"
    }

    fn handle(
        &self,
        message: &NousProtocolMessage,
        _context: &ExecutionContext,
    ) -> nous::runtime::error::RuntimeResult<HandlerResult> {
        let user = message
            .params
            .iter()
            .find(|p| p.name == "user_id")
            .and_then(|p| match &p.value {
                ParamValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("unknown");

        println!("  [handler] Deleting account for user: {user}");

        Ok(HandlerResult {
            data: Some(format!("Account {user} deleted")),
            result_embedding: message.embedding.clone(),
            confidence: 0.95,
        })
    }
}

// ---------------------------------------------------------------------------
// 2. A handler that queries data
// ---------------------------------------------------------------------------
struct QueryHandler;

impl MessageHandler for QueryHandler {
    fn action(&self) -> &str {
        "query_data"
    }

    fn handle(
        &self,
        message: &NousProtocolMessage,
        _context: &ExecutionContext,
    ) -> nous::runtime::error::RuntimeResult<HandlerResult> {
        println!("  [handler] Querying data...");
        Ok(HandlerResult {
            data: Some("42 records found".into()),
            result_embedding: message.embedding.clone(),
            confidence: 0.88,
        })
    }
}

fn main() {
    println!("=== Nous Protocol — Basic Flow ===\n");

    // -- Embeddings & Similarity -----------------------------------------------
    println!("1. Embedding similarity");
    let emb_a = Embedding::new(vec![0.8, 0.2, 0.1, 0.5]).unwrap();
    let emb_b = Embedding::new(vec![0.7, 0.3, 0.1, 0.4]).unwrap();
    let emb_c = Embedding::new(vec![-0.5, 0.9, -0.3, 0.1]).unwrap();

    let sim_ab = cosine_similarity(&emb_a, &emb_b).unwrap();
    let sim_ac = cosine_similarity(&emb_a, &emb_c).unwrap();
    println!("   A vs B: {sim_ab:.4} (similar)");
    println!("   A vs C: {sim_ac:.4} (different)");

    // -- Quantization ----------------------------------------------------------
    println!("\n2. Binary quantization (97% compression)");
    let big = Embedding::new((0..1536).map(|i| (i as f64 - 768.0) / 768.0).collect()).unwrap();
    let quantized = binary_quantize(&big, "text-embedding-3-small");
    println!(
        "   {} bytes -> {} bytes (ratio: {:.0}x)",
        quantized.stats.original_bytes,
        quantized.stats.compressed_bytes,
        quantized.stats.compression_ratio
    );

    let q1 = binary_quantize(&emb_a, "test");
    let q2 = binary_quantize(&emb_b, "test");
    let dist = hamming_distance(&q1, &q2).unwrap();
    let approx_sim = hamming_to_cosine_similarity(dist, emb_a.len());
    println!("   Hamming distance A-B: {dist}, approx cosine: {approx_sim:.4}");

    // -- Confidence ------------------------------------------------------------
    println!("\n3. Confidence maps");
    let confidence = create_confidence(0.85)
        .factual(0.9, Some("verified against database"))
        .completeness(0.7, Some("missing 2 optional fields"))
        .relevance(0.95, None)
        .recalculate_overall()
        .build();
    println!("   Overall: {:.2}", confidence.overall.value());
    println!(
        "   Meets 0.8 threshold: {}",
        confidence.meets_threshold(0.8, None)
    );
    println!(
        "   Factual meets 0.85: {}",
        confidence.meets_threshold(0.85, Some("factual"))
    );

    // -- Logical Atoms ---------------------------------------------------------
    println!("\n4. Logical atoms & conflict detection");
    let atoms_a = extract_logical_atoms_sync("Delete all inactive users older than 30 days");
    let atoms_b = extract_logical_atoms_sync("Keep some inactive users for audit");
    let comparison = compare_logical_atoms(&atoms_a, &atoms_b);
    println!("   Compatible: {}", comparison.compatible);
    for c in &comparison.conflicts {
        println!("   Conflict: {c}");
    }

    // -- Hybrid Validation -----------------------------------------------------
    println!("\n5. Hybrid validation (embeddings + atoms)");
    let text_a = "Delete all user records";
    let text_b = "Remove some user records";
    let atoms_x = extract_logical_atoms_sync(text_a);
    let atoms_y = extract_logical_atoms_sync(text_b);
    // Simulate embeddings that are very similar
    let e1 = Embedding::new(vec![0.9, 0.1, 0.3, 0.5]).unwrap();
    let e2 = Embedding::new(vec![0.85, 0.15, 0.28, 0.48]).unwrap();
    let result = validate_hybrid(&e1, &e2, Some(&atoms_x), Some(&atoms_y), 0.8).unwrap();
    println!("   Embedding similarity: {:.4}", result.embedding_similarity);
    println!(
        "   Atoms compatible: {}",
        result
            .atoms_result
            .as_ref()
            .map(|r| r.compatible)
            .unwrap_or(true)
    );
    println!("   Overall compatible: {}", result.compatible);
    for reason in &result.failure_reasons {
        println!("   Reason: {reason}");
    }

    // -- Build Protocol Message ------------------------------------------------
    println!("\n6. Build & validate a NousProtocolMessage");
    let intent = IntentBuilder::new()
        .action("delete_account")
        .confidence(0.92)
        .description("Delete a user account permanently")
        .build()
        .unwrap();

    let msg = NousProtocolBuilder::new()
        .embedding(Embedding::new(vec![0.5, 0.3, 0.8, 0.1]).unwrap())
        .intent(intent)
        .param(TypedParam {
            name: "user_id".into(),
            param_type: ParamType::String,
            value: ParamValue::String("usr_42".into()),
            uncertainty: Confidence::new(0.05), // 5% uncertainty
            alternatives: Vec::new(),
            source: Some(ParamSource::User),
        })
        .text("Please delete user usr_42")
        .build()
        .unwrap();

    let validation = validate_message(&msg);
    println!("   Message ID: {}", msg.id.as_str());
    println!("   Action: {}", msg.intent.action);
    println!("   Valid: {}", validation.valid);
    println!("   Warnings: {}", validation.warnings.len());

    // -- Execute with NousExecutor ---------------------------------------------
    println!("\n7. Execute message through NousExecutor");
    let store = InMemoryStore::new();
    let opts = ExecutorOptions::default().with_message_validation(false);
    let mut executor = NousExecutor::new(store, opts);
    executor.register_handler(Box::new(DeleteAccountHandler));
    executor.register_handler(Box::new(QueryHandler));

    let ctx = ExecutionContext::new();
    let result = executor.execute(&msg, &ctx);
    println!("   Success: {}", result.success);
    println!("   Data: {:?}", result.data);
    println!("   Duration: {}ms", result.metrics.duration_ms);

    // -- Execute a second message (query) --------------------------------------
    println!("\n8. Execute a query message");
    let query_intent = IntentBuilder::new()
        .action("query_data")
        .confidence(0.88)
        .build()
        .unwrap();

    let query_msg = NousProtocolBuilder::new()
        .embedding(Embedding::new(vec![0.1, 0.9, 0.2, 0.4]).unwrap())
        .intent(query_intent)
        .text("How many records exist?")
        .build()
        .unwrap();

    let result2 = executor.execute(&query_msg, &ctx);
    println!("   Success: {}", result2.success);
    println!("   Data: {:?}", result2.data);

    // -- Store & retrieve messages ---------------------------------------------
    println!("\n9. Store & retrieve messages");
    executor.store_mut().add(msg.clone());
    executor.store_mut().add(query_msg.clone());
    println!("   Store size: {}", executor.store().len());

    let search_emb = Embedding::new(vec![0.5, 0.3, 0.8, 0.1]).unwrap();
    let similar = executor.store().find_by_similarity(&search_emb, 0.5);
    println!("   Similar messages found: {}", similar.len());
    for sim_msg in &similar {
        let score = cosine_similarity(&search_emb, &sim_msg.embedding).unwrap();
        println!("     - {} (sim: {score:.4})", sim_msg.intent.action);
    }

    // -- Serialization ---------------------------------------------------------
    println!("\n10. Serialization (serde)");
    let json = serde_json::to_string_pretty(&msg).unwrap();
    println!("   JSON size: {} bytes", json.len());
    let preview_len = json.len().min(200);
    println!("   Preview: {}...", &json[..preview_len]);

    let deserialized: NousProtocolMessage = serde_json::from_str(&json).unwrap();
    println!("   Roundtrip OK: {}", deserialized.id == msg.id);

    println!("\n=== Done! ===");
}
