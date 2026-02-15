use nous_core::types::Timestamp;
use nous_protocol::types::{
    ExecutionMetrics, ExecutionResult, ExecutionError, ErrorType,
    FallbackStrategy, FallbackTrigger, NousProtocolMessage,
    PostconditionResult, PreconditionResult,
};
use nous_protocol::validation::{
    check_preconditions, validate_contracts, validate_message, verify_postconditions,
};

use crate::error::RuntimeError;
use crate::store::MessageStore;

use super::context::ExecutionContext;
use super::handler::{HandlerRegistry, HandlerResult};
use super::options::ExecutorOptions;

/// Main executor for NousProtocolMessages.
///
/// Orchestrates the full execution flow:
/// 1. Validate message structure
/// 2. Validate contracts consistency
/// 3. Check preconditions against context state
/// 4. Find and invoke the registered handler
/// 5. Verify postconditions against handler result
/// 6. Handle fallbacks when steps fail
pub struct NousExecutor<S: MessageStore> {
    options: ExecutorOptions,
    store: S,
    handlers: HandlerRegistry,
}

impl<S: MessageStore> NousExecutor<S> {
    /// Create a new executor with a message store and options.
    pub fn new(store: S, options: ExecutorOptions) -> Self {
        Self {
            options,
            store,
            handlers: HandlerRegistry::new(),
        }
    }

    /// Create a new executor with default options.
    pub fn with_store(store: S) -> Self {
        Self::new(store, ExecutorOptions::default())
    }

    /// Get a reference to the handler registry.
    pub fn handlers(&self) -> &HandlerRegistry {
        &self.handlers
    }

    /// Get a mutable reference to the handler registry for registration.
    pub fn handlers_mut(&mut self) -> &mut HandlerRegistry {
        &mut self.handlers
    }

    /// Register a handler (convenience wrapper).
    pub fn register_handler(&mut self, handler: Box<dyn super::handler::MessageHandler>) {
        self.handlers.register(handler);
    }

    /// Get a reference to the backing store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get a mutable reference to the backing store.
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Execute a message with full contract validation and fallback handling.
    pub fn execute(
        &self,
        message: &NousProtocolMessage,
        context: &ExecutionContext,
    ) -> ExecutionResult<String> {
        let start_time = Timestamp::now();
        let retry_count = 0;

        // Step 1: Validate message structure
        if self.options.validate_messages {
            let validation = validate_message(message);
            if !validation.valid {
                let error_msgs: Vec<String> = validation
                    .errors
                    .iter()
                    .map(|e| e.message.clone())
                    .collect();
                return self.failed_result(
                    start_time,
                    retry_count,
                    ErrorType::Execution,
                    format!("Validation failed: {}", error_msgs.join(", ")),
                    None,
                );
            }
        }

        // Step 2: Validate contracts consistency
        if self.options.validate_contracts {
            let issues = validate_contracts(message);
            if !issues.is_empty() {
                return self.failed_result(
                    start_time,
                    retry_count,
                    ErrorType::Execution,
                    format!("Contract validation failed: {}", issues.join(", ")),
                    None,
                );
            }
        }

        // Step 3: Check preconditions
        let (all_satisfied, precondition_results) =
            check_preconditions(message, &context.state_embeddings);

        if !all_satisfied {
            // Extract info before moving the results vector
            let condition_id: Option<String> = precondition_results
                .iter()
                .find(|r| !r.satisfied)
                .map(|f| f.condition_id.clone());
            let error_msg = format!(
                "Precondition failed: {}",
                condition_id.as_deref().unwrap_or("unknown")
            );

            // Look for a precondition_failed fallback
            if let Some(fallback) = find_fallback(
                message,
                &FallbackTrigger::PreconditionFailed,
                condition_id.as_deref(),
            ) {
                return self.handle_fallback(
                    message,
                    context,
                    &fallback.strategy,
                    start_time,
                    retry_count,
                    precondition_results,
                    Vec::new(),
                );
            }

            return ExecutionResult {
                success: false,
                data: None,
                preconditions: precondition_results,
                postconditions: Vec::new(),
                fallback_triggered: None,
                error: Some(ExecutionError {
                    error_type: ErrorType::Precondition,
                    message: error_msg,
                    condition_id,
                }),
                metrics: self.make_metrics(start_time, retry_count),
            };
        }

        // Step 4+5: Execute with retry support
        self.execute_with_retry(
            message,
            context,
            start_time,
            retry_count,
            precondition_results,
        )
    }

    /// Internal: execute the handler with retry support.
    fn execute_with_retry(
        &self,
        message: &NousProtocolMessage,
        context: &ExecutionContext,
        start_time: Timestamp,
        retry_count: u32,
        precondition_results: Vec<PreconditionResult>,
    ) -> ExecutionResult<String> {
        // Find handler
        let handler = match self.handlers.get(&message.intent.action) {
            Some(h) => h,
            None => {
                return self.failed_result(
                    start_time,
                    retry_count,
                    ErrorType::Execution,
                    format!(
                        "No handler registered for intent: {}",
                        message.intent.action
                    ),
                    None,
                );
            }
        };

        // Execute handler
        let handler_result: Result<HandlerResult, RuntimeError> =
            handler.handle(message, context);

        match handler_result {
            Ok(result) => {
                // Verify postconditions
                let (all_verified, postcondition_results) =
                    verify_postconditions(message, &result.result_embedding, result.confidence);

                if !all_verified {
                    // Extract info before moving the results vector
                    let condition_id: Option<String> = postcondition_results
                        .iter()
                        .find(|r| !r.verified)
                        .map(|f| f.condition_id.clone());
                    let error_msg = format!(
                        "Postcondition failed: {}",
                        condition_id.as_deref().unwrap_or("unknown")
                    );

                    // Look for a postcondition_failed fallback
                    if let Some(fallback) = find_fallback(
                        message,
                        &FallbackTrigger::PostconditionFailed,
                        condition_id.as_deref(),
                    ) {
                        return self.handle_fallback(
                            message,
                            context,
                            &fallback.strategy,
                            start_time,
                            retry_count,
                            precondition_results,
                            postcondition_results,
                        );
                    }

                    return ExecutionResult {
                        success: false,
                        data: None,
                        preconditions: precondition_results,
                        postconditions: postcondition_results,
                        fallback_triggered: None,
                        error: Some(ExecutionError {
                            error_type: ErrorType::Postcondition,
                            message: error_msg,
                            condition_id,
                        }),
                        metrics: self.make_metrics(start_time, retry_count),
                    };
                }

                // Success
                ExecutionResult {
                    success: true,
                    data: result.data,
                    preconditions: precondition_results,
                    postconditions: postcondition_results,
                    fallback_triggered: None,
                    error: None,
                    metrics: self.make_metrics(start_time, retry_count),
                }
            }
            Err(err) => {
                let is_timeout = matches!(&err, RuntimeError::Timeout(_));
                let trigger = if is_timeout {
                    FallbackTrigger::Timeout
                } else {
                    FallbackTrigger::ExecutionError
                };

                // Check for retry fallback
                if let Some(fallback) = find_fallback(message, &trigger, None) {
                    if let FallbackStrategy::Retry { max_retries } = &fallback.strategy {
                        let effective_max = (*max_retries).min(self.options.max_retries);
                        if retry_count < effective_max {
                            return self.execute_with_retry(
                                message,
                                context,
                                start_time,
                                retry_count + 1,
                                precondition_results,
                            );
                        }
                    }

                    return self.handle_fallback(
                        message,
                        context,
                        &fallback.strategy,
                        start_time,
                        retry_count,
                        precondition_results,
                        Vec::new(),
                    );
                }

                let error_type = if is_timeout {
                    ErrorType::Timeout
                } else {
                    ErrorType::Execution
                };

                self.failed_result(
                    start_time,
                    retry_count,
                    error_type,
                    err.to_string(),
                    None,
                )
            }
        }
    }

    /// Handle a fallback strategy.
    #[allow(clippy::too_many_arguments)]
    fn handle_fallback(
        &self,
        _message: &NousProtocolMessage,
        context: &ExecutionContext,
        strategy: &FallbackStrategy,
        start_time: Timestamp,
        retry_count: u32,
        _preconditions: Vec<PreconditionResult>,
        _postconditions: Vec<PostconditionResult>,
    ) -> ExecutionResult<String> {
        match strategy {
            FallbackStrategy::Retry { .. } => {
                // If we get here, retries are exhausted
                self.failed_result(
                    start_time,
                    retry_count,
                    ErrorType::Execution,
                    "Exhausted retry attempts".into(),
                    None,
                )
            }
            FallbackStrategy::Alternative { message_id } => {
                let id_str = message_id.as_str();
                if let Some(alt_msg) = self.store.get(id_str) {
                    let alt_msg_cloned = alt_msg.clone();
                    return self.execute(&alt_msg_cloned, context);
                }
                self.failed_result(
                    start_time,
                    retry_count,
                    ErrorType::Execution,
                    "Alternative message not found".into(),
                    None,
                )
            }
            FallbackStrategy::Degrade { .. } => {
                // In a real implementation we would modify params and re-execute.
                // For now, we report degradation as failed because the Rust type
                // system makes it non-trivial to clone-and-mutate the message
                // without an owned variant.
                self.failed_result(
                    start_time,
                    retry_count,
                    ErrorType::Execution,
                    "Degraded execution not yet supported".into(),
                    None,
                )
            }
            FallbackStrategy::Abort => self.failed_result(
                start_time,
                retry_count,
                ErrorType::Execution,
                "Aborted by fallback strategy".into(),
                None,
            ),
            FallbackStrategy::Escalate => self.failed_result(
                start_time,
                retry_count,
                ErrorType::Execution,
                "Escalated - requires manual intervention".into(),
                None,
            ),
        }
    }

    /// Build a failed execution result.
    fn failed_result(
        &self,
        start_time: Timestamp,
        retry_count: u32,
        error_type: ErrorType,
        message: String,
        condition_id: Option<String>,
    ) -> ExecutionResult<String> {
        ExecutionResult {
            success: false,
            data: None,
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            fallback_triggered: None,
            error: Some(ExecutionError {
                error_type,
                message,
                condition_id,
            }),
            metrics: self.make_metrics(start_time, retry_count),
        }
    }

    /// Build execution metrics.
    fn make_metrics(&self, start_time: Timestamp, retry_count: u32) -> ExecutionMetrics {
        let end_time = Timestamp::now();
        let duration_ms = end_time
            .as_millis()
            .saturating_sub(start_time.as_millis());
        ExecutionMetrics {
            start_time,
            end_time,
            duration_ms,
            retry_count,
        }
    }
}

impl<S: MessageStore + std::fmt::Debug> std::fmt::Debug for NousExecutor<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NousExecutor")
            .field("options", &self.options)
            .field("handlers", &self.handlers)
            .finish()
    }
}

/// Find a fallback matching the given trigger and optional condition ID.
fn find_fallback<'a>(
    message: &'a NousProtocolMessage,
    trigger: &FallbackTrigger,
    condition_id: Option<&str>,
) -> Option<&'a nous_protocol::types::Fallback> {
    // First try specific condition match
    if let Some(cid) = condition_id {
        let specific = message.contracts.fallbacks.iter().find(|f| {
            f.trigger == *trigger
                && f.condition_id.as_deref() == Some(cid)
        });
        if specific.is_some() {
            return specific;
        }
    }

    // Then try general match (no condition_id)
    message.contracts.fallbacks.iter().find(|f| {
        f.trigger == *trigger && f.condition_id.is_none()
    })
}

/// Create an executor with a new store and default options.
pub fn create_executor<S: MessageStore + Default>(
) -> NousExecutor<S> {
    NousExecutor::new(S::default(), ExecutorOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RuntimeResult;
    use crate::executor::handler::{HandlerResult, MessageHandler};
    use crate::store::InMemoryStore;
    use nous_core::types::*;
    use nous_protocol::types::*;

    // A simple test handler
    struct EchoHandler;

    impl MessageHandler for EchoHandler {
        fn action(&self) -> &str {
            "echo"
        }

        fn handle(
            &self,
            message: &NousProtocolMessage,
            _context: &ExecutionContext,
        ) -> RuntimeResult<HandlerResult> {
            Ok(HandlerResult {
                data: message.text.clone(),
                result_embedding: message.embedding.clone(),
                confidence: 0.95,
            })
        }
    }

    // A handler that always fails
    struct FailHandler;

    impl MessageHandler for FailHandler {
        fn action(&self) -> &str {
            "fail"
        }

        fn handle(
            &self,
            _message: &NousProtocolMessage,
            _context: &ExecutionContext,
        ) -> RuntimeResult<HandlerResult> {
            Err(RuntimeError::Execution("intentional failure".into()))
        }
    }

    fn make_message(action: &str, embedding_vals: Vec<f64>) -> NousProtocolMessage {
        NousProtocolMessage {
            id: MessageId::random(),
            embedding: Embedding::new(embedding_vals).unwrap(),
            embedding_model: EmbeddingModel::default(),
            intent: Intent {
                action: action.into(),
                confidence: IntentConfidence::from(0.9),
                description: None,
                embedding: None,
            },
            params: Vec::new(),
            contracts: Contracts::default(),
            references: Vec::new(),
            meta: ProtocolMeta {
                sender: SenderInfo {
                    sender_type: "system".into(),
                    id: "test".into(),
                    version: None,
                },
                timestamp: Timestamp::now(),
                conversation_id: None,
                sequence: None,
                protocol_version: ProtocolVersion::default(),
                custom: None,
            },
            text: Some("hello".into()),
            logical_atoms: None,
        }
    }

    #[test]
    fn test_execute_success() {
        let store = InMemoryStore::new();
        let mut executor = NousExecutor::new(store, ExecutorOptions::default());
        executor.register_handler(Box::new(EchoHandler));

        let msg = make_message("echo", vec![1.0, 0.0, 0.0]);
        let ctx = ExecutionContext::new();
        let result = executor.execute(&msg, &ctx);

        assert!(result.success);
        assert_eq!(result.data.as_deref(), Some("hello"));
        assert!(result.error.is_none());
    }

    #[test]
    fn test_handler_not_found() {
        let store = InMemoryStore::new();
        let executor = NousExecutor::<InMemoryStore>::new(store, ExecutorOptions::default());

        let msg = make_message("nonexistent", vec![1.0, 0.0, 0.0]);
        let ctx = ExecutionContext::new();
        let result = executor.execute(&msg, &ctx);

        assert!(!result.success);
        assert!(result.error.is_some());
        let err = result.error.unwrap();
        assert_eq!(err.error_type, ErrorType::Execution);
        assert!(err.message.contains("No handler registered"));
    }

    #[test]
    fn test_handler_failure() {
        let store = InMemoryStore::new();
        let mut executor = NousExecutor::new(store, ExecutorOptions::default());
        executor.register_handler(Box::new(FailHandler));

        let msg = make_message("fail", vec![1.0, 0.0, 0.0]);
        let ctx = ExecutionContext::new();
        let result = executor.execute(&msg, &ctx);

        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_metrics_recorded() {
        let store = InMemoryStore::new();
        let mut executor = NousExecutor::new(store, ExecutorOptions::default());
        executor.register_handler(Box::new(EchoHandler));

        let msg = make_message("echo", vec![1.0, 0.0, 0.0]);
        let ctx = ExecutionContext::new();
        let result = executor.execute(&msg, &ctx);

        assert!(result.metrics.end_time.as_millis() >= result.metrics.start_time.as_millis());
        assert_eq!(result.metrics.retry_count, 0);
    }

    #[test]
    fn test_validation_disabled() {
        let store = InMemoryStore::new();
        let opts = ExecutorOptions::default()
            .with_message_validation(false)
            .with_contract_validation(false);
        let mut executor = NousExecutor::new(store, opts);
        executor.register_handler(Box::new(EchoHandler));

        let msg = make_message("echo", vec![1.0, 0.0, 0.0]);
        let ctx = ExecutionContext::new();
        let result = executor.execute(&msg, &ctx);

        assert!(result.success);
    }
}
