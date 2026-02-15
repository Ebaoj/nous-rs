/// Configuration options for the [`super::NousExecutor`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutorOptions {
    /// Default timeout in milliseconds for handler execution.
    pub timeout_ms: u64,

    /// Maximum retries when a retry fallback strategy is active.
    pub max_retries: u32,

    /// Whether to validate message structure before execution.
    pub validate_messages: bool,

    /// Whether to validate contracts (pre/postconditions).
    pub validate_contracts: bool,
}

impl Default for ExecutorOptions {
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,
            max_retries: 3,
            validate_messages: true,
            validate_contracts: true,
        }
    }
}

impl ExecutorOptions {
    /// Create options with all validations enabled and sensible defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the execution timeout in milliseconds.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set the maximum retry count.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Enable or disable message validation.
    pub fn with_message_validation(mut self, enable: bool) -> Self {
        self.validate_messages = enable;
        self
    }

    /// Enable or disable contract validation.
    pub fn with_contract_validation(mut self, enable: bool) -> Self {
        self.validate_contracts = enable;
        self
    }
}
