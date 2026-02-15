use nous_core::types::Timestamp;

use super::fallback::Fallback;

/// Result of precondition validation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PreconditionResult {
    pub condition_id: String,
    pub satisfied: bool,
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub similarity: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub reason: Option<String>,
}

/// Result of postcondition verification.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostconditionResult {
    pub condition_id: String,
    pub verified: bool,
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub actual_confidence: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub reason: Option<String>,
}

/// Type of execution error.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum ErrorType {
    Precondition,
    Execution,
    Postcondition,
    Timeout,
}

/// Execution error details.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutionError {
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub error_type: ErrorType,
    pub message: String,
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub condition_id: Option<String>,
}

/// Execution performance metrics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutionMetrics {
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    pub duration_ms: u64,
    pub retry_count: u32,
}

/// Full execution result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutionResult<T> {
    pub success: bool,

    /// The actual result data
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub data: Option<T>,

    /// Precondition check results
    pub preconditions: Vec<PreconditionResult>,

    /// Postcondition verification results
    pub postconditions: Vec<PostconditionResult>,

    /// If a fallback was triggered
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub fallback_triggered: Option<Fallback>,

    /// Execution error if any
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub error: Option<ExecutionError>,

    /// Execution metrics
    pub metrics: ExecutionMetrics,
}
