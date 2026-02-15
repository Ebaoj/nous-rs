use std::collections::HashMap;

use nous_core::types::MessageId;

use super::param::ParamValue;

/// What triggers a fallback action.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum FallbackTrigger {
    PreconditionFailed,
    ExecutionError,
    PostconditionFailed,
    Timeout,
    Custom(String),
}

/// Strategy to handle the failure.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "strategy", rename_all = "snake_case"))]
pub enum FallbackStrategy {
    Retry {
        max_retries: u32,
    },
    Alternative {
        message_id: MessageId,
    },
    Degrade {
        params: HashMap<String, ParamValue>,
    },
    Abort,
    Escalate,
}

/// A fallback action combining trigger and strategy.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fallback {
    /// What triggers this fallback
    pub trigger: FallbackTrigger,

    /// Specific condition ID that triggers (optional)
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub condition_id: Option<String>,

    /// Strategy to handle the failure
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub strategy: FallbackStrategy,
}
