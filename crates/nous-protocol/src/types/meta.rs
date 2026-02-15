use std::collections::HashMap;

use nous_core::types::Timestamp;

/// Protocol version identifier.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ProtocolVersion {
    #[default]
    #[cfg_attr(feature = "serde", serde(rename = "0.2"))]
    V0_2,
    #[cfg_attr(feature = "serde", serde(rename = "0.3"))]
    V0_3,
}

impl std::fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolVersion::V0_2 => write!(f, "0.2"),
            ProtocolVersion::V0_3 => write!(f, "0.3"),
        }
    }
}

/// Information about the sender agent.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SenderInfo {
    /// Sender type (e.g., "claude", "gpt", "system")
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub sender_type: String,

    /// Sender identifier
    pub id: String,

    /// Sender version
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub version: Option<String>,
}

/// Message metadata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProtocolMeta {
    /// Sender agent identifier
    pub sender: SenderInfo,

    /// Unix timestamp in milliseconds
    pub timestamp: Timestamp,

    /// Conversation/session identifier
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub conversation_id: Option<String>,

    /// Sequence number in conversation
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub sequence: Option<u32>,

    /// Protocol version
    pub protocol_version: ProtocolVersion,

    /// Custom metadata
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub custom: Option<HashMap<String, String>>,
}
