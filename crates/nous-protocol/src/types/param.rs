use std::collections::HashMap;

use nous_core::types::{Confidence, Embedding};

/// Supported parameter value types.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum ParamType {
    String,
    Number,
    Boolean,
    Array,
    Object,
    Embedding,
}

/// Concrete parameter value with data.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum ParamValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ParamValue>),
    Object(HashMap<String, ParamValue>),
    Embedding(Embedding),
}

impl ParamValue {
    /// Returns the expected `ParamType` for this value.
    pub fn param_type(&self) -> ParamType {
        match self {
            ParamValue::String(_) => ParamType::String,
            ParamValue::Number(_) => ParamType::Number,
            ParamValue::Boolean(_) => ParamType::Boolean,
            ParamValue::Array(_) => ParamType::Array,
            ParamValue::Object(_) => ParamType::Object,
            ParamValue::Embedding(_) => ParamType::Embedding,
        }
    }
}

/// Source of the parameter value.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum ParamSource {
    User,
    Inferred,
    Default,
    Derived,
}

/// An alternative value with its probability.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Alternative {
    pub value: ParamValue,
    pub probability: f64,
}

/// A typed parameter with uncertainty quantification.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypedParam {
    /// Parameter name
    pub name: String,

    /// Parameter type for validation
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub param_type: ParamType,

    /// The value
    pub value: ParamValue,

    /// Uncertainty in the value (0-1, lower = more certain)
    pub uncertainty: Confidence,

    /// Alternative values if primary is uncertain
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub alternatives: Vec<Alternative>,

    /// Source of the parameter value
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub source: Option<ParamSource>,
}
