use nous_core::types::Confidence;

use crate::error::{ProtocolError, ProtocolResult};
use crate::types::{Alternative, ParamSource, ParamType, ParamValue, TypedParam};

/// Fluent builder for creating `TypedParam` values.
#[derive(Debug)]
pub struct ParamBuilder {
    name: Option<String>,
    param_type: Option<ParamType>,
    value: Option<ParamValue>,
    uncertainty: Confidence,
    alternatives: Vec<Alternative>,
    source: Option<ParamSource>,
}

impl Default for ParamBuilder {
    fn default() -> Self {
        Self {
            name: None,
            param_type: None,
            value: None,
            uncertainty: Confidence::zero(),
            alternatives: Vec::new(),
            source: None,
        }
    }
}

impl ParamBuilder {
    /// Create a new builder instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the parameter name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the parameter type.
    pub fn param_type(mut self, param_type: ParamType) -> Self {
        self.param_type = Some(param_type);
        self
    }

    /// Set the parameter value.
    pub fn value(mut self, value: ParamValue) -> Self {
        self.value = Some(value);
        self
    }

    /// Set the uncertainty level (0-1, lower = more certain).
    pub fn uncertainty(mut self, uncertainty: f64) -> Self {
        self.uncertainty = Confidence::new(uncertainty);
        self
    }

    /// Add an alternative value with probability.
    pub fn alternative(mut self, value: ParamValue, probability: f64) -> Self {
        self.alternatives.push(Alternative {
            value,
            probability,
        });
        self
    }

    /// Set the source of the parameter value.
    pub fn source(mut self, source: ParamSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Build the typed parameter, validating required fields.
    pub fn build(self) -> ProtocolResult<TypedParam> {
        let name = self
            .name
            .ok_or_else(|| ProtocolError::MissingField("name".into()))?;

        let param_type = self
            .param_type
            .ok_or_else(|| ProtocolError::MissingField("param_type".into()))?;

        let value = self
            .value
            .ok_or_else(|| ProtocolError::MissingField("value".into()))?;

        // Validate type matches value
        let actual_type = value.param_type();
        if actual_type != param_type {
            return Err(ProtocolError::InvalidParam(format!(
                "{name}: declared type {param_type:?} but value is {actual_type:?}"
            )));
        }

        // Validate alternative probabilities
        for alt in &self.alternatives {
            if alt.probability < 0.0 || alt.probability > 1.0 {
                return Err(ProtocolError::InvalidParam(format!(
                    "{name}: alternative probability must be between 0 and 1"
                )));
            }
        }

        Ok(TypedParam {
            name,
            param_type,
            value,
            uncertainty: self.uncertainty,
            alternatives: self.alternatives,
            source: self.source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_string_param() {
        let param = ParamBuilder::new()
            .name("query")
            .param_type(ParamType::String)
            .value(ParamValue::String("hello world".into()))
            .uncertainty(0.1)
            .source(ParamSource::User)
            .build()
            .unwrap();

        assert_eq!(param.name, "query");
        assert_eq!(param.param_type, ParamType::String);
        assert!((param.uncertainty.value() - 0.1).abs() < 1e-10);
        assert_eq!(param.source, Some(ParamSource::User));
    }

    #[test]
    fn test_build_number_param() {
        let param = ParamBuilder::new()
            .name("count")
            .param_type(ParamType::Number)
            .value(ParamValue::Number(42.0))
            .build()
            .unwrap();

        assert_eq!(param.name, "count");
        assert_eq!(param.param_type, ParamType::Number);
    }

    #[test]
    fn test_type_mismatch() {
        let result = ParamBuilder::new()
            .name("query")
            .param_type(ParamType::String)
            .value(ParamValue::Number(42.0))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_missing_name() {
        let result = ParamBuilder::new()
            .param_type(ParamType::String)
            .value(ParamValue::String("test".into()))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_with_alternatives() {
        let param = ParamBuilder::new()
            .name("city")
            .param_type(ParamType::String)
            .value(ParamValue::String("New York".into()))
            .uncertainty(0.4)
            .alternative(ParamValue::String("Newark".into()), 0.3)
            .alternative(ParamValue::String("New Haven".into()), 0.2)
            .build()
            .unwrap();

        assert_eq!(param.alternatives.len(), 2);
        assert!((param.alternatives[0].probability - 0.3).abs() < 1e-10);
    }
}
