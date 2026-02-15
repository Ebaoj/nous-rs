use crate::types::{ParamType, ParamValue, TypedParam};

/// Result of validating a single parameter's type.
#[derive(Debug, Clone)]
pub struct ParamValidationResult {
    pub valid: bool,
    pub error: Option<String>,
}

/// Validate that a parameter's value matches its declared type.
pub fn validate_param_type(param: &TypedParam) -> ParamValidationResult {
    let type_matches = matches!(
        (&param.param_type, &param.value),
        (ParamType::String, ParamValue::String(_))
            | (ParamType::Number, ParamValue::Number(_))
            | (ParamType::Boolean, ParamValue::Boolean(_))
            | (ParamType::Array, ParamValue::Array(_))
            | (ParamType::Object, ParamValue::Object(_))
            | (ParamType::Embedding, ParamValue::Embedding(_))
    );

    if type_matches {
        ParamValidationResult {
            valid: true,
            error: None,
        }
    } else {
        let actual = param.value.param_type();
        ParamValidationResult {
            valid: false,
            error: Some(format!(
                "{}: expected {:?}, got {:?}",
                param.name, param.param_type, actual
            )),
        }
    }
}

/// Validate all parameters in a list.
/// Returns a list of error strings (empty if all valid).
pub fn validate_params(params: &[TypedParam]) -> Vec<String> {
    let mut errors = Vec::new();

    for param in params {
        let result = validate_param_type(param);
        if let Some(err) = result.error {
            errors.push(err);
        }

        // Validate uncertainty is in range
        let u = param.uncertainty.value();
        if !(0.0..=1.0).contains(&u) {
            errors.push(format!(
                "{}: uncertainty must be between 0 and 1",
                param.name
            ));
        }

        // Validate alternatives have valid probabilities
        for alt in &param.alternatives {
            if !(0.0..=1.0).contains(&alt.probability) {
                errors.push(format!(
                    "{}: alternative probability must be between 0 and 1",
                    param.name
                ));
            }
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use nous_core::types::Confidence;

    use super::*;

    fn string_param(name: &str, value: &str) -> TypedParam {
        TypedParam {
            name: name.into(),
            param_type: ParamType::String,
            value: ParamValue::String(value.into()),
            uncertainty: Confidence::zero(),
            alternatives: Vec::new(),
            source: None,
        }
    }

    #[test]
    fn test_validate_matching_type() {
        let param = string_param("query", "hello");
        let result = validate_param_type(&param);
        assert!(result.valid);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_validate_mismatched_type() {
        let param = TypedParam {
            name: "count".into(),
            param_type: ParamType::Number,
            value: ParamValue::String("not a number".into()),
            uncertainty: Confidence::zero(),
            alternatives: Vec::new(),
            source: None,
        };
        let result = validate_param_type(&param);
        assert!(!result.valid);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_validate_params_all_valid() {
        let params = vec![
            string_param("a", "hello"),
            TypedParam {
                name: "b".into(),
                param_type: ParamType::Number,
                value: ParamValue::Number(42.0),
                uncertainty: Confidence::new(0.1),
                alternatives: Vec::new(),
                source: None,
            },
        ];
        let errors = validate_params(&params);
        assert!(errors.is_empty());
    }
}
