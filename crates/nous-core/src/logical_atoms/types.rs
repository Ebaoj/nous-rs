/// Quantifier type extracted from text
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum QuantifierType {
    All,
    Some,
    None,
    Most,
    Few,
    Exactly,
    AtLeast,
    AtMost,
}

/// Extracted quantifier with context
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtractedQuantifier {
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub quantifier_type: QuantifierType,
    pub scope: String,
    pub original: String,
}

/// Extracted negation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtractedNegation {
    pub target: String,
    pub negated: bool,
    pub original: String,
}

/// Extracted number/quantity
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtractedNumber {
    pub name: String,
    pub value: f64,
    pub unit: Option<String>,
    pub original: String,
}

/// Constraint type
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum ConstraintType {
    Max,
    Min,
    Exact,
    Range,
    Before,
    After,
}

/// Constraint value — single number or range
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum ConstraintValue {
    Single(f64),
    Range(f64, f64),
}

/// Extracted constraint/bound
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtractedConstraint {
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub constraint_type: ConstraintType,
    pub value: ConstraintValue,
    pub target: String,
    pub original: String,
}

/// Ordering type
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum OrderingType {
    Before,
    After,
    During,
    While,
    Until,
    Then,
}

/// Extracted temporal/causal ordering
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtractedOrdering {
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub ordering_type: OrderingType,
    pub first: String,
    pub second: String,
    pub original: String,
}

/// Complete logical atoms extracted from text
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogicalAtoms {
    pub quantifiers: Vec<ExtractedQuantifier>,
    pub negations: Vec<ExtractedNegation>,
    pub numbers: Vec<ExtractedNumber>,
    pub constraints: Vec<ExtractedConstraint>,
    pub orderings: Vec<ExtractedOrdering>,
    pub source_text: String,
}

impl LogicalAtoms {
    /// Create empty atoms
    pub fn empty(text: impl Into<String>) -> Self {
        Self {
            quantifiers: Vec::new(),
            negations: Vec::new(),
            numbers: Vec::new(),
            constraints: Vec::new(),
            orderings: Vec::new(),
            source_text: text.into(),
        }
    }

    /// Check if no logical atoms were extracted
    pub fn is_empty(&self) -> bool {
        self.quantifiers.is_empty()
            && self.negations.is_empty()
            && self.numbers.is_empty()
            && self.constraints.is_empty()
            && self.orderings.is_empty()
    }

    /// Format for display
    pub fn format(&self) -> String {
        let mut lines = Vec::new();

        if !self.quantifiers.is_empty() {
            lines.push("Quantifiers:".to_string());
            for q in &self.quantifiers {
                lines.push(format!("  {:?} {}", q.quantifier_type, q.scope));
            }
        }

        if !self.negations.is_empty() {
            lines.push("Negations:".to_string());
            for n in &self.negations {
                let label = if n.negated { "NOT" } else { "IS" };
                lines.push(format!("  {label} {}", n.target));
            }
        }

        if !self.numbers.is_empty() {
            lines.push("Numbers:".to_string());
            for n in &self.numbers {
                let unit = n.unit.as_deref().unwrap_or("");
                lines.push(format!("  {} = {}{}", n.name, n.value, unit));
            }
        }

        if !self.constraints.is_empty() {
            lines.push("Constraints:".to_string());
            for c in &self.constraints {
                lines.push(format!("  {:?} {} = {:?}", c.constraint_type, c.target, c.value));
            }
        }

        if !self.orderings.is_empty() {
            lines.push("Orderings:".to_string());
            for o in &self.orderings {
                lines.push(format!("  {} {:?} {}", o.first, o.ordering_type, o.second));
            }
        }

        if lines.is_empty() {
            "(no logical atoms extracted)".to_string()
        } else {
            lines.join("\n")
        }
    }
}
