use std::collections::HashMap;
use super::agent::{AgentIdentifier, AgentType};
use super::embedding::EmbeddingModel;

/// Nous protocol configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NousConfig {
    /// Default embedding model
    pub embedding_model: EmbeddingModel,
    /// OpenAI API key (for embeddings)
    pub openai_api_key: Option<String>,
    /// Default agent identifier
    pub default_agent: AgentIdentifier,
    /// Confidence propagation factors per transform type
    pub propagation_factors: HashMap<String, f64>,
}

impl Default for NousConfig {
    fn default() -> Self {
        let mut factors = HashMap::new();
        factors.insert("merge".into(), 0.95);
        factors.insert("summarization".into(), 0.85);
        factors.insert("inference".into(), 0.70);
        factors.insert("decode".into(), 0.80);

        Self {
            embedding_model: EmbeddingModel::default(),
            openai_api_key: None,
            default_agent: AgentIdentifier::new(AgentType::System, "nous-core"),
            propagation_factors: factors,
        }
    }
}
