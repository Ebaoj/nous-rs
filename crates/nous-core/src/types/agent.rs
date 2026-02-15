/// Agent type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum AgentType {
    Claude,
    Gpt,
    Human,
    System,
    Custom(String),
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Claude => write!(f, "claude"),
            Self::Gpt => write!(f, "gpt"),
            Self::Human => write!(f, "human"),
            Self::System => write!(f, "system"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Identifies an agent in the Nous protocol
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AgentIdentifier {
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub agent_type: AgentType,
    pub id: String,
    pub version: Option<String>,
}

impl AgentIdentifier {
    pub fn new(agent_type: AgentType, id: impl Into<String>) -> Self {
        Self {
            agent_type,
            id: id.into(),
            version: None,
        }
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn system(id: impl Into<String>) -> Self {
        Self::new(AgentType::System, id)
    }
}
