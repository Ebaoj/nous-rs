use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("build error: missing field {0}")]
    MissingField(String),
    #[error("invalid parameter: {0}")]
    InvalidParam(String),
    #[error("contract error: {0}")]
    Contract(String),
    #[error(transparent)]
    Core(#[from] nous_core::error::NousError),
}

pub type ProtocolResult<T> = Result<T, ProtocolError>;
