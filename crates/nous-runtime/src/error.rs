use thiserror::Error;

/// Errors emitted by the nous-runtime crate.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("execution error: {0}")]
    Execution(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("handler not found: {0}")]
    HandlerNotFound(String),

    #[error("timeout after {0}ms")]
    Timeout(u64),

    #[error("memory error: {0}")]
    Memory(String),

    #[error("concept error: {0}")]
    Concept(String),

    #[error(transparent)]
    Core(#[from] nous_core::error::NousError),

    #[error(transparent)]
    Protocol(#[from] nous_protocol::error::ProtocolError),
}

/// Convenience alias for `Result<T, RuntimeError>`.
pub type RuntimeResult<T> = Result<T, RuntimeError>;
