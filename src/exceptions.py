"""Project-wide exceptions."""


class MCTError(Exception):
    """Base error."""


class ModelRegistryError(MCTError):
    """configs/models.yaml and models/MANIFEST.yaml disagree or are invalid."""


class ModelRejectedError(MCTError):
    """A REJECTED artifact was requested. Never load or ship it."""


class ModelVerificationError(MCTError):
    """Post-download VERIFY failed. Artifact must be marked REJECTED."""


class ModelNotApprovedError(MCTError):
    """Attempted to load a model that is not APPROVED."""


class ConfigurationError(MCTError):
    """Invalid or missing configuration."""


class PersistenceError(MCTError):
    """Database / gallery persistence failure."""


class ProductionDatabaseError(MCTError):
    """Production requires PostgreSQL + pgvector."""


class IdentityUnresolved(MCTError):
    """No reliable identity assignment is available."""
