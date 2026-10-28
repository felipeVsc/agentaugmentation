"""Model-calling layer shared by Strategy A and Strategy B.

- ``providers``: one interface (``Provider.generate``) over API and local models,
  with a registry so new providers can be plugged in from config.
- ``caller``: cache lookup, rate limiting, retries and energy measurement around a provider.
- ``cache``, ``energy``, ``tokenizers``: the pieces the caller uses.
"""

from .config import ModelConfig
from .types import CallFailed, ProviderError, Request, Response

__all__ = ["ModelConfig", "Request", "Response", "ProviderError", "CallFailed"]
