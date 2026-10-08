"""
inference/backend.py
--------------------
Abstract interface that every inference backend must implement.

Design principles
-----------------
* The interface is intentionally minimal: ``infer`` + ``is_available``.
* Backends must never fabricate "real model" identity — the ``backend``
  field on :class:`~inference.result.InferenceResult` must accurately
  reflect which backend produced the result.
* ``elapsed_ms`` inside the result should measure only the time the
  backend itself spends, not any surrounding cache or router logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from inference.config import ModelConfig
from inference.result import InferenceResult


class InferenceBackend(ABC):
    """Generic interface for executing a single inference request.

    Sub-classes
    -----------
    Concrete backends (e.g. :class:`~inference.mock_backend.MockBackend`,
    a future ``OpenAIBackend``, ``HuggingFaceBackend``, …) must implement
    the two abstract methods below.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique identifier for this backend (e.g. ``"mock"``).

        Used to populate :attr:`~inference.result.InferenceResult.backend`.
        Must be a non-empty string and should be URL-safe (lowercase,
        hyphens/underscores only).
        """

    @abstractmethod
    def infer(self, prompt: str, config: ModelConfig) -> InferenceResult:
        """Execute one inference call and return a result.

        Parameters
        ----------
        prompt:
            The raw prompt string to send to the model.
        config:
            Model configuration (id, size tier, token budget, …).

        Returns
        -------
        InferenceResult
            A fully populated result.  ``is_cached`` must be ``False``
            here — caching is the router's responsibility, not the
            backend's.

        Raises
        ------
        RuntimeError
            If the backend is unreachable or returns an error response.
        ValueError
            If *prompt* is empty or *config* is invalid for this backend.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend is currently reachable.

        For live backends this should perform a lightweight health-check
        (e.g. a ``/health`` ping).  For the :class:`~inference.mock_backend.MockBackend`
        this always returns ``True``.
        """
