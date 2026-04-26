"""
Representation base class and auto-registry.

Every concrete representation registers itself by inheriting from BaseRepr
and setting a class-level `key` attribute.  The registry is populated at
import time — just importing a repr module is enough to register it.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type

from core.board import SokobanBoard


_REGISTRY: Dict[str, Type["BaseRepr"]] = {}


class BaseRepr(ABC):
    """Abstract base for all board representations."""

    key: str = ""  # e.g. "01_ASCII_RAW"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.key:
            _REGISTRY[cls.key] = cls

    @abstractmethod
    def render(self, board: "SokobanBoard") -> str:
        """Return the representation as a string."""

    def __call__(self, board: "SokobanBoard") -> str:
        return self.render(board)


def get_repr(key: str) -> "BaseRepr":
    """Instantiate a representation by key."""
    if key not in _REGISTRY:
        raise KeyError(f"Unknown representation: {key!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def all_representations(board: "SokobanBoard") -> Dict[str, str]:
    """Render all registered representations for a board."""
    return {key: cls().render(board) for key, cls in sorted(_REGISTRY.items())}


def registered_keys():
    return sorted(_REGISTRY.keys())