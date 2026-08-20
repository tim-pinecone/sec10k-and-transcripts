"""The common shape every source produces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Record:
    """One embeddable unit, already assigned its ID, namespace, and metadata.

    `namespace` is the year the record belongs to — it decides which staged
    subdirectory the record lands in, and therefore which Pinecone namespace the
    import writes it to.
    """

    id: str
    namespace: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
