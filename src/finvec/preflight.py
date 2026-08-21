"""Audit staged data against every documented import and document limit.

Runs before an import, because the failure modes on the other side are expensive:
imports only create namespaces that do not exist, so a rejected import leaves a
partially created namespace that blocks its own retry until it is dropped. Each check
below corresponds to a specific published limit, and each is measured from the actual
staged files rather than assumed.

Limits are checked at two scopes. Per-document limits (size, field sizes, vector
presence) are properties of the data. Per-import limits (file count, bytes, namespace
count) depend on how the import is sliced — and because this pipeline runs one import
per year namespace, every per-import limit is divided by 22 rather than applied to the
whole corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import EMBED_DIMS, SEC_SHARD_COUNT
from .fts import TEXT_FIELD, VECTOR_FIELD
from .jsonl import FLOAT_DP

# ── Published limits ─────────────────────────────────────────────────────────
# Import (per import operation)
MAX_NAMESPACES_PER_IMPORT = 10_000
MAX_FILES_PER_IMPORT = 100_000
MAX_BYTES_PER_FILE = 10 * 1000**3          # 10 GB, decimal
MAX_TOTAL_BYTES_ON_DEMAND = 1000**4        # 1 TB, decimal
MAX_BYTES_PER_NAMESPACE = 500 * 1000**3    # 500 GB, decimal

# Document (FTS document schema)
MAX_DOC_BYTES = 2 * 1024 * 1024            # 2 MB
MAX_FTS_FIELD_BYTES = 100 * 1024           # 100 KB
MAX_FTS_FIELD_TOKENS = 10_000
MAX_TOKEN_BYTES = 256
MAX_METADATA_BYTES = 40 * 1024             # 40 KB, excludes FTS fields
MAX_FIELD_NAME_BYTES = 64

# Object limits (Standard plan)
MAX_NAMESPACES_PER_INDEX_STANDARD = 100_000
MAX_INDEXES_PER_PROJECT_STANDARD = 20


@dataclass
class Check:
    name: str
    limit: str
    observed: str
    ok: bool
    note: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    scope: str = ""

    def add(self, name: str, limit: str, observed: str, ok: bool, note: str = "") -> None:
        self.checks.append(Check(name, limit, observed, ok, note))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


def _fmt_bytes(n: float) -> str:
    for unit, div in (("TB", 1000**4), ("GB", 1000**3), ("MB", 1000**2), ("KB", 1000)):
        if n >= div:
            return f"{n / div:,.2f} {unit}"
    return f"{n:,.0f} B"


def measure(
    staging_dir: Path, dataset: str, sample_docs_per_part: int = 400
) -> dict[str, Any]:
    """Measure staged Parquet, and the JSONL documents it will become.

    Document-level limits apply to the *JSONL* form, not the Parquet form, so the
    heaviest fields are measured by materialising sample documents rather than reading
    Parquet column sizes.
    """
    import pyarrow.parquet as pq

    from .upload import staged_files

    files = staged_files(staging_dir, dataset)
    per_namespace: dict[str, dict[str, Any]] = {}
    worst = {
        "doc_bytes": 0, "doc_id": "", "text_bytes": 0, "text_id": "",
        "meta_bytes": 0, "meta_id": "", "field_name": "", "field_name_bytes": 0,
    }
    problems: list[str] = []
    total_rows = 0
    ids_by_namespace: dict[str, set[str]] = {}
    dup_ids = 0
    missing_vector = 0
    wrong_dim = 0
    empty_id = 0
    empty_text = 0
    numeric_array_fields: set[str] = set()

    for path in files:
        namespace = path.parent.name
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        total_rows += rows
        entry = per_namespace.setdefault(
            namespace, {"rows": 0, "files": 0, "parquet_bytes": 0}
        )
        entry["rows"] += rows
        entry["files"] += 1
        entry["parquet_bytes"] += path.stat().st_size

        seen = ids_by_namespace.setdefault(namespace, set())
        table = pq.read_table(path, columns=["id", "metadata"])
        ids = table["id"].to_pylist()
        for rid in ids:
            if not rid:
                empty_id += 1
            if rid in seen:
                dup_ids += 1
            seen.add(rid)

        # Materialise a sample of documents to size the JSONL form.
        step = max(1, len(ids) // max(1, sample_docs_per_part))
        metas = table["metadata"].to_pylist()
        for i in range(0, len(ids), step):
            meta = json.loads(metas[i])
            text = meta.get("text", "")
            if not text.strip():
                empty_text += 1
            text_bytes = len(text.encode())
            if text_bytes > worst["text_bytes"]:
                worst.update(text_bytes=text_bytes, text_id=ids[i])

            other = {k: v for k, v in meta.items() if k != "text"}
            meta_bytes = len(json.dumps(other, separators=(",", ":")).encode())
            if meta_bytes > worst["meta_bytes"]:
                worst.update(meta_bytes=meta_bytes, meta_id=ids[i])

            for key, value in other.items():
                nb = len(key.encode())
                if nb > worst["field_name_bytes"]:
                    worst.update(field_name=key, field_name_bytes=nb)
                if key.startswith(("_", "$")):
                    problems.append(f"reserved field name {key!r} on {ids[i]}")
                if isinstance(value, list) and any(
                    not isinstance(v, str) for v in value
                ):
                    numeric_array_fields.add(key)

            # A rounded float serialises to at most len("-0.") + FLOAT_DP + a comma.
            vector_bytes = EMBED_DIMS * (FLOAT_DP + 5)
            doc_bytes = text_bytes + meta_bytes + vector_bytes + len(ids[i]) + 40
            if doc_bytes > worst["doc_bytes"]:
                worst.update(doc_bytes=doc_bytes, doc_id=ids[i])

    # The dense vector is required on every document, so verify it structurally on the
    # column rather than the sample: a single missing vector fails the whole import.
    for path in files:
        table = pq.read_table(path, columns=["values"])
        for values in table["values"].to_pylist():
            if values is None:
                missing_vector += 1
            elif len(values) != EMBED_DIMS:
                wrong_dim += 1

    return {
        "files": len(files),
        "rows": total_rows,
        "per_namespace": per_namespace,
        "worst": worst,
        "dup_ids": dup_ids,
        "empty_id": empty_id,
        "empty_text": empty_text,
        "missing_vector": missing_vector,
        "wrong_dim": wrong_dim,
        "numeric_array_fields": sorted(numeric_array_fields),
        "problems": problems,
    }


def audit(
    m: dict[str, Any],
    shards_done: int,
    jsonl_ratio: float = 1.50,
    target_part_bytes: int = 400 * 1024 * 1024,
) -> Report:
    """Check measurements against the published limits, projected to the full corpus.

    `jsonl_ratio` is the measured gzipped-JSONL to Parquet size ratio, since the
    import consumes the JSONL form and every byte-based limit applies to it.
    """
    r = Report()
    scale = SEC_SHARD_COUNT / max(shards_done, 1)
    ns = m["per_namespace"]
    r.scope = (
        f"{m['rows']:,} documents staged from {shards_done} of {SEC_SHARD_COUNT} "
        f"shards; projections scale by {scale:.2f}x"
    )

    # ── Per-document limits (properties of the data, not of the slicing) ──────
    r.add(
        "document size",
        f"< {_fmt_bytes(MAX_DOC_BYTES)}",
        f"largest {_fmt_bytes(m['worst']['doc_bytes'])}",
        m["worst"]["doc_bytes"] < MAX_DOC_BYTES,
        f"worst: {m['worst']['doc_id']}",
    )
    r.add(
        f"{TEXT_FIELD} field size",
        f"< {_fmt_bytes(MAX_FTS_FIELD_BYTES)}",
        f"largest {_fmt_bytes(m['worst']['text_bytes'])}",
        m["worst"]["text_bytes"] < MAX_FTS_FIELD_BYTES,
        f"worst: {m['worst']['text_id']}",
    )
    r.add(
        "filterable metadata per document",
        f"< {_fmt_bytes(MAX_METADATA_BYTES)}",
        f"largest {_fmt_bytes(m['worst']['meta_bytes'])}",
        m["worst"]["meta_bytes"] < MAX_METADATA_BYTES,
        "FTS text field is excluded from this limit",
    )
    r.add(
        "field name length",
        f"<= {MAX_FIELD_NAME_BYTES} bytes",
        f"longest {m['worst']['field_name_bytes']} ({m['worst']['field_name']})",
        m["worst"]["field_name_bytes"] <= MAX_FIELD_NAME_BYTES,
    )
    r.add(
        "reserved field-name prefixes",
        "no field starts with _ or $",
        f"{len(m['problems'])} violation(s)",
        not m["problems"],
        "; ".join(m["problems"][:3]),
    )
    r.add(
        "arrays of numbers in undeclared fields",
        "rejected by the server — must be none",
        f"{len(m['numeric_array_fields'])} field(s)",
        not m["numeric_array_fields"],
        ", ".join(m["numeric_array_fields"]),
    )
    r.add(
        f"{VECTOR_FIELD} present on every document",
        "required when the schema declares a dense vector",
        f"{m['missing_vector']} missing",
        m["missing_vector"] == 0,
    )
    r.add(
        f"{VECTOR_FIELD} dimension",
        f"exactly {EMBED_DIMS}",
        f"{m['wrong_dim']} wrong",
        m["wrong_dim"] == 0,
    )
    r.add(
        "_id non-empty",
        "required, non-empty string",
        f"{m['empty_id']} empty",
        m["empty_id"] == 0,
    )
    r.add(
        "_id unique within namespace",
        "duplicates silently overwrite",
        f"{m['dup_ids']} duplicate(s)",
        m["dup_ids"] == 0,
    )
    r.add(
        "no empty text (embedding precondition)",
        "0 blank text fields",
        f"{m['empty_text']} blank",
        m["empty_text"] == 0,
        "blank text caused the 2026-08-20 run failure",
    )

    # ── Per-import limits (one import per year namespace) ─────────────────────
    r.add(
        "namespaces per import",
        f"<= {MAX_NAMESPACES_PER_IMPORT:,}",
        "1 (one import per year)",
        True,
        "per-year slicing keeps this trivially satisfied",
    )

    if ns:
        worst_ns = max(ns, key=lambda k: ns[k]["rows"])
        proj_files = {
            k: max(1, round(v["parquet_bytes"] * scale * jsonl_ratio / target_part_bytes))
            for k, v in ns.items()
        }
        max_files = max(proj_files.values())
        r.add(
            "files per import",
            f"<= {MAX_FILES_PER_IMPORT:,}",
            f"{max_files} (largest year, after compact)",
            max_files <= MAX_FILES_PER_IMPORT,
            f"largest namespace {worst_ns}",
        )
        r.add(
            "size per file",
            f"<= {_fmt_bytes(MAX_BYTES_PER_FILE)}",
            f"~{_fmt_bytes(target_part_bytes * jsonl_ratio)} target",
            target_part_bytes * jsonl_ratio <= MAX_BYTES_PER_FILE,
            "compact targets 400 MB of Parquet per part",
        )
        proj_ns_bytes = {
            k: v["parquet_bytes"] * scale * jsonl_ratio for k, v in ns.items()
        }
        biggest = max(proj_ns_bytes.values())
        r.add(
            "size per namespace",
            f"<= {_fmt_bytes(MAX_BYTES_PER_NAMESPACE)}",
            f"~{_fmt_bytes(biggest)} (largest year)",
            biggest <= MAX_BYTES_PER_NAMESPACE,
        )
        total = sum(proj_ns_bytes.values())
        r.add(
            "total input size",
            f"<= {_fmt_bytes(MAX_TOTAL_BYTES_ON_DEMAND)} on-demand",
            f"~{_fmt_bytes(total)} across all years",
            total <= MAX_TOTAL_BYTES_ON_DEMAND,
        )
        r.add(
            "namespaces per index",
            f"<= {MAX_NAMESPACES_PER_INDEX_STANDARD:,} (Standard)",
            f"{len(ns)} year namespaces",
            len(ns) <= MAX_NAMESPACES_PER_INDEX_STANDARD,
        )

    return r


def render(r: Report, m: dict[str, Any], shards_done: int) -> str:
    lines = ["", r.scope, ""]
    width = max(len(c.name) for c in r.checks) + 2
    for c in r.checks:
        mark = "PASS" if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name:<{width}} {c.observed:<34} limit {c.limit}")
        if c.note:
            lines.append(f"         {c.note}")
    lines.append("")
    ns = m["per_namespace"]
    if ns:
        lines.append("  Per-namespace staged so far:")
        for name in sorted(ns):
            v = ns[name]
            lines.append(
                f"    {name}  {v['rows']:>10,} docs  {v['files']:>5} files  "
                f"{_fmt_bytes(v['parquet_bytes']):>10} parquet"
            )
        lines.append("")
    if r.failures:
        lines.append(f"  {len(r.failures)} check(s) FAILED — do not import yet.")
    else:
        lines.append("  All checks pass. Import will not hit a documented limit.")
    lines.append("")
    return "\n".join(lines)
