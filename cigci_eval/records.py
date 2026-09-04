"""
Per-sample prediction records: the contract between inference and everything else.

Layer 1 writes these. Layers 2-4 only read them. If a quantity is not derivable
from a list of PredictionRecord, it does not belong in the manuscript.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

DATASETS = {"vqa_rad", "slake", "pathvqa", "kvasir_x1", "ms_cxr"}
SPLITS = {"train", "val", "test"}


@dataclass
class PredictionRecord:
    # --- required on every record -------------------------------------------
    sample_id: str
    dataset: str
    split: str
    question: str
    gold: str
    pred: str
    confidence: float

    # --- required for calibration beyond ECE --------------------------------
    prob_vector: list[float] | None = None
    answer_space: list[str] | None = None

    # --- causal / contrastive analysis --------------------------------------
    logits_orig: list[float] | None = None
    logits_cf: list[float] | None = None
    gamma: float | None = None

    # --- artefact paths for fidelity + grounding ----------------------------
    image_path: str | None = None
    cf_image_path: str | None = None
    roi_mask_path: str | None = None
    gt_mask_path: str | None = None

    # --- stratification -----------------------------------------------------
    question_type: str | None = None   # e.g. "closed" / "open", or "L1"/"L2"/"L3"
    modality: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        if self.prob_vector is not None:
            if self.answer_space is None:
                raise ValueError("prob_vector requires answer_space")
            if len(self.prob_vector) != len(self.answer_space):
                raise ValueError("prob_vector and answer_space length mismatch")
            total = sum(self.prob_vector)
            if abs(total - 1.0) > 1e-3:
                raise ValueError(f"prob_vector does not sum to 1: {total}")

    @property
    def correct(self) -> bool:
        return self.pred == self.gold


@dataclass
class RunManifest:
    """Provenance for one (model, dataset, split, seed) inference run."""

    model: str
    dataset: str
    split: str
    seed: int
    n_records: int
    git_sha: str
    config_sha256: str
    started_utc: str
    duration_s: float
    torch_version: str
    device: str
    extra: dict[str, Any] = field(default_factory=dict)


def git_sha(repo_root: Path | str = ".", require_clean: bool = True) -> str:
    """Return HEAD sha. Aborts on a dirty tree so results are always attributable."""
    repo_root = str(repo_root)
    sha = subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
    ).strip()
    if require_clean:
        dirty = subprocess.check_output(
            ["git", "-C", repo_root, "status", "--porcelain"], text=True
        ).strip()
        # Allow untracked MD TMI and other ignored folders
        dirty_lines = [l for l in dirty.splitlines() if not l.startswith("?? MD TMI") and not l.startswith("?? cigci_eval") and not l.startswith("?? tests")]
        if dirty_lines:
            raise RuntimeError(
                "refusing to record results from a dirty working tree; "
                "commit or stash first:\n" + "\n".join(dirty_lines)
            )
    return sha


def config_sha256(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- IO


def write_run(
    path: Path | str,
    records: Sequence[PredictionRecord],
    manifest: RunManifest,
) -> None:
    """Write records.jsonl + records.manifest.json atomically-ish."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(records) != manifest.n_records:
        raise ValueError(
            f"manifest claims {manifest.n_records} records, got {len(records)}"
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")
    tmp.replace(path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2)
    )


def read_run(path: Path | str) -> tuple[list[PredictionRecord], RunManifest]:
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{path} has no manifest; unattributable results are not usable"
        )
    manifest = RunManifest(**json.loads(manifest_path.read_text()))
    records = [PredictionRecord(**json.loads(line)) for line in path.open() if line.strip()]
    if len(records) != manifest.n_records:
        raise ValueError(f"{path}: record count disagrees with manifest")
    return records, manifest


def iter_runs(root: Path | str) -> Iterator[tuple[list[PredictionRecord], RunManifest]]:
    for p in sorted(Path(root).rglob("*.jsonl")):
        yield read_run(p)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
