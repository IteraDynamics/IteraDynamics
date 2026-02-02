"""
Model Version Lineage and Provenance Tracking

Tracks model versions, their relationships (parent-child), and metric evolution
over time. Helps understand how models improve through retraining, tuning, and
governance adjustments.

Key concepts:
- Version nodes: Individual model versions with metrics
- Lineage edges: Parent-child relationships between versions
- Metric deltas: How metrics change between versions
- Provenance: What action led to each new version (retrain, tune, etc.)

Example:
    ```python
    from . import (
        VersionNode,
        ModelLineage,
        build_lineage_graph
    )

    # Define model versions
    versions = {
        "v0.7.0": VersionNode(
            version="v0.7.0",
            parent=None,
            precision=0.75,
            recall=0.70,
            f1=0.72,
            ece=0.06,
            derived_from="initial"
        ),
        "v0.7.1": VersionNode(
            version="v0.7.1",
            parent="v0.7.0",
            precision=0.78,
            recall=0.72,
            f1=0.75,
            ece=0.055,
            derived_from="retrain"
        ),
        "v0.7.2": VersionNode(
            version="v0.7.2",
            parent="v0.7.1",
            precision=0.80,
            recall=0.73,
            f1=0.76,
            ece=0.050,
            derived_from="threshold_adjustment"
        )
    }

    # Build lineage
    lineage = build_lineage_graph(versions)

    # Analyze improvements
    for edge in lineage.edges:
        print(f"{edge.parent} → {edge.child}: Δprecision = {edge.precision_delta:+.2f}")
    ```
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Version directory pattern (e.g., v0.7.0, v1.2.3)
VERSION_DIR_RE = re.compile(r"^v\d+\.\d+(\.\d+)?$")


# -----------------------
# Data Models
# -----------------------

@dataclass
class VersionNode:
    """
    A single model version with metrics and metadata.

    Attributes:
        version: Version identifier (e.g., "v0.7.0")
        parent: Parent version this was derived from
        precision: Precision metric
        recall: Recall metric
        f1: F1 score
        ece: Expected Calibration Error
        brier: Brier score
        trigger_count: Number of predictions/triggers made
        label_count: Number of labeled examples used for training
        derived_from: Action that created this version (e.g., "retrain", "tune")
        node_size: Visual size for graphing (auto-computed)
    """
    version: str
    parent: Optional[str] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    ece: Optional[float] = None
    brier: Optional[float] = None
    trigger_count: Optional[int] = None
    label_count: Optional[int] = None
    derived_from: Optional[str] = None
    node_size: float = 1.0

    def __post_init__(self):
        """Auto-compute node size based on label/trigger count."""
        if self.label_count and self.label_count > 0:
            self.node_size = max(1.0, math.sqrt(float(self.label_count)))
        elif self.trigger_count and self.trigger_count > 0:
            self.node_size = max(1.0, math.sqrt(float(self.trigger_count) * 0.5))
        else:
            self.node_size = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version": self.version,
            "parent": self.parent,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "ece": self.ece,
            "brier": self.brier,
            "trigger_count": self.trigger_count,
            "label_count": self.label_count,
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VersionNode:
        """Create from dictionary representation."""
        return cls(
            version=data["version"],
            parent=data.get("parent"),
            precision=data.get("precision"),
            recall=data.get("recall"),
            f1=data.get("f1") or data.get("F1"),
            ece=data.get("ece"),
            brier=data.get("brier"),
            trigger_count=data.get("trigger_count"),
            label_count=data.get("label_count"),
            derived_from=data.get("derived_from"),
        )


@dataclass
class LineageEdge:
    """
    An edge in the lineage graph representing version evolution.

    Attributes:
        parent: Parent version
        child: Child version derived from parent
        precision_delta: Change in precision (child - parent)
        recall_delta: Change in recall
        f1_delta: Change in F1 score
        ece_delta: Change in ECE (negative is improvement)
        action: Action that created child from parent
    """
    parent: str
    child: str
    precision_delta: Optional[float] = None
    recall_delta: Optional[float] = None
    f1_delta: Optional[float] = None
    ece_delta: Optional[float] = None
    action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "parent": self.parent,
            "child": self.child,
            "precision_delta": self.precision_delta,
            "recall_delta": self.recall_delta,
            "f1_delta": self.f1_delta,
            "ece_delta": self.ece_delta,
            "action": self.action,
        }


@dataclass
class ModelLineage:
    """
    Complete model lineage graph.

    Attributes:
        nodes: Dictionary of version -> VersionNode
        edges: List of lineage edges
        generated_at: Timestamp of lineage computation
    """
    nodes: Dict[str, VersionNode] = field(default_factory=dict)
    edges: List[LineageEdge] = field(default_factory=list)
    generated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "versions": [node.to_dict() for node in sorted(self.nodes.values(), key=lambda x: x.version)],
            "edges": [edge.to_dict() for edge in self.edges],
        }


# -----------------------
# Core Logic
# -----------------------

def compute_metric_delta(
    parent_value: Optional[float],
    child_value: Optional[float]
) -> Optional[float]:
    """
    Compute delta between parent and child metric values.

    Args:
        parent_value: Metric value for parent version
        child_value: Metric value for child version

    Returns:
        Delta (child - parent) or None if either is missing

    Example:
        >>> compute_metric_delta(0.75, 0.80)
        0.05
        >>> compute_metric_delta(None, 0.80)
        None
    """
    if parent_value is None or child_value is None:
        return None
    try:
        return float(child_value) - float(parent_value)
    except (ValueError, TypeError):
        return None


def build_lineage_edges(
    nodes: Dict[str, VersionNode]
) -> List[LineageEdge]:
    """
    Build lineage edges from nodes with parent relationships.

    Args:
        nodes: Dictionary of version -> VersionNode

    Returns:
        List of LineageEdge objects

    Example:
        >>> nodes = {
        ...     "v0.7.0": VersionNode("v0.7.0", parent=None, precision=0.75),
        ...     "v0.7.1": VersionNode("v0.7.1", parent="v0.7.0", precision=0.78)
        ... }
        >>> edges = build_lineage_edges(nodes)
        >>> len(edges)
        1
        >>> edges[0].precision_delta
        0.03
    """
    edges = []

    for child_node in nodes.values():
        if not child_node.parent:
            continue

        parent_node = nodes.get(child_node.parent)
        if not parent_node:
            # Parent not found in nodes
            continue

        # Compute deltas
        edge = LineageEdge(
            parent=child_node.parent,
            child=child_node.version,
            precision_delta=compute_metric_delta(parent_node.precision, child_node.precision),
            recall_delta=compute_metric_delta(parent_node.recall, child_node.recall),
            f1_delta=compute_metric_delta(parent_node.f1, child_node.f1),
            ece_delta=compute_metric_delta(parent_node.ece, child_node.ece),
            action=child_node.derived_from,
        )
        edges.append(edge)

    return edges


def build_lineage_graph(
    nodes: Dict[str, VersionNode]
) -> ModelLineage:
    """
    Build complete lineage graph from version nodes.

    Args:
        nodes: Dictionary of version -> VersionNode

    Returns:
        ModelLineage with nodes and edges

    Example:
        >>> nodes = {
        ...     "v0.7.0": VersionNode("v0.7.0", parent=None, precision=0.75),
        ...     "v0.7.1": VersionNode("v0.7.1", parent="v0.7.0", precision=0.78, derived_from="retrain")
        ... }
        >>> lineage = build_lineage_graph(nodes)
        >>> len(lineage.edges)
        1
    """
    edges = build_lineage_edges(nodes)

    return ModelLineage(
        nodes=nodes,
        edges=edges,
        generated_at=datetime.now(timezone.utc),
    )


def discover_versions_from_directories(
    models_dir: Path,
    metrics_files: List[str] = None
) -> Dict[str, VersionNode]:
    """
    Discover model versions from directory structure.

    Scans for directories matching version pattern (v*.*.*)  and reads
    metrics from JSON files in each version directory.

    Args:
        models_dir: Root models directory containing version subdirectories
        metrics_files: List of JSON filenames to check for metrics
            (default: ["metrics.json", "eval.json", "meta.json"])

    Returns:
        Dictionary of version -> VersionNode

    Example:
        >>> from pathlib import Path
        >>> # Assuming models/ has v0.7.0/, v0.7.1/ directories
        >>> nodes = discover_versions_from_directories(Path("models"))
        >>> "v0.7.0" in nodes
        True
    """
    if metrics_files is None:
        metrics_files = ["metrics.json", "eval.json", "meta.json"]

    nodes = {}

    if not models_dir.exists():
        return nodes

    for child in models_dir.iterdir():
        if not child.is_dir():
            continue

        name = child.name.strip()
        if not VERSION_DIR_RE.match(name):
            continue

        # Read parent version
        parent = None
        parent_file = child / "parent.txt"
        if parent_file.exists():
            try:
                parent = parent_file.read_text(encoding="utf-8").strip() or None
            except Exception:
                pass

        # Read metrics from JSON files
        metrics = {}
        for fname in metrics_files:
            metrics_path = child / fname
            if metrics_path.exists():
                try:
                    import json
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    break
                except Exception:
                    continue

        # Override parent from metrics if not in parent.txt
        if not parent:
            parent = metrics.get("parent") or metrics.get("derived_from_version")

        # Extract metrics safely
        def safe_float(key: str) -> Optional[float]:
            val = metrics.get(key)
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def safe_int(key: str) -> Optional[int]:
            val = metrics.get(key)
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        # Create node
        node = VersionNode(
            version=name,
            parent=parent if isinstance(parent, str) and parent else None,
            precision=safe_float("precision"),
            recall=safe_float("recall"),
            f1=safe_float("f1") or safe_float("F1"),
            ece=safe_float("ece"),
            brier=safe_float("brier"),
            trigger_count=safe_int("trigger_count"),
            label_count=safe_int("label_count"),
            derived_from=metrics.get("derived_from") or metrics.get("action"),
        )

        nodes[name] = node

    return nodes


def format_lineage_text(
    lineage: ModelLineage,
    include_deltas: bool = True
) -> List[str]:
    """
    Format lineage as human-readable text lines.

    Args:
        lineage: ModelLineage graph
        include_deltas: Whether to include metric deltas in output

    Returns:
        List of formatted text lines

    Example:
        >>> nodes = {
        ...     "v0.7.0": VersionNode("v0.7.0", parent=None, precision=0.75),
        ...     "v0.7.1": VersionNode("v0.7.1", parent="v0.7.0", precision=0.78, derived_from="retrain")
        ... }
        >>> lineage = build_lineage_graph(nodes)
        >>> lines = format_lineage_text(lineage)
        >>> any("v0.7.0 → v0.7.1" in line for line in lines)
        True
    """
    lines = []

    if not lineage.edges:
        lines.append("No lineage edges found.")
        return lines

    # Sort edges by parent version
    sorted_edges = sorted(lineage.edges, key=lambda e: e.parent)

    for edge in sorted_edges:
        parts = [f"{edge.parent} → {edge.child}"]

        if include_deltas:
            deltas = []
            if edge.precision_delta is not None:
                deltas.append(f"Δprecision {edge.precision_delta:+.2f}")
            if edge.f1_delta is not None:
                deltas.append(f"ΔF1 {edge.f1_delta:+.2f}")
            if edge.ece_delta is not None:
                deltas.append(f"ΔECE {edge.ece_delta:+.2f}")

            if deltas:
                parts.append(f"({', '.join(deltas)})")

        if edge.action:
            parts.append(f"[{edge.action}]")

        lines.append(" ".join(parts))

    return lines


__all__ = [
    "VersionNode",
    "LineageEdge",
    "ModelLineage",
    "compute_metric_delta",
    "build_lineage_edges",
    "build_lineage_graph",
    "discover_versions_from_directories",
    "format_lineage_text",
    "VERSION_DIR_RE",
]
