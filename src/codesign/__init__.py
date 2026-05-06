"""codesign — adversarial attacks and mech-interp for code-LLM SVD."""

from codesign.attacker import (
    AttackTrace,
    ControlFlowFlattening,
    DeadCodeInsertion,
    MutationStep,
    RLAdversary,
    VariableRenaming,
)
from codesign.dataset import Sample, load_samples
from codesign.metrics import (
    AttackResult,
    BenchmarkSummary,
    confidence_drop,
    dfg_preserved,
    evaded,
    parse_valid,
    summarise,
)
from codesign.parser import NodeWrapper, ProgramGraph, ProgramGraphExtractor

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "AttackResult",
    "AttackTrace",
    "BenchmarkSummary",
    "ControlFlowFlattening",
    "DeadCodeInsertion",
    "MutationStep",
    "NodeWrapper",
    "ProgramGraph",
    "ProgramGraphExtractor",
    "RLAdversary",
    "Sample",
    "VariableRenaming",
    "confidence_drop",
    "dfg_preserved",
    "evaded",
    "load_samples",
    "parse_valid",
    "summarise",
]
