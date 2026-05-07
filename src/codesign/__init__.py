"""codesign: adversarial attacks and mech-interp for code-LLM SVD."""

from codesign.attacker import (
    AttackTrace,
    ControlFlowFlattening,
    DeadCodeInsertion,
    DocstringInsertion,
    EquivalentExpressionSubstitution,
    InlineCommentInsertion,
    MutationStep,
    NaturalIdentifierRenaming,
    RLAdversary,
    TypeAnnotationsAdded,
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
from codesign.targets import (
    HeuristicClassifier,
    HFPipelineClassifier,
    OllamaPromptClassifier,
    make_target,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "AttackResult",
    "AttackTrace",
    "BenchmarkSummary",
    "ControlFlowFlattening",
    "DeadCodeInsertion",
    "DocstringInsertion",
    "EquivalentExpressionSubstitution",
    "HeuristicClassifier",
    "HFPipelineClassifier",
    "InlineCommentInsertion",
    "MutationStep",
    "NaturalIdentifierRenaming",
    "NodeWrapper",
    "OllamaPromptClassifier",
    "ProgramGraph",
    "ProgramGraphExtractor",
    "RLAdversary",
    "Sample",
    "TypeAnnotationsAdded",
    "VariableRenaming",
    "confidence_drop",
    "dfg_preserved",
    "evaded",
    "load_samples",
    "make_target",
    "parse_valid",
    "summarise",
]
