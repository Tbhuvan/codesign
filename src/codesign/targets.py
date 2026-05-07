"""SVD target classifiers: callable code -> [0,1] vuln confidence.

Three backends:

- ``ollama:<model>``     LLM-as-classifier, prompted for a binary
                         VULN/SAFE response, parsed to 0.95 / 0.05.
- ``hf:<model_id>``      HuggingFace text-classification pipeline,
                         interpreting LABEL_1 / "insecure" as the
                         vulnerable class.
- ``heuristic``          Simple substring rule used by tests / CI.

A factory ``make_target(spec)`` parses the colon-separated spec and
returns a callable. The same string identifies the target in
``experiments.json`` and benchmark JSON, so experiments are
self-describing.
"""

from __future__ import annotations

import logging
from typing import Protocol

import requests

log = logging.getLogger(__name__)


class TargetClassifier(Protocol):
    """Score code in [0, 1]; higher = more vulnerable."""

    spec: str

    def score(self, code: str) -> float: ...

    def info(self) -> dict: ...

    def __call__(self, code: str) -> float:
        return self.score(code)


class OllamaPromptClassifier:
    """Prompt an Ollama model to label code VULN or SAFE."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        n_shots: int = 0,
        shots: list[tuple[str, str]] | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.n_shots = max(0, int(n_shots))
        self.shots = shots or []
        self.spec = f"ollama:{model}"

    def _build_prompt(self, code: str) -> str:
        parts = [
            "You are a static code security classifier.",
            "Reply with exactly one word: VULN if the code contains a security",
            "vulnerability, otherwise SAFE.",
            "",
        ]
        for snippet, label in self.shots[: self.n_shots]:
            parts.append("Code:")
            parts.append(snippet.rstrip())
            parts.append(f"Label: {label}")
            parts.append("")
        parts.append("Code:")
        parts.append(code.rstrip())
        parts.append("Label:")
        return "\n".join(parts)

    def score(self, code: str) -> float:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self._build_prompt(code),
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 4},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            text = r.json().get("response", "").strip().upper()
        except (requests.RequestException, ValueError) as e:
            log.warning("ollama call failed: %s", e)
            return 0.5
        first = text.split()[0] if text else ""
        if first.startswith("VULN"):
            return 0.95
        if first.startswith("SAFE"):
            return 0.05
        return 0.5  # model didn't follow format

    def info(self) -> dict:
        return {
            "kind": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "n_shots": self.n_shots,
        }

    __call__ = score


class HFPipelineClassifier:
    """Wrap a HuggingFace text-classification pipeline.

    The model card semantics for vulnerability classifiers are
    inconsistent. Heuristic: label string starts with "label_1" /
    "unsafe" / "insecure" / "vulnerable" / "defective" -> vulnerable
    class. We expose ``positive_label`` and ``invert`` for explicit
    control when this guess is wrong.
    """

    DEFAULT_VULN_TOKENS = ("label_1", "unsafe", "insecure", "vulnerable", "defective")

    def __init__(
        self,
        model_id: str,
        max_chars: int = 512,
        positive_label: str | None = None,
        invert: bool = False,
    ) -> None:
        if not model_id:
            raise ValueError("model_id is required")
        self.model_id = model_id
        self.max_chars = max_chars
        self.positive_label = positive_label.lower() if positive_label else None
        self.invert = bool(invert)
        self._pipe = None
        self.spec = f"hf:{model_id}"

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:  # pragma: no cover
            raise ImportError("install torch + transformers") from e
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline("text-classification", model=self.model_id, device=device)

    def score(self, code: str) -> float:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        self._ensure_loaded()
        out = self._pipe(code[: self.max_chars])[0]  # type: ignore[index]
        label = out["label"].lower()
        score = float(out["score"])

        if self.positive_label is not None:
            is_vuln = self.positive_label in label
        else:
            is_vuln = any(t in label for t in self.DEFAULT_VULN_TOKENS)
        prob_vuln = score if is_vuln else 1.0 - score
        return 1.0 - prob_vuln if self.invert else prob_vuln

    def info(self) -> dict:
        return {
            "kind": "hf",
            "model_id": self.model_id,
            "max_chars": self.max_chars,
            "positive_label": self.positive_label,
            "invert": self.invert,
        }

    __call__ = score


class HeuristicClassifier:
    """Substring-based scorer used by tests and CI when no real model is available."""

    def __init__(self, base: float = 0.95) -> None:
        if not 0.0 <= base <= 1.0:
            raise ValueError("base must be in [0, 1]")
        self.base = base
        self.spec = "heuristic"

    def score(self, code: str) -> float:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        s = self.base
        if "_adv" in code:
            s -= 0.4
        if "_vrtg_decoy" in code:
            s -= 0.3
        if "if True:" in code:
            s -= 0.1
        return max(0.01, s)

    def info(self) -> dict:
        return {"kind": "heuristic", "base": self.base}

    __call__ = score


def make_target(spec: str) -> TargetClassifier:
    """Parse a target spec string into a classifier.

    >>> make_target("heuristic")
    >>> make_target("ollama:qwen3:8b")
    >>> make_target("hf:mrm8488/codebert-base-finetuned-detect-insecure-code")
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("spec must be a non-empty string")
    spec = spec.strip()
    if spec == "heuristic":
        return HeuristicClassifier()
    if spec.startswith("ollama:"):
        model = spec[len("ollama:"):]
        return OllamaPromptClassifier(model=model)
    if spec.startswith("hf:"):
        model_id = spec[len("hf:"):]
        return HFPipelineClassifier(model_id=model_id)
    raise ValueError(
        f"unknown target spec {spec!r}; expected 'heuristic', "
        "'ollama:<model>', or 'hf:<model_id>'"
    )


__all__ = [
    "TargetClassifier",
    "OllamaPromptClassifier",
    "HFPipelineClassifier",
    "HeuristicClassifier",
    "make_target",
]
