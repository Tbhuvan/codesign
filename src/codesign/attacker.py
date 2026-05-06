"""MAB-driven problem-space attacker."""

from __future__ import annotations

import ast
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from codesign.parser import NodeWrapper

log = logging.getLogger(__name__)


class Mutator(Protocol):
    name: str

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str: ...


@dataclass
class VariableRenaming:
    """VRTG via byte-precise tree-sitter ranges.

    Naive str.replace breaks when identifiers share prefixes (i vs index).
    We sort by start_byte descending and rewrite in-place so earlier offsets
    stay valid as later bytes grow.
    """
    name: str = "variable_renaming"
    deny: frozenset[str] = field(default_factory=lambda: frozenset({
        "True", "False", "None", "self", "cls", "print", "os", "sys"
    }))
    suffix_range: tuple[int, int] = (100, 999)

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        if not variables:
            return code

        rename: dict[str, str] = {}
        ordered = sorted(variables, key=lambda n: n.start_byte, reverse=True)
        out = bytearray(code.encode("utf8"))

        for var in ordered:
            if var.val in self.deny:
                continue
            new = rename.setdefault(var.val, f"{var.val}_adv{rng.randint(*self.suffix_range)}")
            out[var.start_byte:var.end_byte] = new.encode("utf8")
        return out.decode("utf8")


@dataclass
class DeadCodeInsertion:
    name: str = "dead_code_insertion"

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        del variables  # unused
        decoy = f"_vrtg_decoy_{rng.randint(1000, 9999)}"
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") and stripped.endswith(":"):
                indent = " " * (len(line) - len(line.lstrip()) + 4)
                lines.insert(i + 1, f"{indent}{decoy} = {{'safe': True}}; {decoy}['_'] = len({decoy})")
                break
        return "\n".join(lines)


@dataclass
class ControlFlowFlattening:
    """Wrap each function body in `if True: ...`. CFG-level no-op."""
    name: str = "control_flow_flattening"

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        del variables, rng

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.body:
                    continue
                first = node.body[0]
                # already wrapped — skip to keep idempotent
                if (len(node.body) == 1 and isinstance(first, ast.If)
                        and isinstance(first.test, ast.Constant) and first.test.value is True):
                    continue
                node.body = [ast.If(test=ast.Constant(value=True), body=node.body, orelse=[])]
                changed = True

        if not changed:
            return code
        try:
            return ast.unparse(tree)
        except (AttributeError, ValueError):
            return code


@dataclass
class MutationStep:
    step: int
    strategy: str
    score_before: float
    score_after: float
    reward: float
    parse_valid: bool


@dataclass
class AttackTrace:
    steps: list[MutationStep] = field(default_factory=list)
    q_table: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    best_score: float = 1.0
    final_code: str = ""


ScoreFn = Callable[[str], float]


class RLAdversary:
    """Epsilon-greedy MAB over a set of mutators.

    Reward = score_before - score_after. A bad mutation (parse fail or
    score regression) gets -1.0 so the bandit learns to avoid it.
    """

    def __init__(
        self,
        target_model_score_fn: ScoreFn,
        mutators: list[Mutator] | None = None,
        epsilon: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if not callable(target_model_score_fn):
            raise TypeError("target_model_score_fn must be callable")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        self.score_fn = target_model_score_fn
        self.mutators = mutators or [
            VariableRenaming(),
            DeadCodeInsertion(),
            ControlFlowFlattening(),
        ]
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table = {m.name: 0.0 for m in self.mutators}
        self.counts = {m.name: 0 for m in self.mutators}

    @staticmethod
    def _ast_ok(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _select(self) -> Mutator:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.mutators)
        best = max(self.q_table, key=lambda n: self.q_table[n])
        for m in self.mutators:
            if m.name == best:
                return m
        return self.mutators[0]

    def attack(
        self,
        clean_code: str,
        variables: list[NodeWrapper],
        max_steps: int = 15,
    ) -> tuple[str, AttackTrace]:
        if not isinstance(clean_code, str):
            raise TypeError("clean_code must be str")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        try:
            base = float(self.score_fn(clean_code))
        except Exception as e:
            log.warning("score_fn failed on clean code: %s", e)
            return clean_code, AttackTrace(final_code=clean_code, best_score=1.0)

        best_code = code = clean_code
        best = base
        trace = AttackTrace(final_code=clean_code, best_score=base)

        for step in range(max_steps):
            m = self._select()
            try:
                cand = m(code, variables=variables, rng=self.rng)
            except Exception as e:
                log.warning("mutator %s raised: %s", m.name, e)
                trace.steps.append(MutationStep(step, m.name, base, base, -1.0, False))
                self._update_q(m.name, -1.0)
                continue

            ok = self._ast_ok(cand)
            if not ok:
                reward = -1.0
                new = base
            else:
                try:
                    new = float(self.score_fn(cand))
                except Exception as e:
                    log.warning("score_fn failed on mutation: %s", e)
                    new = base
                    reward = -1.0
                else:
                    reward = base - new
                    if new < best:
                        best = new
                        best_code = cand
                    if new < base:
                        # commit the improvement and keep mutating from here
                        code = cand
                        base = new

            trace.steps.append(MutationStep(step, m.name, base, new, reward, ok))
            self._update_q(m.name, reward)

        trace.q_table = dict(self.q_table)
        trace.counts = dict(self.counts)
        trace.best_score = best
        trace.final_code = best_code
        return best_code, trace

    def _update_q(self, name: str, reward: float) -> None:
        self.counts[name] += 1
        a = 1.0 / self.counts[name]
        self.q_table[name] += a * (reward - self.q_table[name])
