import pytest

from codesign.targets import (
    HeuristicClassifier,
    HFPipelineClassifier,
    OllamaPromptClassifier,
    make_target,
)


class TestMakeTarget:
    def test_heuristic(self):
        t = make_target("heuristic")
        assert isinstance(t, HeuristicClassifier)
        assert 0.0 <= t.score("def f(): pass") <= 1.0

    def test_ollama_spec(self):
        t = make_target("ollama:qwen3:8b")
        assert isinstance(t, OllamaPromptClassifier)
        assert t.model == "qwen3:8b"

    def test_hf_spec(self):
        t = make_target("hf:some-org/some-model")
        assert isinstance(t, HFPipelineClassifier)
        assert t.model_id == "some-org/some-model"

    def test_unknown_spec(self):
        with pytest.raises(ValueError):
            make_target("not-a-real-backend:thing")

    def test_empty_spec(self):
        with pytest.raises(ValueError):
            make_target("")


class TestHeuristic:
    def test_score_drops_with_adv(self):
        t = HeuristicClassifier()
        clean = t.score("def f(x): return x")
        adv = t.score("def f_adv1(x_adv2): return x_adv2")
        assert adv < clean

    def test_score_drops_with_decoy(self):
        t = HeuristicClassifier()
        clean = t.score("def f(): pass")
        adv = t.score("def f():\n    _vrtg_decoy_1234 = {}\n    pass")
        assert adv < clean

    def test_score_in_range(self):
        t = HeuristicClassifier()
        for code in ["", "x", "def f(): pass", "if True: pass"]:
            s = t.score(code)
            assert 0.0 <= s <= 1.0

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            HeuristicClassifier(base=2.0)

    def test_rejects_non_str(self):
        with pytest.raises(TypeError):
            HeuristicClassifier().score(42)


class TestOllamaClassifierPromptBuilder:
    def test_zero_shot_prompt(self):
        t = OllamaPromptClassifier(model="qwen3:8b")
        p = t._build_prompt("def f(): pass")
        assert "VULN" in p
        assert "SAFE" in p
        assert "def f(): pass" in p
        assert p.rstrip().endswith("Label:")

    def test_three_shot_prompt(self):
        shots = [
            ("import os; os.system(x)", "VULN"),
            ("def add(a,b): return a+b", "SAFE"),
        ]
        t = OllamaPromptClassifier(model="qwen3:8b", n_shots=2, shots=shots)
        p = t._build_prompt("def g(): pass")
        # both demonstrations + the query should appear
        assert "os.system(x)" in p
        assert "def add(a,b): return a+b" in p
        assert "def g(): pass" in p
        assert p.count("Label:") == 3  # two shots + the query trailing label

    def test_info(self):
        t = OllamaPromptClassifier(model="qwen3:8b", n_shots=2)
        info = t.info()
        assert info["kind"] == "ollama"
        assert info["model"] == "qwen3:8b"
        assert info["n_shots"] == 2

    def test_rejects_empty_model(self):
        with pytest.raises(ValueError):
            OllamaPromptClassifier(model="")


class TestHFPipelineClassifierConfig:
    def test_defaults(self):
        t = HFPipelineClassifier(model_id="x/y")
        assert t.model_id == "x/y"
        assert t.invert is False
        assert t.positive_label is None
        assert t.spec == "hf:x/y"

    def test_invert(self):
        t = HFPipelineClassifier(model_id="x/y", invert=True)
        assert t.invert is True

    def test_rejects_empty_model_id(self):
        with pytest.raises(ValueError):
            HFPipelineClassifier(model_id="")
