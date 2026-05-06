import pytest

from codesign.parser import ProgramGraph, ProgramGraphExtractor


@pytest.fixture(scope="module")
def extractor() -> ProgramGraphExtractor:
    return ProgramGraphExtractor()


class TestParse:
    def test_rejects_non_str(self, extractor):
        with pytest.raises(TypeError):
            extractor.parse(123)

    def test_rejects_empty(self, extractor):
        with pytest.raises(ValueError):
            extractor.parse("   ")

    def test_unsupported_language(self):
        with pytest.raises(ValueError):
            ProgramGraphExtractor(language="rust")


class TestVariables:
    def test_extracts_user_variables(self, extractor, vuln_os_system):
        tree = extractor.parse(vuln_os_system)
        v = extractor.get_variables(tree, vuln_os_system.encode("utf-8"))
        assert "host" in v
        assert "sanitized" in v
        assert "cmd" in v

    def test_excludes_function_definition_names(self, extractor, vuln_os_system):
        tree = extractor.parse(vuln_os_system)
        v = extractor.get_variables(tree, vuln_os_system.encode("utf-8"))
        assert "vulnerable_ping" not in v

    def test_byte_ranges_unique_per_occurrence(self, extractor, shared_prefix_code):
        tree = extractor.parse(shared_prefix_code)
        v = extractor.get_variables(tree, shared_prefix_code.encode("utf-8"))
        spans = [(n.start_byte, n.end_byte) for n in v["i"]]
        assert len(spans) == len(set(spans))

    def test_distinguishes_overlapping_prefixes(self, extractor, shared_prefix_code):
        tree = extractor.parse(shared_prefix_code)
        v = extractor.get_variables(tree, shared_prefix_code.encode("utf-8"))
        assert "i" in v and "index" in v
        assert v["i"] is not v["index"]


class TestDFG:
    def test_assignment_targets_have_sources(self, extractor, vuln_os_system):
        dfg = extractor.extract_dfg(vuln_os_system)
        assert "cmd" in dfg
        assert "sanitized" in dfg["cmd"]

    def test_dfg_returns_dict_of_str_to_set(self, extractor, vuln_sql_injection):
        dfg = extractor.extract_dfg(vuln_sql_injection)
        for k, val in dfg.items():
            assert isinstance(k, str)
            assert isinstance(val, set)
            assert all(isinstance(s, str) for s in val)


class TestBuild:
    def test_returns_program_graph(self, extractor, vuln_os_system):
        g = extractor.build(vuln_os_system)
        assert isinstance(g, ProgramGraph)
        assert g.language == "python"
        assert g.source == vuln_os_system
        assert len(g.dfg) >= 1
        sinks = {s.val for s in g.sinks}
        assert "os.system" in sinks or "system" in sinks

    def test_byte_ranges_round_trip(self, extractor, vuln_os_system):
        g = extractor.build(vuln_os_system)
        b = vuln_os_system.encode("utf-8")
        for n in g.all_variable_nodes:
            assert b[n.start_byte:n.end_byte].decode("utf-8") == n.val
