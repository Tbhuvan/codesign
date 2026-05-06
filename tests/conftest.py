import textwrap

import pytest


@pytest.fixture
def vuln_os_system() -> str:
    return textwrap.dedent("""
        import os

        def vulnerable_ping(host):
            sanitized = host.strip()
            cmd = "ping -c 1 " + sanitized
            os.system(cmd)
            return True
    """).strip()


@pytest.fixture
def vuln_sql_injection() -> str:
    return textwrap.dedent("""
        import sqlite3

        def lookup_user(username):
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            query = "SELECT * FROM users WHERE name = '" + username + "'"
            cursor.execute(query)
            return cursor.fetchall()
    """).strip()


@pytest.fixture
def shared_prefix_code() -> str:
    # Naive substring rename would corrupt `index` when renaming `i`.
    return textwrap.dedent("""
        def loop():
            i = 0
            index = 10
            valid = True
            while i < index and valid:
                i += 1
            return i
    """).strip()


class _MockSVD:
    # Heuristic scorer so tests don't need network or torch.
    def __init__(self, base: float = 0.95) -> None:
        self.base = base

    def __call__(self, code: str) -> float:
        s = self.base
        if "_adv" in code:
            s -= 0.4
        if "_vrtg_decoy" in code:
            s -= 0.3
        if "if True:" in code:
            s -= 0.1
        return max(0.01, s)


@pytest.fixture
def mock_svd() -> _MockSVD:
    return _MockSVD()
