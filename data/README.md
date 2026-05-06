# CWE sample corpus

Hand-curated Python vulnerability samples. Each `.py` has a sibling
`.json` describing it.

| File | CWE | Class | Vulnerable |
|---|---|---|---|
| `cwe_078_os_command_injection.py` | CWE-78 | OS command injection | y |
| `cwe_089_sql_injection.py` | CWE-89 | SQL injection | y |
| `cwe_022_path_traversal.py` | CWE-22 | Path traversal | y |
| `cwe_502_unsafe_deserialization.py` | CWE-502 | Unsafe deserialization | y |
| `cwe_094_eval_injection.py` | CWE-94 | Code injection | y |
| `cwe_798_hardcoded_credentials.py` | CWE-798 | Hardcoded credentials | y |
| `cwe_259_hardcoded_password_db.py` | CWE-259 | Hardcoded DB password | y |
| `cwe_327_weak_crypto.py` | CWE-327 | Weak cryptography | y |
| `cwe_611_xxe.py` | CWE-611 | XXE | y |
| `cwe_918_ssrf.py` | CWE-918 | SSRF | y |
| `cwe_330_insecure_random.py` | CWE-330 | Insecure random | y |
| `safe_parameterized_query.py` | NONE | Negative control | n |

The set is deliberately small — 12 samples cover enough CWE variance
to discriminate signal from noise on a CPU laptop in under a minute.
For full-scale evaluation, point `--dataset` at a Devign / BigVul /
DiverseVul export following the same `.py` + `.json` convention.

## Schema

```json
{
  "id": "lowercase-kebab-id",
  "cwe": "CWE-78",
  "vulnerable": true,
  "description": "One-sentence description.",
  "tags": ["taxonomy", "labels"]
}
```

`id` is required and must be unique.
