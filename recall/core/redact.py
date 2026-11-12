"""Mask live credentials before session text reaches a model or the card store.

Generic format patterns only: no secret value is ever stored here.
"""
import re

_V = r"[A-Za-z0-9_\-.~+/]"  # value alphabet for key=value style secrets

# (kind, compiled pattern). A pattern with a group named "v" masks only that group,
# which keeps the surrounding key name readable; otherwise the whole match is masked.
PATTERNS = [
    ("private-key", re.compile(
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----(?:[A-Za-z0-9+/=\s]|\\n)*"
        r"(?:-----END (?:[A-Z]+ )?PRIVATE KEY-----)?")),
    ("aws-access-key", re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])")),
    ("aws-secret-key", re.compile(
        r"(?i:aws_?secret_?access_?key)[\"'\\]*\s*[:=]\s*[\"'\\]*(?P<v>[A-Za-z0-9/+]{40})(?![A-Za-z0-9/+])")),
    ("aws-signin-token", re.compile(r"SigninToken=(?P<v>[A-Za-z0-9_\-]{20,})")),
    ("github-token", re.compile(
        r"(?<![A-Za-z0-9])(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,})")),
    ("anthropic-key", re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai-key", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("gcp-api-key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_\-]{35}")),
    ("gcp-oauth-token", re.compile(r"(?<![A-Za-z0-9])ya29\.[0-9A-Za-z_\-]{30,}")),
    ("gcp-service-account-key", re.compile(r"\"private_key_id\\?\"\s*:\s*\\?\"(?P<v>[0-9a-f]{40})")),
    ("nebius-token", re.compile(r"(?<![A-Za-z0-9])(?:t1|v1)\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{30,}")),
    ("hf-token", re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}")),
    ("kaggle-key", re.compile(r"(?<![A-Za-z0-9])KGAT_[0-9a-f]{32}(?![0-9a-f])")),
    ("slack-token", re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}")),
    ("slack-webhook", re.compile(r"hooks\.slack\.com/services/(?P<v>T[0-9A-Z]{8,}[A-Za-z0-9/]*)")),
    ("gitlab-token", re.compile(r"(?<![A-Za-z0-9])glpat-[0-9A-Za-z_\-]{20,}")),
    ("stripe-key", re.compile(r"(?<![A-Za-z0-9])(?:sk|rk)_live_[0-9A-Za-z]{20,}")),
    ("telegram-bot-token", re.compile(r"(?<![0-9A-Za-z])[0-9]{8,16}:A[0-9A-Za-z_\-]{34}(?![0-9A-Za-z_\-])")),
    ("url-password", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/:@\s\"'<>]+:(?P<v>[^/@\s\"'<>]{4,})@")),
    ("canvas-token", re.compile(r"(?<![A-Za-z0-9~])\d{3,5}~[A-Za-z0-9]{40,80}(?![A-Za-z0-9])")),
    ("jwt", re.compile(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("bearer-token", re.compile(r"(?i:bearer)\s+(?P<v>[A-Za-z0-9\-._~+/]{24,}=*)")),
    ("api-key", re.compile(
        r"(?:\b[A-Za-z][A-Za-z0-9_]*(?:API_KEY|APIKEY|SECRET|SECRET_KEY|TOKEN|PASSWORD|PASSWD)"
        r"|(?i:\b(?:access|refresh|id|auth)_token|\bapi_?key|\bsecret_key|\bclient_secret|\bpassword))"
        r"[\"'\\]*\s*[:=]\s*[\"'\\]*(?P<v>" + _V + r"{16,})")),
    ("api-key", re.compile(r"[?&](?i:api[-_]?key|apikey|access_token|token)=(?P<v>[A-Za-z0-9_\-]{16,})")),
]


def _looks_secret(kind, value):
    """Key=value hits must look random (letters and digits, no placeholder or variable)."""
    if kind == "url-password":   # user:password@host; skip placeholders such as $PASS or <password>
        return not value.startswith(("$", "<", "{", "*", "[", "%")) and "REDACTED" not in value \
            and value.lower() not in ("password", "passwd", "pass", "secret", "token", "xxxx")
    if kind != "api-key":
        return True
    if value.startswith(("$", "phc_")) or "REDACTED" in value:  # phc_: public PostHog key
        return False
    return bool(re.search(r"\d", value)) and bool(re.search(r"[A-Za-z]", value))


def _sub(kind, rx, text, counts):
    def repl(m):
        grp = "v" if "v" in rx.groupindex else 0
        val = m.group(grp)
        if not _looks_secret(kind, val):
            return m.group(0)
        counts[kind] = counts.get(kind, 0) + 1
        if grp == 0:
            return f"[REDACTED:{kind}]"
        s, e = m.span("v")
        base = m.start(0)
        whole = m.group(0)
        return whole[:s - base] + f"[REDACTED:{kind}]" + whole[e - base:]
    return rx.sub(repl, text)


def find(text):
    """Return a list of (kind, value) secrets found in text."""
    hits = []
    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            val = m.group("v") if "v" in rx.groupindex else m.group(0)
            if _looks_secret(kind, val):
                hits.append((kind, val))
    return hits


def redact(text, counts=None):
    """Return text with every recognised credential replaced by [REDACTED:<kind>]."""
    if not isinstance(text, str) or not text:
        return text
    counts = {} if counts is None else counts
    for kind, rx in PATTERNS:
        text = _sub(kind, rx, text, counts)
    return text
