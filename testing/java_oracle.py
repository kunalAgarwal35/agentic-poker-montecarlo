from __future__ import annotations
import os
import re
import shutil
import subprocess

_JAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "java_files")
_JAR = os.path.join(_JAR_DIR, "p2.jar")


def java_available() -> bool:
    return shutil.which("java") is not None and os.path.exists(_JAR)


def run_java_pql(query: str, max_trials: int = 600000, max_seconds: int = 60) -> dict:
    """
    Run a PQL query through p2.jar's RunPQL CLI and return {alias_lower: float}.

    Observed stdout format (from actual jar execution):
        P1 = 0.89535
        P2 = 0.10465
        100000 trials

    RunPQL prints the query's `as` aliases in UPPERCASE, followed by ` = ` and a
    decimal fraction in [0, 1].  There is no percent sign — values are already
    fractions.  The final "N trials" line is ignored by the regex.

    Parsing strategy:
      - Match lines of the form  WORD = FLOAT  (no percent sign).
      - Lower-case the key so callers can use the same lowercase aliases as our
        native engine (e.g. 'p1', 'p2').
      - Values are already in [0, 1]; no division needed.
    """
    cmd = [
        "java", "-XX:+TieredCompilation", "-XX:TieredStopAtLevel=1", "-Xshare:auto",
        "-cp", _JAR, "propokertools.cli.RunPQL",
        "-mt", str(max_trials), "-ms", str(max_seconds), query,
    ]
    out = subprocess.check_output(
        cmd, cwd=_JAR_DIR, text=True, timeout=max_seconds + 30
    )
    results: dict = {}
    for m in re.finditer(
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]*\.?[0-9]+)\s*(%?)\s*(?:\(\d+\))?\s*$",
        out,
        re.MULTILINE,
    ):
        name = m.group(1).lower()
        val = float(m.group(2))
        pct = m.group(3)
        if pct == "%":
            val /= 100.0
        results[name] = val
    return results
