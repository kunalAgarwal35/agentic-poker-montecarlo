"""The Dockerfile must COPY every root module `import server` actually needs.

Why this test exists: the deployed image COPYs a hand-listed, deliberately
minimal file set rather than the whole repo. That is a real memory/size win,
but it means the image's dependency list is maintained by hand while the
imports it must satisfy change with the code -- and nothing connected the two.

The failure that followed is the one this test prevents. `server.py` grew
`from range_ladder import ...` at module scope; the COPY list was not updated;
every local test passed because the repo has all the files. The image built
clean, then the container died on import at startup and returned 502 for
EVERY endpoint -- including /pql, which serves live traffic and has no
relationship to the range ladder. A missing COPY is not a degraded feature,
it is a total outage, and it is invisible until deploy.

So: parse the Dockerfile's COPY lines, compute the transitive closure of
root-level module imports starting from `server`, and assert the first covers
the second. This is a static check -- no Docker daemon, no build, runs in
milliseconds -- so it can sit in the normal suite.
"""

import ast
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root_modules():
    """Names of every .py at the repo root -- the set an import can resolve to
    only because the file sits next to server.py in the image's WORKDIR."""
    return {
        f[:-3]
        for f in os.listdir(REPO_ROOT)
        if f.endswith(".py") and not f.startswith("_")
    }


def _local_packages():
    """Root-level package DIRECTORIES (pql/). These are COPYed wholesale, so
    they are local code, not something pip must install -- without this the
    dependency check reports `pql` as a missing third-party package."""
    return {
        d
        for d in os.listdir(REPO_ROOT)
        if os.path.isdir(os.path.join(REPO_ROOT, d))
        and os.path.exists(os.path.join(REPO_ROOT, d, "__init__.py"))
    }


def _import_closure(entry="server"):
    """Every root module reachable from `entry` by following imports.

    Only root-level modules are tracked; packages (pql/, numpy, ...) are
    COPYed or pip-installed wholesale and are not what this guards.
    """
    local = _root_modules()
    seen, stack = set(), [entry]
    while stack:
        mod = stack.pop()
        if mod in seen or mod not in local:
            continue
        seen.add(mod)
        path = os.path.join(REPO_ROOT, mod + ".py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                stack.extend(a.name.split(".")[0] for a in node.names)
            # level > 0 is a relative import, which a root module can't have.
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                stack.append(node.module.split(".")[0])
    return seen


def _copied_py_files():
    """Root .py filenames named by COPY lines in the Dockerfile.

    Handles backslash line-continuations and multi-file COPY forms; ignores the
    trailing destination argument and any COPY of a directory (pql/).
    """
    with open(os.path.join(REPO_ROOT, "Dockerfile"), encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r"\\\s*\n", " ", text)  # join continuations
    copied = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.upper().startswith("COPY "):
            continue
        # Drop the COPY keyword and the destination (last token).
        parts = line.split()[1:-1]
        copied.update(p[:-3] for p in parts if p.endswith(".py"))
    return copied


def test_dockerfile_copies_every_module_server_imports():
    needed = _import_closure("server")
    copied = _copied_py_files()
    missing = sorted(needed - copied)
    assert not missing, (
        "Dockerfile does not COPY these root modules that `import server` "
        f"needs: {missing}. The image will build fine and then crash on "
        "import at container start, 502ing every endpoint. Add them to a "
        "COPY line."
    )


def test_closure_is_not_trivially_empty():
    """Guards the guard: if the parser silently returned nothing, the test
    above would pass no matter what the Dockerfile said."""
    needed = _import_closure("server")
    assert "server" in needed
    assert "range_ladder" in needed, (
        "range_ladder dropped out of server's import closure -- either the "
        "endpoint was removed or this test's parser is broken"
    )
    assert len(needed) > 5


def _third_party_imports(closure):
    """Non-stdlib, non-root packages the closure imports -- i.e. the things
    that must come from requirements-engine.txt."""
    stdlib = set(sys.stdlib_module_names)
    local = _root_modules()
    external = set()
    for mod in closure:
        with open(os.path.join(REPO_ROOT, mod + ".py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                external.add(node.module.split(".")[0])
    return external - stdlib - local - _local_packages()


def _declared_requirements():
    path = os.path.join(REPO_ROOT, "requirements-engine.txt")
    names = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(re.split(r"[=<>!\[;]", line)[0].strip().lower())
    return names


# Import name -> distribution name, where they differ.
_IMPORT_TO_DIST = {"cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn"}


def test_engine_requirements_cover_every_third_party_import():
    """The COPY list is only half the image contract; the other half is pip.

    The same outage had a second layer behind it: even with every module
    COPYed, the container still could not import, because the modules pulled
    in `pandas` and `cv2` and the engine installs neither. Catching that
    needed a dep check, not a file check -- so here it is. The fix was to stop
    importing them at all (see process_pool.py), which is why this passes with
    a deliberately short requirements file rather than a padded one.
    """
    closure = _import_closure("server")
    external = _third_party_imports(closure)
    declared = _declared_requirements()
    missing = sorted(
        name for name in external
        if _IMPORT_TO_DIST.get(name, name).lower() not in declared
    )
    assert not missing, (
        f"`import server` reaches these third-party packages that "
        f"requirements-engine.txt does not install: {missing}. The container "
        "will fail to import at startup. Either add them to the requirements "
        "or (usually better) break the import chain that reaches them."
    )


def test_heavy_deps_stay_out_of_the_engine_closure():
    """Named explicitly because these two are what actually broke it, and
    because 'it imports fine locally' will never catch them again."""
    external = _third_party_imports(_import_closure("server"))
    for heavy in ("pandas", "cv2"):
        assert heavy not in external, (
            f"{heavy} is back in the engine's import closure. It is not in the "
            "image and computes nothing on the /pql or /range_ladder path -- "
            "something started importing multithread_ploequities3 (or another "
            "module that reaches it) instead of process_pool."
        )


def test_copy_parser_finds_the_known_lines():
    """Guards the guard, other direction: a parser that returned nothing would
    make the completeness test fail loudly rather than pass silently, but one
    that over-matched would hide real omissions."""
    copied = _copied_py_files()
    assert "server" in copied
    assert "card_encoding" in copied
    assert "requirements-engine" not in copied  # .txt, not .py
