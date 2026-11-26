import nox

nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[test]")
    session.run("pytest", "tests/", "-x", "-q", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Run ruff linter and formatter check."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run pyright type checking."""
    session.install(".[dev,demo]")
    session.run("pyright", "fsd_path_planning")


@nox.session(python="3.12")
def differential(session: nox.Session) -> None:
    """Compare the working tree with the frozen baseline, golden-v1.

    Needs the tag and its history: in CI, check out with fetch-depth 0. Pass
    runner options after --, for example: nox -s differential -- --with-cache
    """
    session.install(".")
    session.run("python", "tools/diff_runner/diff_runner.py", *session.posargs)
