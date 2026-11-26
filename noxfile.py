import nox

nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[test]")
    session.run("pytest", "tests/", "-x", "-q", *session.posargs)


@nox.session(python=PYTHON_VERSIONS[0], name="tests-lowest")
def tests_lowest(session: nox.Session) -> None:
    """Run the test suite against the oldest dependency versions allowed.

    The runtime dependencies in pyproject.toml carry floors and no caps; this
    keeps the floors true. Only the package's own dependencies are resolved to
    their lowest versions: pytest is installed afterwards at its latest.
    scikit-learn is left out, so its parity test is skipped.
    """
    # Wheels only for the compiled dependencies: their oldest releases do not
    # build from source on a current toolchain.
    session.install(
        "--resolution=lowest-direct",
        "--only-binary=numba,llvmlite,numpy,scipy",
        ".",
    )
    session.install("pytest")
    session.run(
        "python",
        "-c",
        "import numba, numpy, scipy; "
        "print('numba', numba.__version__, 'numpy', numpy.__version__, "
        "'scipy', scipy.__version__)",
    )
    session.run("pytest", "tests/", "-x", "-q", *session.posargs)


@nox.session(python="3.12")
def demo(session: nox.Session) -> None:
    """Run the demos headless: the Streamlit app and the command line demo."""
    session.install(".[demo,test]")
    session.run("pytest", "tests/test_demo_smoke.py", "-m", "demo", "-q")


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
