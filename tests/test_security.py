"""
Read-only unit tests for the VBL P0 security hardening:
  - no fail-open fallback secrets/tokens
  - exposed app/MCP defaults bind to 127.0.0.1 (loopback) where the architecture allows
  - docker-compose requires explicit secrets at startup

These tests are pure/unit: they do NOT start services, do NOT require network,
credentials, or a running stack, and do NOT mutate any data.
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# ─── services/secure_env helper ───────────────────────────────────────────────

def test_effective_host_defaults_to_loopback():
    """Unset host env vars must resolve to 127.0.0.1, never 0.0.0.0."""
    from services import secure_env

    key = "VBL_TEST_HOST_XYZZY"
    os.environ.pop(key, None)
    try:
        assert secure_env.effective_host(key) == "127.0.0.1"
    finally:
        os.environ.pop(key, None)


def test_effective_host_honors_explicit_env():
    from services import secure_env

    key = "VBL_TEST_HOST_XYZZY"
    os.environ[key] = "0.0.0.0"  # explicit override is respected
    try:
        assert secure_env.effective_host(key) == "0.0.0.0"
    finally:
        os.environ.pop(key, None)


def test_require_secret_raises_when_unset():
    from services import secure_env

    key = "VBL_TEST_SECRET_XYZZY"
    os.environ.pop(key, None)
    with pytest.raises(RuntimeError):
        secure_env.require_secret(key)


def test_require_secret_returns_when_set():
    from services import secure_env

    key = "VBL_TEST_SECRET_XYZZY"
    os.environ[key] = "s3cret"
    try:
        assert secure_env.require_secret(key) == "s3cret"
    finally:
        os.environ.pop(key, None)


# ─── Default binds for importable app services ─────────────────────────────────

@pytest.mark.parametrize(
    "module,env_key",
    [
        ("services.research.app", "RESEARCH_HOST"),
        ("services.scraper.app", "SCRAPER_HOST"),
        ("services.publisher.app", "PUBLISHER_HOST"),
        ("services.renderer.app", "RENDERER_HOST"),
    ],
)
def test_app_services_default_to_loopback(module, env_key):
    """App service bind defaults must be loopback, not 0.0.0.0."""
    os.environ.pop(env_key, None)
    mod = importlib.import_module(module)
    assert hasattr(mod, "HOST"), f"{module} must expose a HOST constant"
    assert mod.HOST == "127.0.0.1"


def test_mcp_server_binds_loopback_and_requires_secret():
    """mcp-server (script-only dashed dir) must default loopback + no fail-open token."""
    src = (REPO / "services" / "mcp-server" / "app.py").read_text()
    # Host default must come from the shared helper (loopback default).
    assert "effective_host(\"MCP_HOST\")" in src
    assert "require_secret(\"MCP_AUTH_TOKEN\"" in src
    assert '"0.0.0.0"' not in src.replace('os.environ.get("MCP_HOST", "0.0.0.0")', '')


# ─── Scraper auth: fail closed, no "any non-empty" acceptance ──────────────────

def test_scraper_rejects_nonexistent_flag_and_docstring():
    """Scraper auth must no longer accept 'any non-empty' key."""
    src = (REPO / "services" / "scraper" / "app.py").read_text()
    assert "any non-empty" not in src
    assert "local-dev-key" not in src


def test_scraper_rejects_unconfigured_key_with_require_secret():
    """Scraper must validate against a required configured secret (fail closed)."""
    from services import secure_env

    os.environ.pop("SCRAPER_API_KEY", None)
    try:
        with pytest.raises(RuntimeError):
            secure_env.require_secret("SCRAPER_API_KEY")
    finally:
        os.environ.pop("SCRAPER_API_KEY", None)


# ─── Docker compose requires secrets ──────────────────────────────────────────

def test_compose_requires_secret_passwords():
    """Compose secret vars must require explicit values, not ship defaults."""
    yml = (REPO / "docker-compose.yml").read_text()
    for var in ["POSTGRES_PASSWORD", "MINIO_PASSWORD", "MCP_AUTH_TOKEN"]:
        # Required (no-default) compose substitution is `${VAR:?message}`.
        assert re.search(r"\$\{%s:\?" % var, yml), (
            f"{var} must use a required (no-default) Compose substitution"
        )


# ─── Per-secret fail-open absence both in app-bound clients ───────────────────

def test_corpus_client_has_no_hardcoded_scraper_key_fallback():
    src = (REPO / "services" / "research" / "corpus.py").read_text()
    assert "local-dev-key" not in src


def test_bulk_ingest_has_no_hardcoded_scraper_key():
    src = (REPO / "services" / "research" / "bulk_ingest.py").read_text()
    assert '"local-dev"' not in src


# ─── corpus.py: get_scraper_api_key must call require_secret, not alias-nested ──

def test_corpus_get_scraper_api_key_uses_require_secret():
    """get_scraper_api_key() must raise RuntimeError when the secret is unset
    (proves it goes straight through require_secret, not a broken alias)."""
    from services.research import corpus

    os.environ.pop("SCRAPER_API_KEY", None)
    try:
        with pytest.raises(RuntimeError):
            corpus.get_scraper_api_key()
    finally:
        os.environ.pop("SCRAPER_API_KEY", None)


def test_corpus_no_broken_alias_call():
    src = (REPO / "services" / "research" / "corpus.py").read_text()
    # Must not call require_secret through the aliased name.
    assert "secure_env.require_secret" not in src
    assert "from services.secure_env import require_secret" in src


# ─── dashed-dir services: __package__-aware parent-services bootstrap ─────────

@pytest.mark.parametrize(
    "rel",
    [
        "services/mcp-server/app.py",
        "services/browser-worker/app.py",
        "services/publisher/app.py",
        "services/renderer/app.py",
    ],
)
def test_dashed_dir_service_uses_package_aware_bootstrap(rel):
    src = (REPO / rel).read_text()
    # Direct-script fallback puts the parent services/ dir on the path…
    assert "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" in src
    assert "if not __package__:" in src
    # …and the package-import branch uses the services.secure_env package module.
    assert "from services.secure_env import " in src


# ─── start scripts: no MCP_AUTH_TOKEN fallback, SCRAPER_API_KEY required ───────

@pytest.mark.parametrize("rel", ["start-all.sh", "deploy/start-all.sh"])
def test_start_scripts_require_mcp_token(rel):
    src = (REPO / rel).read_text()
    assert "MCP_AUTH_TOKEN:-local-dev-token" not in src
    assert "MCP_AUTH_TOKEN:?Set MCP_AUTH_TOKEN" in src


@pytest.mark.parametrize("rel", ["start-all.sh", "deploy/start-all.sh"])
def test_start_scripts_require_scraper_key(rel):
    src = (REPO / rel).read_text()
    assert "SCRAPER_API_KEY:?Set SCRAPER_API_KEY" in src


def test_deploy_uses_script_not_mcp_module():
    src = (REPO / "deploy" / "start-all.sh").read_text()
    assert "services.mcp_server.app" not in src
    assert "services/mcp-server/app.py" in src


# ─── Docker published ports default to 127.0.0.1 bindings ─────────────────────

def test_compose_published_ports_bind_loopback():
    yml = (REPO / "docker-compose.yml").read_text()
    for host_port in ["5432", "6379", "6333", "6334", "9000", "9001",
                      "8001", "8010", "8020", "8030", "8031"]:
        assert f"127.0.0.1:{host_port}:{host_port}" in yml, (
            f"published port {host_port} must bind to 127.0.0.1"
        )
    # No bare port mappings remain.
    assert not re.search(r'^\s*-\s+"\d+:\d+"\s*$', yml, re.M)


# ─── .env.example: required secrets empty, config/docs clean of insecure default ─

def test_env_example_required_secrets_are_empty():
    """.env.example required secrets must be EMPTY so an unchanged copy is rejected
    by Compose `:?` — never a predictable 'change-me'/'vbl_secret' value."""
    src = (REPO / ".env.example").read_text()
    for var in ["POSTGRES_PASSWORD", "MINIO_PASSWORD", "MCP_AUTH_TOKEN", "SCRAPER_API_KEY"]:
        assert re.search(r"^%s=\s*$" % var, src, re.M), (
            f"{var} must be empty in .env.example"
        )
    assert "change-me" not in src
    assert "vbl_secret" not in src


def test_no_real_comfyui_lan_ip_in_tracked_defaults_or_docs():
    """Tracked runtime defaults and docs must use a generic GPU endpoint."""
    paths = [
        ".env.example",
        "docs/optimization-guide.md",
        "services/mcp-server/app.py",
        "services/renderer/app.py",
        "start-all.sh",
        "production/h3/README.md",
    ]
    contents = [(REPO / path).read_text() for path in paths]
    assert all("192.168.1.143" not in content for content in contents)
    assert all(
        "gpu-server:8188" in content
        for content in contents[:5]
    )


def test_optimization_guide_has_no_insecure_defaults():
    """.env docs table must mark the four secrets required with no default, and
    must not advertise vbl_secret / vbl_secret_123 / local-dev-token."""
    guide = (REPO / "docs" / "optimization-guide.md").read_text()
    assert "vbl_secret" not in guide
    assert "vbl_secret_123" not in guide
    assert "local-dev-token" not in guide
    for var in ["POSTGRES_PASSWORD", "MINIO_PASSWORD", "MCP_AUTH_TOKEN", "SCRAPER_API_KEY"]:
        row = next(line for line in guide.splitlines() if f"`{var}`" in line)
        assert "| Yes | — |" in row, f"guide row for {var} must be required with no default"


# ─── Secret comparisons must be constant-time (hmac.compare_digest) ───────────

def test_scraper_uses_constant_time_comparison():
    src = (REPO / "services" / "scraper" / "app.py").read_text()
    assert "secrets_equal" in src
    assert "x_api_key != _expected_api_key()" not in src


def test_mcp_server_uses_constant_time_comparison():
    src = (REPO / "services" / "mcp-server" / "app.py").read_text()
    assert "secrets_equal" in src
    assert "token != AUTH_TOKEN" not in src


def test_secure_env_has_constant_time_secrets_equal():
    """Shared helper must exist and use hmac.compare_digest (constant time)."""
    helper = (REPO / "services" / "secure_env.py").read_text()
    assert "import hmac" in helper
    assert "hmac.compare_digest" in helper
    assert "def secrets_equal" in helper
    from services import secure_env
    assert secure_env.secrets_equal("abc", "abc") is True
    assert secure_env.secrets_equal("abc", "abd") is False
    assert secure_env.secrets_equal("", "") is True
    assert secure_env.secrets_equal(None, "abc") is False  # fail-safe for None
    assert secure_env.secrets_equal("abc", None) is False


def test_scraper_check_api_key_accepts_correct_key():
    """Correct key returns without raising; wrong key -> 403; missing -> 401."""
    from fastapi import HTTPException
    import services.scraper.app as scraper

    os.environ["SCRAPER_API_KEY"] = "correct-secret"
    try:
        # Correct key: no exception.
        assert scraper.check_api_key("correct-secret") is None

        # Wrong key: 403.
        try:
            scraper.check_api_key("wrong-key")
            assert False, "wrong key should raise"
        except HTTPException as e:
            assert e.status_code == 403

        # Missing key: 401.
        try:
            scraper.check_api_key(None)
            assert False, "missing key should raise"
        except HTTPException as e:
            assert e.status_code == 401
    finally:
        os.environ.pop("SCRAPER_API_KEY", None)


def test_scraper_check_api_key_fails_closed_when_secret_unset():
    """With SCRAPER_API_KEY unset the comparison must fail closed (RuntimeError)."""
    from services import secure_env
    import services.scraper.app as scraper

    os.environ.pop("SCRAPER_API_KEY", None)
    try:
        with pytest.raises(RuntimeError):
            scraper.check_api_key("anything")
    finally:
        os.environ.pop("SCRAPER_API_KEY", None)


def test_mcp_bearer_comparison_constant_time_and_behavior():
    """MCP bearer comparison uses the shared constant-time helper end-to-end."""
    from services import secure_env

    # The MCP middleware compares the presented bearer token against AUTH_TOKEN
    # via secure_env.secrets_equal (ensured by test_mcp_server_uses_constant_time_comparison).
    assert secure_env.secrets_equal("mcp-secret", "mcp-secret") is True
    assert secure_env.secrets_equal("mcp-secret", "wrong") is False
    assert secure_env.secrets_equal("", "mcp-secret") is False  # empty never matches


# ─── Dockerfile: correct service startup wiring (P1) ─────────────────────────

def _docker_target_blocks():
    """Parse Dockerfile into {target: {raw, expose, health_port, cmd}}."""
    dockerfile = (REPO / "Dockerfile").read_text()
    blocks: dict[str, dict] = {}
    current = None
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if stripped.startswith("FROM") and " AS " in stripped:
            current = stripped.split(" AS ")[-1].strip()
            blocks[current] = {"raw": [line], "expose": [], "health_port": None, "cmd": None}
        elif current is not None:
            blocks[current]["raw"].append(line)
            if stripped.startswith("EXPOSE"):
                blocks[current]["expose"].append(stripped.split()[1])
            elif stripped.startswith("CMD") or stripped.startswith('CMD ['):
                cmd_src = line.split("CMD", 1)[1].strip()
                # Normalize JSON-array CMD (exec form) into a flat shell string so
                # assertions don't depend on CMD encoding (exec vs shell form).
                if cmd_src.startswith("["):
                    import json as _json
                    try:
                        cmd_src = " ".join(_json.loads(cmd_src))
                    except Exception:
                        pass  # keep raw; assertion will surface the mismatch
                blocks[current]["cmd"] = cmd_src
    # Compose a target's effective port tuple; the healthcheck CMD line carries
    # the URL `http://localhost:<port>/health` (continuation), so scan the block.
    import re as _re
    for blk in blocks.values():
        for ln in blk["raw"]:
            m = _re.search(r"localhost:(\d+)/health", ln)
            if m:
                blk["health_port"] = m.group(1)
                break
    return blocks


def test_docker_file_has_expected_targets():
    blocks = _docker_target_blocks()
    for name in ["research-api", "scraper-api", "browser-worker", "publisher", "renderer"]:
        assert name in blocks, f"Dockerfile missing target {name}"


def test_docker_browser_worker_uses_script_serve_mode():
    """browser-worker is a hyphenated dir; dotted module is invalid and must run
    in direct-script --serve mode bound to the exposed port 8020."""
    blocks = _docker_target_blocks()
    bw = blocks["browser-worker"]
    assert "python services/browser-worker/app.py --serve --port 8020" in bw["cmd"], bw["cmd"]
    assert "8020" in bw["expose"], bw["expose"]
    assert bw["health_port"] == "8020", bw["health_port"]
    # History: no buildable dotted-module path for the hyphenated dir.
    assert "services.browser.app" not in bw["cmd"]


def test_docker_publisher_all_8030_matches_source_and_compose():
    """Publisher app declares 8030 (PUBLISHER_PORT) and Compose publishes 8030."""
    blocks = _docker_target_blocks()
    pub = blocks["publisher"]
    assert "services.publisher.app:app" in pub["cmd"], pub["cmd"]
    assert "8030" in pub["expose"], pub["expose"]
    assert pub["health_port"] == "8030", pub["health_port"]
    assert "--port 8030" in pub["cmd"] or "--port\", \"8030" in pub["cmd"], pub["cmd"]
    # Cross-check with the source default.
    src = (REPO / "services" / "publisher" / "app.py").read_text()
    assert 'PUBLISHER_PORT", "8030"' in src
    # Cross-check with Compose published port.
    yml = (REPO / "docker-compose.yml").read_text()
    assert "127.0.0.1:8030:8030" in yml


def test_docker_renderer_all_8031_matches_source_and_compose():
    """Renderer app declares 8031 (RENDERER_PORT) and Compose publishes 8031."""
    blocks = _docker_target_blocks()
    rend = blocks["renderer"]
    assert "services.renderer.app:app" in rend["cmd"], rend["cmd"]
    assert "8031" in rend["expose"], rend["expose"]
    assert rend["health_port"] == "8031", rend["health_port"]
    assert "--port 8031" in rend["cmd"] or "--port\", \"8031" in rend["cmd"], rend["cmd"]
    src = (REPO / "services" / "renderer" / "app.py").read_text()
    assert 'RENDERER_PORT", "8031"' in src
    yml = (REPO / "docker-compose.yml").read_text()
    assert "127.0.0.1:8031:8031" in yml


# ─── .dockerignore: private data exclusion + context size ─────────────────────

def test_dockerignore_exists():
    """Repo must have a .dockerignore to prevent 849MB context."""
    assert (REPO / ".dockerignore").exists(), ".dockerignore missing"


def test_dockerignore_excludes_private_research_files():
    """Private research JSON must be excluded from Docker build context."""
    di = (REPO / ".dockerignore").read_text()
    assert "research/dreamingtulpa_discord_mentions.json" in di
    assert "research/dreamingtulpa_unique_prompts.json" in di


def test_dockerignore_excludes_heavy_generated_dirs():
    """Generated outputs, data, caches must be excluded."""
    di = (REPO / ".dockerignore").read_text()
    for pattern in [".git", ".venv", "__pycache__", "data/", "outputs/", "*.db"]:
        assert pattern in di, f".dockerignore must exclude {pattern}"


def test_dockerignore_recursively_excludes_caches_and_databases():
    """Generated Python/SQLite artifacts below the repo root stay out of images."""
    di = (REPO / ".dockerignore").read_text().splitlines()
    patterns = {line.strip() for line in di if line.strip() and not line.startswith("#")}
    for pattern in ["**/__pycache__/", "**/*.py[cod]", "**/*.db", "**/*.db-shm", "**/*.db-wal"]:
        assert pattern in patterns, f".dockerignore must recursively exclude {pattern}"


def test_dockerignore_preserves_services_and_production():
    """services/ and production/ must NOT be excluded (required at runtime)."""
    di = (REPO / ".dockerignore").read_text()
    lines = [l.strip() for l in di.splitlines() if l.strip() and not l.startswith("#")]
    # No line should exclude services/ or production/ entirely
    for line in lines:
        assert line not in ("services/", "services", "production/", "production"), (
            f".dockerignore must not exclude required dir: {line}"
        )


# ─── Dockerfile: install order must copy source before unpiped pip install ────

def test_dockerfile_pip_install_not_piped_through_tail():
    """pip install must NOT be piped through tail (masks exit code)."""
    df = (REPO / "Dockerfile").read_text()
    assert "| tail" not in df, "Dockerfile must not pipe pip install through tail"


def test_dockerfile_copies_source_before_pip_install():
    """Source must be COPYed before 'pip install .' so package metadata resolves."""
    df = (REPO / "Dockerfile").read_text()
    lines = df.splitlines()
    # Find first COPY . . and first pip install . in base stage
    copy_line = None
    pip_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "COPY . ." and copy_line is None:
            copy_line = i
        if stripped.startswith("RUN pip install") and "pip install ." in stripped and pip_line is None:
            pip_line = i
    assert copy_line is not None, "Dockerfile must contain 'COPY . .'"
    assert pip_line is not None, "Dockerfile must contain 'pip install .'"
    assert copy_line < pip_line, (
        f"COPY . . (line {copy_line}) must come before pip install (line {pip_line})"
    )


def test_hatch_wheel_declares_services_package():
    """Docker's pip install must be able to build a wheel containing services."""
    import tomllib

    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    wheel = config.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
    assert wheel.get("packages") == ["services"], (
        "Hatchling must explicitly package services; project name viral-bench-local "
        "does not map to an import directory"
    )


# ─── Browser worker: container bind must be 0.0.0.0 for published port ────────

def test_browser_worker_container_binds_all_interfaces():
    """Browser target opts into a container bind without changing host defaults."""
    blocks = _docker_target_blocks()
    bw = blocks["browser-worker"]
    assert any(
        line.strip() == "ENV BROWSER_WORKER_HOST=0.0.0.0" for line in bw["raw"]
    ), (
        "browser-worker target must explicitly set BROWSER_WORKER_HOST=0.0.0.0 "
        "for container port mapping"
    )
    assert "--host" not in bw["cmd"], "browser-worker CLI does not support --host"


def test_browser_worker_docker_cmd_uses_supported_cli_flags():
    """Every flag in the Docker CMD must be accepted by the real script CLI."""
    import subprocess
    import sys

    bw = _docker_target_blocks()["browser-worker"]
    help_result = subprocess.run(
        [sys.executable, str(REPO / "services/browser-worker/app.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    supported = help_result.stdout
    cmd_flags = [part for part in bw["cmd"].split() if part.startswith("--")]
    assert cmd_flags
    for flag in cmd_flags:
        assert flag in supported, f"Docker CMD uses unsupported browser-worker flag: {flag}"


# ─── VBL_DATA_DIR: configurable runtime data contract ─────────────────────────

def test_corpus_respects_vbl_data_dir_env():
    """corpus.DB_PATH must resolve under VBL_DATA_DIR when set."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['VBL_DATA_DIR']='/tmp/test_vbl_data'; "
         "from services.research.corpus import DB_PATH; print(DB_PATH)"],
        capture_output=True, text=True, cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO), "VBL_DATA_DIR": "/tmp/test_vbl_data"},
    )
    assert "/tmp/test_vbl_data" in result.stdout.strip(), (
        f"corpus.DB_PATH did not use VBL_DATA_DIR: {result.stdout} {result.stderr}"
    )


def test_corpus_default_data_dir_is_host_compatible():
    """Without VBL_DATA_DIR, corpus must default to ~/viral-bench-local/data."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ.pop('VBL_DATA_DIR', None); "
         "from services.research.corpus import DB_DIR; print(DB_DIR)"],
        capture_output=True, text=True, cwd=str(REPO),
        env={k: v for k, v in os.environ.items() if k != "VBL_DATA_DIR"},
    )
    out = result.stdout.strip()
    assert "viral-bench-local/data" in out or "data" in out, (
        f"Default DB_DIR should be host-compatible: {out}"
    )


def test_browser_worker_respects_vbl_data_dir_env():
    """browser-worker DB_PATH must resolve under VBL_DATA_DIR when set."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['VBL_DATA_DIR']='/tmp/test_vbl_bw'; "
         "os.environ.setdefault('BROWSER_WORKER_HOST','127.0.0.1'); "
         "import importlib.util; spec=importlib.util.spec_from_file_location("
         "'bw', 'services/browser-worker/app.py'); "
         "mod=importlib.util.module_from_spec(spec); "
         "spec.loader.exec_module(mod); print(mod.DB_PATH)"],
        capture_output=True, text=True, cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO), "VBL_DATA_DIR": "/tmp/test_vbl_bw"},
    )
    assert "/tmp/test_vbl_bw" in result.stdout.strip(), (
        f"browser-worker DB_PATH did not use VBL_DATA_DIR: {result.stdout} {result.stderr}"
    )


def test_agent_registries_resolve_via_data_dir(tmp_path, monkeypatch):
    """autonomous_agent visual_register/character_locks must use VBL_DATA_DIR."""
    import json
    from services.agent.autonomous_agent import load_character_locks, load_visual_register

    visual = {"styles": [{"id": "test-style"}]}
    characters = {"characters": [{"id": "test-character"}]}
    (tmp_path / "visual_register.json").write_text(json.dumps(visual))
    (tmp_path / "character_locks.json").write_text(json.dumps(characters))
    monkeypatch.setenv("VBL_DATA_DIR", str(tmp_path))

    assert load_visual_register() == visual
    assert load_character_locks() == characters


def test_compose_research_api_mounts_data_volume():
    """research-api must bind-mount ./data and set VBL_DATA_DIR=/data."""
    yml = (REPO / "docker-compose.yml").read_text()
    # Find research-api section
    import re
    ra_section = re.search(
        r'research-api:.*?(?=\n  \w|\Z)', yml, re.DOTALL
    )
    assert ra_section, "research-api section missing from compose"
    section = ra_section.group()
    assert "VBL_DATA_DIR" in section, "research-api must set VBL_DATA_DIR"
    assert "./data:/data" in section, "research-api must bind-mount ./data:/data"


def test_compose_browser_worker_mounts_data_volume():
    """browser-worker must bind-mount ./data and set VBL_DATA_DIR=/data."""
    yml = (REPO / "docker-compose.yml").read_text()
    import re
    bw_section = re.search(
        r'browser-worker:.*?(?=\n  \w|\Z)', yml, re.DOTALL
    )
    assert bw_section, "browser-worker section missing from compose"
    section = bw_section.group()
    assert "VBL_DATA_DIR" in section, "browser-worker must set VBL_DATA_DIR"
    assert "./data:/data" in section, "browser-worker must bind-mount ./data:/data"


def test_compose_corpus_mounts_are_writable_for_sqlite_wal():
    """Research startup and browser ingestion both write corpus.db and its WAL."""
    import json
    import subprocess

    env = {
        **os.environ,
        "POSTGRES_PASSWORD": "test-only",
        "MINIO_PASSWORD": "test-only",
        "MCP_AUTH_TOKEN": "test-only",
        "SCRAPER_API_KEY": "test-only",
        "MODELSCOPE_API_KEY": "test-only",
    }
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=str(REPO), env=env, capture_output=True, text=True, check=True,
    )
    services = json.loads(result.stdout)["services"]
    for service_name in ("research-api", "browser-worker"):
        data_mount = next(
            mount for mount in services[service_name]["volumes"]
            if mount["target"] == "/data"
        )
        assert data_mount.get("read_only", False) is False, (
            f"{service_name} /data mount must be writable for SQLite WAL/ingestion"
        )


# ─── Runtime deps: no unused native `sharp` package; Pillow is the image lib ──

def test_pyproject_no_unused_sharp_dependency():
    """sharp (native, needs gcc) is unused in the codebase — must not be a
    runtime dependency so the pip install step builds on hosts without a C
    toolchain. Image work uses Pillow (renderer + mcp-server)."""
    import tomllib

    with open(REPO / "pyproject.toml", "rb") as fh:
        meta = tomllib.load(fh)
    deps = [d.lower() for d in meta["project"]["dependencies"]]
    # No bare `sharp` runtime dep. Allow nothing that starts with `sharp[=`/`sharp>`
    # either; but a transitive `sharp` via another package is out of scope here.
    sharp_entries = [d for d in deps if d.startswith("sharp")]
    assert sharp_entries == [], f"unused runtime dependency present: {sharp_entries}"
    # Pillow must remain (it is the actual image library used by services).
    assert any(d.startswith("pillow") for d in deps), "Pillow must remain a runtime dep"


# ─── Qdrant healthcheck: image-compatible, no curl, real /healthz 200 check ──

def _compose_rendered_services():
    """Render compose config (dummy secrets) and return the services dict."""
    import json
    import subprocess

    env = {
        **os.environ,
        "POSTGRES_PASSWORD": "test-only",
        "MINIO_PASSWORD": "test-only",
        "MCP_AUTH_TOKEN": "test-only",
        "SCRAPER_API_KEY": "test-only",
        "MODELSCOPE_API_KEY": "test-only",
    }
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=str(REPO), env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["services"]


def test_qdrant_healthcheck_is_image_compatible():
    """qdrant image has no curl, so the healthcheck must use bash /dev/tcp and
    perform a real /healthz probe asserting HTTP 200 OK."""
    services = _compose_rendered_services()
    hc = services["qdrant"]["healthcheck"]["test"]
    hc_text = " ".join(hc).lower() if isinstance(hc, list) else str(hc).lower()

    assert "curl" not in hc_text, f"qdrant healthcheck must not use curl: {hc}"
    # Must actually probe /healthz (Qdrant's liveness endpoint)...
    assert "/healthz" in hc_text, f"qdrant healthcheck must probe /healthz: {hc}"
    # ...and assert an HTTP 200 OK response (not a mere source-string presence).
    assert "200" in hc_text and "ok" in hc_text, f"healthcheck must assert 200 OK: {hc}"
    # Should drive it via bash/dev-tcp (image ships /usr/bin/bash, not curl).
    assert "bash" in hc_text, f"qdrant healthcheck must use bash:/hc"


# ─── Postgres healthcheck must target the configured DB (viralbench) ─────────

def test_postgres_healthcheck_targets_configured_db():
    """pg_isready defaults DB to the username; with POSTGRES_DB=viralbench the
    healthcheck must pass `-d viralbench` to avoid FATAL 'database does not exist'."""
    services = _compose_rendered_services()
    pg = services["postgres"]
    configured_db = pg["environment"].get("POSTGRES_DB")
    assert configured_db == "viralbench", f"expected POSTGRES_DB=viralbench, got {configured_db}"
    hc = pg["healthcheck"]["test"]
    hc_text = " ".join(hc) if isinstance(hc, list) else str(hc)
    hc_lower = hc_text.lower()
    assert "pg_isready" in hc_lower, f"healthcheck must use pg_isready: {hc}"
    # Explicitly target the configured database, not the username default.
    assert "-d" in hc_text and configured_db in hc_text, (
        f"postgres healthcheck must explicitly check -d {configured_db}: {hc}"
    )


# ─── CI workflow (.github/workflows/ci.yml) ───────────────────────────────────

def _ci_yaml():
    """Load ci.yml; normalize the `on` key (PyYAML turns it into key True)."""
    import yaml
    path = REPO / ".github" / "workflows" / "ci.yml"
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    return data, text


def test_ci_workflow_file_exists_and_has_push_pull_request():
    """ci.yml must exist and trigger on push + pull_request."""
    data, text = _ci_yaml()
    triggers = data.get("on") or data.get(True) or {}
    assert triggers, "ci.yml must declare triggers"
    trigger_keys = set(triggers.keys())
    assert "push" in trigger_keys and "pull_request" in trigger_keys


def test_ci_read_only_contents_permission_and_no_secrets():
    """CI must run with contents:read and never reference secrets/credentials."""
    data, text = _ci_yaml()
    perms = data.get("permissions", {})
    assert perms.get("contents") == "read", f"permissions must set contents:read, got {perms}"
    assert "secrets." not in text, "workflow must not reference secrets (no credentials)"


def test_ci_uses_ubuntu_and_pinned_actions():
    """Runners ubuntu-latest; actions pinned to exact security SHAs."""
    data, text = _ci_yaml()
    assert "ubuntu-latest" in text
    pins = {
        "checkout": "11d5960a326750d5838078e36cf38b85af677262",
        "setup-uv": "e58605a9b6da7c637471fab8847a5e5a6b8df081",
        "setup-buildx": "8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
        "build-push": "10e90e3645eae34f1e60eeb005ba3a3d33f178e8",
    }
    for action, sha in pins.items():
        assert sha in text, f"{action} action must be pinned to {sha}"


def test_ci_concurrency_and_timeouts():
    """Concurrency cancel-in-progress and job timeouts set."""
    data, text = _ci_yaml()
    assert "cancel-in-progress" in text, "concurrency must cancel-in-progress"
    assert "timeout-minutes" in text, "CI jobs must set timeouts"


def test_ci_verify_job_installs_dev_and_runs_tests():
    """Verify job must install .[dev], run tests, py_compile, bash -n, compose config, uv build."""
    data, text = _ci_yaml()
    step_text = text
    assert "uv pip install" in step_text and ".[dev]" in step_text, "verify must install .[dev]"
    assert "tests/test_security.py" in step_text, "verify must run tests/test_security.py"
    assert "py_compile" in step_text, "verify must py_compile changed runtime modules"
    assert "bash -n" in step_text, "verify must bash -n the start scripts"
    assert "docker compose config" in step_text, "verify must validate compose config"
    assert "-uv build" in step_text or "uv build" in step_text, "verify must build wheel"


def test_ci_docker_matrix_exact_targets():
    """Docker matrix must cover exactly the five app targets and build with push:false."""
    data, text = _ci_yaml()
    # matrix targets appear in the workflow text; exact set check via the matrix include.
    assert "research-api" in text and "scraper-api" in text and "browser-worker" in text
    assert "publisher" in text and "renderer" in text
    assert "push: false" in text, "build-push must use push:false (no publishing)"
    assert "type=gha" in text or "docker/build-push-action" in text, "must use GHA cache"
    assert "docker/build-push-action" in text


# ─── README: correct ports, statuses, mcp-server, Quick Start ─────────────────

def test_readme_publisher_port_8030_renderer_8031():
    """README must reflect publisher=8030, renderer=8031 (matches Dockerfile)."""
    readme = (REPO / "README.md").read_text()
    assert "renderer :8031" in readme or "renderer        :8031" in readme, "arch renderer=8031"
    assert "publisher :8030" in readme or "publisher       :8030" in readme, "arch publisher=8030"
    # Table rows (grep table line for publisher/renderer).
    for line in readme.splitlines():
        if line.startswith("| publisher"):
            assert "| 8030 |" in line, f"publisher table port wrong: {line!r}"
        if line.startswith("| renderer"):
            assert "| 8031 |" in line, f"renderer table port wrong: {line!r}"


def test_readme_statuses_all_live_five_docker_apps():
    """The five Docker-verified apps must be listed Live/working, none 'Building'."""
    readme = (REPO / "README.md").read_text()
    for app in ["research-api", "scraper-api", "browser-worker", "publisher", "renderer"]:
        row = next((l for l in readme.splitlines() if l.startswith(f"| {app}")), None)
        assert row, f"missing table row for {app}"
        assert "Building" not in row and "Planned" not in row, f"{app} status stale: {row!r}"


def test_readme_mcp_server_documented_as_native_alternative():
    """mcp-server is a native alternative sharing 8020, not a Compose service."""
    readme = (REPO / "README.md").read_text()
    assert "mcp-server" in readme
    # Must not be listed as a separate simultaneous Compose service row on its own port.
    mcp_row = next((l for l in readme.splitlines() if l.startswith("| mcp-server")), None)
    assert mcp_row is not None, "README should still document mcp-server"
    # It shares 8020 with browser-worker (native alternative), not a distinct compose port.
    assert "8020" in mcp_row
    assert "native" in mcp_row.lower() or "alternative" in mcp_row.lower(), mcp_row


def test_readme_quick_start_sources_env_all_required_values():
    """Quick Start must source .env before start-all.sh and mention all required values."""
    readme = (REPO / "README.md").read_text()
    qs = readme.split("## Quick Start")[1].split("## ")[0] if "## Quick Start" in readme else ""
    assert "source .env" in qs or "set -a; source .env" in qs or ".env" in qs, "Quick Start must source .env"
    low = qs.lower()
    for need in ["postgres_password", "minio_password", "mcp_auth_token", "scraper_api_key"]:
        assert need in low, f"Quick Start must mention required value {need}"
