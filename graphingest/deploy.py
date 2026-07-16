"""
GraphIngest deploy() — push code to the platform.

Scans your project for @node/@graph functions, optionally reads a local
.env file, and uploads everything to the GraphIngest platform. The platform
builds an execution environment and registers your functions for remote
execution via .run(), .map(), and .arun().
"""

import os
import re
import glob
import logging
from pathlib import Path
from typing import Optional

from .client import GraphIngestClient

logger = logging.getLogger(__name__)

_DECORATOR_PATTERN = re.compile(r"@(?:node|graph)\s*\(")


def _find_source_files(project_dir: str) -> list[str]:
    """Find all .py files containing @node or @graph decorators."""
    matches = []
    for py_file in glob.glob(os.path.join(project_dir, "**", "*.py"), recursive=True):
        # Skip hidden dirs, __pycache__, .venv, etc.
        parts = Path(py_file).parts
        if any(p.startswith(".") or p == "__pycache__" or p in ("venv", ".venv", "node_modules") for p in parts):
            continue
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if _DECORATOR_PATTERN.search(content):
                matches.append(py_file)
        except (OSError, UnicodeDecodeError):
            continue
    return matches


def _read_env_file(env_file: str) -> dict[str, str]:
    """Parse a .env file into a dict of key-value pairs."""
    env_vars: dict[str, str] = {}
    if not os.path.isfile(env_file):
        return env_vars
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key:
                    env_vars[key] = value
    return env_vars


def _read_requirements(requirements_path: str) -> Optional[str]:
    """Read requirements.txt if it exists."""
    if os.path.isfile(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def deploy(
    *,
    env_path: Optional[str] = None,
    requirements: str = "requirements.txt",
    project_dir: Optional[str] = None,
) -> dict:
    """Push your code to the GraphIngest platform.

    Scans for files containing @node/@graph decorators and uploads them
    to the platform. The platform builds an execution environment and
    makes your functions available for .run(), .map(), .arun().

    Environment variables:
      - If ``env_path`` is provided, reads that file (relative or absolute)
        and uploads those variables. Dashboard variables with the same key
        take precedence at runtime.
      - If ``env_path`` is omitted, all env vars come from the dashboard
        (Env Variables page).

    Args:
        env_path: Optional path to a .env file (relative or absolute).
                  Accepts any name: ".env", ".env.local", "config/prod.env", etc.
                  If None (default), env vars are read from the dashboard only.
        requirements: Path to requirements file (default: "requirements.txt")
        project_dir: Project root directory (default: current working directory)

    Returns:
        dict with deployment status and registered functions

    Examples:
        # Dashboard-only (no local env file):
        >>> deploy()

        # With a local .env file:
        >>> deploy(env_path=".env")

        # With .env.local (overrides committed .env on dashboard):
        >>> deploy(env_path=".env.local")

        # Absolute path:
        >>> deploy(env_path="/Users/me/secrets/prod.env")
    """
    if project_dir is None:
        project_dir = os.getcwd()

    # 1. Find source files with @node/@graph
    print("Scanning for @node/@graph functions...")
    source_files = _find_source_files(project_dir)
    if not source_files:
        raise FileNotFoundError(
            f"No Python files with @node or @graph decorators found in {project_dir}"
        )
    print(f"  Found {len(source_files)} file(s) with @node/@graph decorators")

    # 2. Read env file (only if env_path provided)
    env_vars: dict[str, str] = {}
    if env_path is not None:
        resolved = env_path if os.path.isabs(env_path) else os.path.join(project_dir, env_path)
        env_vars = _read_env_file(resolved)
        if env_vars:
            print(f"Environment variables (from {env_path}):")
            for key in sorted(env_vars):
                print(f"  ✓ {key}")
        else:
            print(f"  Warning: {env_path} not found or empty — using dashboard variables only")
    else:
        print("  No env_path provided — using dashboard variables only")

    # 3. Read requirements
    req_path = os.path.join(project_dir, requirements) if not os.path.isabs(requirements) else requirements
    requirements_content = _read_requirements(req_path)
    if requirements_content:
        dep_count = len([l for l in requirements_content.strip().splitlines() if l.strip() and not l.startswith("#")])
        print(f"  Found {dep_count} dependencies in {requirements}")
    else:
        print(f"  No {requirements} found — only graphingest will be installed")

    # 4. Prepare payload
    files_payload = {}
    for filepath in source_files:
        rel_path = os.path.relpath(filepath, project_dir)
        with open(filepath, "r", encoding="utf-8") as f:
            files_payload[rel_path] = f.read()

    payload = {
        "files": files_payload,
        "requirements": requirements_content,
        "env_vars": env_vars,
        "language": "python",
    }

    print("Uploading to GraphIngest platform...")
    client = GraphIngestClient()
    result = client.deploy(payload)

    # 6. Show dashboard env var summary
    dashboard_vars = result.get("dashboard_env_vars", [])
    if dashboard_vars:
        dashboard_only = [k for k in dashboard_vars if k not in env_vars]
        overrides = [k for k in dashboard_vars if k in env_vars]
        if dashboard_only:
            print("Dashboard variables:")
            for key in sorted(dashboard_only):
                print(f"  ✓ {key}")
        if overrides:
            print("Dashboard overrides (take precedence over env file):")
            for key in sorted(overrides):
                print(f"  ⚠ {key}")

    # 7. Report success
    functions = result.get("functions", [])
    print(f"Deployed. {len(functions)} function(s) registered:")
    for fn in functions:
        print(f"  • {fn}")

    return result
