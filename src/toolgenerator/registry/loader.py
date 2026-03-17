"""Load ToolBench toolenv tools from data/toolenv/tools/{Category}/{tool_name}.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_toolbench_tools(tools_root: Path) -> list[dict[str, Any]]:
    """
    Discover and load all ToolBench tool JSONs under tools_root.

    Expects layout: tools_root / {Category} / {tool_name}.json
    Only loads direct children of category directories (one level).
    Adds a "category" key to each raw dict from the parent directory name.
    Skips non-JSON files, invalid JSON, and entries without api_list (handled
    later by normalizer); logs and continues on per-file errors.

    Returns:
        List of raw tool dicts (each with added "category" key).
    """
    tools_root = Path(tools_root).resolve()
    if not tools_root.is_dir():
        logger.warning("Tools root is not a directory: %s", tools_root)
        return []

    raw_tools: list[dict[str, Any]] = []

    # Subdirs: tools_root / {Category} / *.json (standard ToolBench layout)
    category_dirs = [p for p in sorted(tools_root.iterdir()) if p.is_dir()]
    if not category_dirs:
        # Flat layout (e.g. data/sample): treat root as single category
        category_dirs = [tools_root]
        category_name = tools_root.name
    else:
        category_name = None  # use each dir's name

    for category_dir in category_dirs:
        category = category_name if category_name is not None else category_dir.name
        for path in sorted(category_dir.iterdir()):
            if path.suffix.lower() != ".json" or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                data = json.loads(text)
            except (OSError, json.JSONDecodeError) as e:
                logger.debug("Skip %s: %s", path, e)
                continue

            if not isinstance(data, dict):
                continue
            data = dict(data)
            data["category"] = category
            raw_tools.append(data)

    return raw_tools
