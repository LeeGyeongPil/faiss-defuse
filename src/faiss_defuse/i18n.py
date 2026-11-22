"""Language selection and message lookup for the command line.

Order: ``--lang`` > ``FAISS_DEFUSE_LANG`` > ``LC_ALL`` / ``LC_MESSAGES`` / ``LANG``
(a value starting with ``ko`` selects Korean) > English.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources
from typing import Mapping

LANGUAGES = ("en", "ko")
ENV_VAR = "FAISS_DEFUSE_LANG"


def resolve_lang(cli_value: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Return ``"en"`` or ``"ko"``."""
    env = os.environ if env is None else env
    if cli_value in LANGUAGES:
        return str(cli_value)
    own = env.get(ENV_VAR, "").strip().lower()
    if own in LANGUAGES:
        return own
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = env.get(name, "").strip()
        if value:  # the first non-empty locale variable decides (POSIX precedence)
            return "ko" if value.lower().startswith("ko") else "en"
    return "en"


@lru_cache(maxsize=None)
def messages(lang: str) -> dict[str, str]:
    text = resources.files("faiss_defuse").joinpath("locales", f"{lang}.json").read_text(encoding="utf-8")
    return json.loads(text)


def t(lang: str, key: str, **kwargs: object) -> str:
    """Look up ``key`` in ``lang`` and fill the placeholders."""
    return messages(lang)[key].format(**kwargs)
