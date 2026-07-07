from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_OUT_ROOT = ROOT / "results" / "kg_to_rule_library" / "architecture_mimo"
DEFAULT_MIMO_API_KEY = os.environ.get("MIMO_API_KEY") or os.environ.get("XIAOMI_API_KEY")

sys.path.insert(0, str(THIS_DIR))
import run_architecture_kg_rule_library as arch  # noqa: E402


def register_mimo_provider(model: str, api_url: str, env_key: str) -> None:
    if DEFAULT_MIMO_API_KEY:
        os.environ[env_key] = DEFAULT_MIMO_API_KEY
    elif not os.environ.get(env_key):
        for alias in ("MIMO_API_KEY", "XIAOMI_API_KEY"):
            if os.environ.get(alias):
                os.environ[env_key] = os.environ[alias]
                break

    arch.base.rv.PROVIDERS["mimo"] = arch.base.rv.Provider(
        name="mimo",
        env_key=env_key,
        url=api_url,
        model=model,
    )
    arch.base.PROVIDER_ALIASES.update(
        {
            "mimo": "mimo",
            "xiaomi": "mimo",
            "xiaomi_mimo": "mimo",
            "xiaomi-mimo": "mimo",
        }
    )


def has_flag(args: list[str], flag: str) -> bool:
    return flag in args or any(item.startswith(flag + "=") for item in args)


def append_default(args: list[str], flag: str, value: str) -> None:
    if not has_flag(args, flag):
        args.extend([flag, value])


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--mimo-model", default="mimo-v2.5-pro")
    wrapper.add_argument("--mimo-api-url", default="https://api.xiaomimimo.com/v1/chat/completions")
    wrapper.add_argument("--mimo-env-key", default="XIAOMI_MIMO_API_KEY")
    wrapper.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    wrapper_args, remaining = wrapper.parse_known_args()

    register_mimo_provider(
        model=wrapper_args.mimo_model,
        api_url=wrapper_args.mimo_api_url,
        env_key=wrapper_args.mimo_env_key,
    )
    arch.OUT_ROOT = wrapper_args.output_root

    append_default(remaining, "--providers", "mimo")
    append_default(remaining, "--batch-size", "1")
    append_default(remaining, "--timeout", "600")
    append_default(remaining, "--max-tokens", "12000")
    append_default(remaining, "--temperature", "0.0")

    sys.argv = [sys.argv[0], *remaining]
    arch.main()


if __name__ == "__main__":
    main()
