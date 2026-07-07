from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
CTHR_ROOT = THIS_DIR.parents[1]
PAPER_DIR = CTHR_ROOT / "paper"
OUT_ROOT = PAPER_DIR / "aviation_rule_library_model_comparison"

sys.path.insert(0, str(THIS_DIR))
import run_llm_rule_validation as rv  # noqa: E402


PROVIDER_ALIASES = {
    "tongyi": "qwen",
    "qwen": "qwen",
    "deepseek": "deepseek",
    "glm": "glm",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_cases(cases_path: Path, domain: str) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = [case for case in payload["cases"] if case.get("domain") == domain]
    return payload.get("version", "unknown"), cases


def normalize_provider_names(raw: str) -> list[str]:
    names = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in PROVIDER_ALIASES:
            raise ValueError(f"Unknown provider: {item}")
        names.append(PROVIDER_ALIASES[key])
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def ensure_provider_env(provider_name: str) -> tuple[bool, str]:
    if provider_name == "qwen":
        if os.environ.get("QWEN_API_KEY"):
            return True, "QWEN_API_KEY"
        if os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["QWEN_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
            return True, "DASHSCOPE_API_KEY"
        return False, "QWEN_API_KEY or DASHSCOPE_API_KEY"
    provider = rv.PROVIDERS[provider_name]
    if os.environ.get(provider.env_key):
        return True, provider.env_key
    return False, provider.env_key


def apply_model_overrides(qwen_model: str, deepseek_model: str, glm_model: str) -> None:
    rv.PROVIDERS["qwen"] = rv.Provider(
        name=rv.PROVIDERS["qwen"].name,
        env_key=rv.PROVIDERS["qwen"].env_key,
        url=rv.PROVIDERS["qwen"].url,
        model=qwen_model,
    )
    rv.PROVIDERS["deepseek"] = rv.Provider(
        name=rv.PROVIDERS["deepseek"].name,
        env_key=rv.PROVIDERS["deepseek"].env_key,
        url=rv.PROVIDERS["deepseek"].url,
        model=deepseek_model,
    )
    rv.PROVIDERS["glm"] = rv.Provider(
        name=rv.PROVIDERS["glm"].name,
        env_key=rv.PROVIDERS["glm"].env_key,
        url=rv.PROVIDERS["glm"].url,
        model=glm_model,
    )


def checkpoint_path(provider_name: str, case_id: str) -> Path:
    return OUT_ROOT / provider_name / f"{case_id}.json"


def is_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def run_one_case(
    provider_name: str,
    case: dict[str, Any],
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    provider = rv.PROVIDERS[provider_name]
    started_at = now()
    try:
        result = rv.run_case(
            case=case,
            system_prompt=system_prompt,
            providers=[provider],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        result["started_at"] = started_at
        result["finished_at"] = now()
        result["model_family"] = provider_name
        return result
    except Exception as exc:  # noqa: BLE001 - this is a checkpointing runner.
        return {
            "case_id": case["case_id"],
            "domain": case.get("domain"),
            "interaction_type": case.get("interaction_type"),
            "status": "failed",
            "model_family": provider_name,
            "started_at": started_at,
            "finished_at": now(),
            "error": str(exc),
        }


def collect_provider_results(provider_name: str, case_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for case_id in case_ids:
        path = checkpoint_path(provider_name, case_id)
        if not path.exists():
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            rows.append({"case_id": case_id, "status": "corrupt_checkpoint"})
    return rows


def build_rule_library(provider_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("rule_base")]
    rules = []
    cases = []
    for row in ok_rows:
        case_rules = row["rule_base"].get("rules", [])
        rules.extend(case_rules)
        cases.append(
            {
                "case_id": row["case_id"],
                "interaction_type": row.get("interaction_type"),
                "rule_ids": [rule.get("rule_id") for rule in case_rules],
                "grounding": row.get("grounding"),
                "semantic": row.get("semantic"),
            }
        )
    return {
        "generated_at": now(),
        "provider": provider_name,
        "source": "completed aviation KG evidence cases supplied by validation_cases.json",
        "num_cases_ok": len(ok_rows),
        "num_rules": len(rules),
        "rules": rules,
        "cases": cases,
    }


def summarize_provider(provider_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") not in {"ok", "skipped_missing_key"}]
    skipped_rows = [row for row in rows if row.get("status") == "skipped_missing_key"]
    aggregate = rv.aggregate_metrics(ok_rows) if ok_rows else {}
    return {
        "provider": provider_name,
        "num_ok": len(ok_rows),
        "num_failed": len(failed_rows),
        "num_skipped_missing_key": len(skipped_rows),
        "aggregate": aggregate,
        "case_status": [
            {
                "case_id": row.get("case_id"),
                "status": row.get("status"),
                "interaction_type": row.get("interaction_type"),
                "rule_count": row.get("grounding", {}).get("rule_count")
                if row.get("grounding")
                else None,
                "constraint_count": row.get("grounding", {}).get("constraint_count")
                if row.get("grounding")
                else None,
                "relation_count": row.get("grounding", {}).get("relation_count")
                if row.get("grounding")
                else None,
                "semantic_pass": row.get("semantic", {}).get("case_pass")
                if row.get("semantic")
                else None,
                "error": row.get("error"),
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aviation CTHR rule libraries with multiple LLM providers and resumable checkpoints."
    )
    parser.add_argument("--cases", type=Path, default=rv.CASES_PATH)
    parser.add_argument("--domain", default="aviation")
    parser.add_argument("--providers", default="qwen,deepseek,glm")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--qwen-model", default="qwen-plus")
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--glm-model", default="GLM-4.7")
    parser.add_argument("--force", action="store_true", help="Re-run completed provider/case checkpoints.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first API or parsing failure instead of checkpointing and continuing.",
    )
    args = parser.parse_args()
    apply_model_overrides(args.qwen_model, args.deepseek_model, args.glm_model)

    version, cases = load_cases(args.cases, args.domain)
    provider_names = normalize_provider_names(args.providers)
    system_prompt = rv.PROMPT_PATH.read_text(encoding="utf-8")
    case_ids = [case["case_id"] for case in cases]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": now(),
        "cases_path": str(args.cases),
        "validation_set_version": version,
        "domain": args.domain,
        "num_cases": len(cases),
        "providers": provider_names,
        "models": {
            "qwen": rv.PROVIDERS["qwen"].model,
            "deepseek": rv.PROVIDERS["deepseek"].model,
            "glm": rv.PROVIDERS["glm"].model,
        },
        "checkpoint_root": str(OUT_ROOT),
        "note": "API keys are read only from environment variables and are not written to outputs.",
    }
    safe_write_json(OUT_ROOT / "manifest.json", manifest)

    for provider_name in provider_names:
        key_ok, key_name = ensure_provider_env(provider_name)
        provider_dir = OUT_ROOT / provider_name
        provider_dir.mkdir(parents=True, exist_ok=True)
        if not key_ok:
            for case in cases:
                path = checkpoint_path(provider_name, case["case_id"])
                if path.exists() and not args.force:
                    continue
                safe_write_json(
                    path,
                    {
                        "case_id": case["case_id"],
                        "domain": case.get("domain"),
                        "interaction_type": case.get("interaction_type"),
                        "status": "skipped_missing_key",
                        "model_family": provider_name,
                        "required_env": key_name,
                        "finished_at": now(),
                    },
                )
            print(f"[{now()}] {provider_name}: skipped because {key_name} is missing")
            continue

        for idx, case in enumerate(cases, start=1):
            path = checkpoint_path(provider_name, case["case_id"])
            if is_completed(path) and not args.force:
                print(f"[{now()}] {provider_name} {idx}/{len(cases)} {case['case_id']}: checkpoint ok, skip")
                continue
            print(f"[{now()}] {provider_name} {idx}/{len(cases)} {case['case_id']}: calling API")
            row = run_one_case(
                provider_name=provider_name,
                case=case,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            safe_write_json(path, row)
            print(f"[{now()}] {provider_name} {case['case_id']}: {row.get('status')}")
            if args.stop_on_error and row.get("status") != "ok":
                raise SystemExit(f"Stopped on error: {provider_name}/{case['case_id']}")

    summaries = []
    for provider_name in provider_names:
        rows = collect_provider_results(provider_name, case_ids)
        library = build_rule_library(provider_name, rows)
        summary = summarize_provider(provider_name, rows)
        summaries.append(summary)
        safe_write_json(OUT_ROOT / f"aviation_rule_library_{provider_name}.json", library)
        safe_write_json(OUT_ROOT / f"summary_{provider_name}.json", summary)

    comparison = {
        "generated_at": now(),
        "cases_path": str(args.cases),
        "validation_set_version": version,
        "domain": args.domain,
        "num_cases": len(cases),
        "summaries": summaries,
    }
    safe_write_json(OUT_ROOT / "model_comparison_summary.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
