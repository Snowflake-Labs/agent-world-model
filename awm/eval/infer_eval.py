"""
Inference + evaluation entry point for agent-world-model.
This is a minimal bridge around `mcp-adapted-bench`: https://github.com/Raibows/mcp-adapted-bench 
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from simpleArgParser import parse_args

_HERE = Path(__file__).resolve()
_AWM_ROOT = _HERE.parent.parent.parent  # agent-world-model/
_CANDIDATE_PATHS = [
    _AWM_ROOT / "mcp-adapted-bench",
]
for _p in _CANDIDATE_PATHS:
    if (_p / "mcp_adapted_bench").exists():
        sys.path.insert(0, str(_p))
        break


from awm.tools import (
    tools_robust_json_loads,
    tools_json_save,
)

from mcp_adapted_bench.common.infer_config import (
    EvalMode,
    InferEvalConfig,
)
from mcp_adapted_bench.common.llm_client import (
    LLMCaller,
    build_openai_client,
)
from mcp_adapted_bench.bfcl.runner import run_bfcl_evaluation
from mcp_adapted_bench.tau2.runner import run_tau2_evaluation
from mcp_adapted_bench.mcp_universe.runner import run_mcp_universe_evaluation



logger.remove()
logger.add(sys.stdout, level="INFO")




def _read_completed_task_ids(output_dir: Path) -> set[str]:
    """Return set of task_ids that have a non-error trajectory on disk."""
    traj_dir = output_dir / "agentfly_trajectories"
    if not traj_dir.exists():
        return set()
    completed: set[str] = set()
    for p in traj_dir.glob("*.json"):
        try:
            data = tools_robust_json_loads(p.as_posix())
        except Exception:
            continue
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            continue
        tid = data.get("task_id")
        if tid is not None:
            completed.add(str(tid))
    return completed


def _setup_resume(config: InferEvalConfig) -> None:
    """
    Implementation of the simplified resume rule from the spec.
    """
    if not config.resume or not config.output_dir:
        return
    od = Path(config.output_dir)
    if not od.exists():
        logger.warning(f"--resume but {od} does not exist; starting fresh")
        return
    prev_cfg_file = od / "config.json"
    if prev_cfg_file.exists():
        try:
            prev = tools_robust_json_loads(prev_cfg_file.as_posix())
            if prev.get("mode") != config.mode.value:
                logger.warning(
                    f"--resume: previous run mode={prev.get('mode')} != {config.mode.value}. "
                    "Ignoring stale config; existing trajectories may be skipped anyway."
                )
        except Exception:
            pass
    n = len(_read_completed_task_ids(od))
    logger.info(f"Resume: found {n} completed trajectories in {od}")


def _save_config(config: InferEvalConfig) -> None:
    if not config.output_dir:
        return
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Shallow snapshot for audit purposes.
    snap = {
        "mode": config.mode.value,
        "model": config.model,
        "api_url": config.api_url,
        "limit": config.limit,
        "num_rollouts": config.num_rollouts,
        "max_turns": config.max_turns,
        "history_limit": config.history_limit,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "max_completion_tokens": config.max_completion_tokens,
        "max_concurrency": config.max_concurrency,
        "tau2_domains": list(config.tau2.domains),
        "mcp_universe_categories": list(config.mcp_universe.categories),
        "bfcl_test_categories": list(config.bfcl.test_categories),
        "note": config.note,
        "saved_at": datetime.now().isoformat(),
    }
    tools_json_save(snap, (out / "config.json").as_posix())



async def _run_async(config: InferEvalConfig) -> dict:
    # Build the LLM client (fails fast if env is misconfigured).
    client, model, api_url = build_openai_client(
        api_url=config.api_url, model_override=config.model
    )
    config.api_url = api_url
    config.model = model
    logger.info(f"[agent LLM] model={model} base_url={api_url}")

    caller = LLMCaller(
        client=client,
        model=model,
        api_url=api_url,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        min_tokens=config.min_tokens,
        repetition_penalty=config.repetition_penalty,
        add_generation_prompt=config.add_generation_prompt,
        enable_thinking=config.enable_thinking,
        max_completion_tokens=config.max_completion_tokens,
        max_concurrency=config.max_concurrency,
        tolerant_mode=config.tolerant_mode,
    )

    _setup_resume(config)
    _save_config(config)

    if config.mode == EvalMode.bfcl:
        return await run_bfcl_evaluation(caller, config)

    if config.mode == EvalMode.tau2:
        return await run_tau2_evaluation(caller, config)

    if config.mode == EvalMode.mcp_universe:
        return await run_mcp_universe_evaluation(caller, config)

    raise ValueError(f"Unknown eval mode: {config.mode}")

def run(config: InferEvalConfig):
    summary = asyncio.run(_run_async(config))

    # Write summary.json
    od = Path(config.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    tools_json_save(summary, (od / "summary.json").as_posix())
    logger.info(f"Wrote summary to {od / 'summary.json'}")



if __name__ == "__main__":
    from simpleArgParser import parse_args
    config: InferEvalConfig = parse_args(InferEvalConfig)
    run(config)
