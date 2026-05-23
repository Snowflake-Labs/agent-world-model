import importlib
from enum import Enum
from simpleArgParser import parse_args_with_commands


class TopCmd(Enum):
    # Synthesis pipeline commands
    gen = "gen"
    # Environment management commands
    env = "env"
    # Agent commands
    agent = "agent"
    # Verify agent run outputs using code-augmented LLM-as-a-Judge or purely code-based Judge
    verify = "verify"
    # please install mcp-adapated-bench first, for benchmarking agents on mcp-adapted-bench including bfclv3, tau2, and mcp-universe
    bench = "bench"


class GenCmd(Enum):
    # Generate scenario names from seed set
    scenario = "scenario"
    # Generate user tasks per scenario
    task = "task"
    # Generate database schema and create SQLite databases
    db = "db"
    # Generate and insert sample data into databases
    sample = "sample"
    # Generate API specification for each scenario
    spec = "spec"
    # Generate MCP environment code
    env = "env"
    # Generate verification code for tasks
    verifier = "verifier"
    # Run the full synthesis pipeline
    all = "all"


class EnvCmd(Enum):
    # Start MCP server for a scenario
    start = "start"
    # Check if an MCP server is running and list its tools
    check = "check"
    # Check all generated environments
    check_all = "check_all"
    # Reset databases to initial state
    reset_db = "reset_db"


def _build_commands() -> dict:
    from awm.core.scenario import Config as ScenarioConfig
    from awm.core.task import Config as TaskConfig
    from awm.core.db import Config as DbConfig
    from awm.core.sample import Config as SampleConfig
    from awm.core.spec import Config as SpecConfig
    from awm.core.env import Config as EnvConfig
    from awm.core.verifier import Config as VerifierConfig
    from awm.core.pipeline import Config as PipelineConfig
    from awm.core.server import Config as ServeConfig
    from awm.core.check import Config as CheckServerConfig
    from awm.core.test_env import Config as TestEnvConfig
    from awm.core.reset import Config as ResetConfig
    from awm.core.agent import Config as AgentConfig
    from awm.core.verify import Config as VerifyConfig

    try:
        from awm.eval.infer_eval import InferEvalConfig
    except ImportError as e:
        InferEvalConfig = None

    return {
        TopCmd.gen: {
            GenCmd.scenario: ScenarioConfig,
            GenCmd.task: TaskConfig,
            GenCmd.db: DbConfig,
            GenCmd.sample: SampleConfig,
            GenCmd.spec: SpecConfig,
            GenCmd.env: EnvConfig,
            GenCmd.verifier: VerifierConfig,
            GenCmd.all: PipelineConfig,
        },
        TopCmd.env: {
            EnvCmd.start: ServeConfig,
            EnvCmd.check: CheckServerConfig,
            EnvCmd.check_all: TestEnvConfig,
            EnvCmd.reset_db: ResetConfig,
        },
        TopCmd.agent: AgentConfig,
        TopCmd.verify: VerifyConfig,
        TopCmd.bench: InferEvalConfig,
    }


# map command paths to their module's run() function
DISPATCH = {
    (TopCmd.gen, GenCmd.scenario): "awm.core.scenario",
    (TopCmd.gen, GenCmd.task): "awm.core.task",
    (TopCmd.gen, GenCmd.db): "awm.core.db",
    (TopCmd.gen, GenCmd.sample): "awm.core.sample",
    (TopCmd.gen, GenCmd.spec): "awm.core.spec",
    (TopCmd.gen, GenCmd.env): "awm.core.env",
    (TopCmd.gen, GenCmd.verifier): "awm.core.verifier",
    (TopCmd.gen, GenCmd.all): "awm.core.pipeline",
    (TopCmd.env, EnvCmd.start): "awm.core.server",
    (TopCmd.env, EnvCmd.check): "awm.core.check",
    (TopCmd.env, EnvCmd.check_all): "awm.core.test_env",
    (TopCmd.env, EnvCmd.reset_db): "awm.core.reset",
    (TopCmd.agent,): "awm.core.agent",
    (TopCmd.verify,): "awm.core.verify",
    (TopCmd.bench,): "awm.eval.infer_eval",
}


def main():
    commands = _build_commands()
    command_path, config = parse_args_with_commands(
        commands,
        description="AWM - Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning",
    )

    module_name = DISPATCH.get(command_path)
    if module_name is None:
        print(f"Error: unknown command path {command_path}")
        exit(1)

    module = importlib.import_module(module_name)
    module.run(config)


if __name__ == "__main__":
    main()
