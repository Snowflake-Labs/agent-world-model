
<h1 align="center"><img src="figures/logo.png" alt="AWM Logo" width="80"> Agent World Model</h1>

<h3 align="center">Infinity Synthetic Environments for Agentic Reinforcement Learning</h3>

<p align="center">
  <a href="https://zhaoyang.win">Zhaoyang Wang<sup>1</sup></a>,
  <a href="https://www.canwenxu.net/">Canwen Xu<sup>2</sup></a>,
  <a href="https://www.snowflake.com/en/blog/authors/boyi-liu/">Boyi Liu<sup>2</sup></a>,
  <a href="https://yitewang.github.io/">Yite Wang<sup>2</sup></a>,
  <a href="https://lillianwei-h.github.io/">Siwei Han<sup>1</sup></a>,<br/>
  <a href="https://yaozhewei.github.io/">Zhewei Yao<sup>2</sup></a>,
  <a href="https://www.huaxiuyao.io/">Huaxiu Yao<sup>1</sup></a>,
  <a href="https://www.snowflake.com/en/blog/authors/yuxiong-he/">Yuxiong He<sup>2</sup></a>
</p>
<p align="center">
  <sup>1</sup>UNC-Chapel Hill &nbsp; <sup>2</sup>Snowflake AI Research &nbsp;
</p>


<p align="center">
    <a href="https://arxiv.org/pdf/2602.10090"><img src="https://img.shields.io/badge/arXiv-2602.10090-b31b1b.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K"><img src="https://img.shields.io/badge/🤗-Environments-yellow.svg" alt="HuggingFace"></a>
    <a href="https://huggingface.co/spaces/ChilleD/agent_world_model_env"><img src="https://img.shields.io/badge/🤗-Live Demo-green.svg" alt="HuggingFace"></a>
    <a href="https://huggingface.co/Snowflake/Arctic-AWM-4B"><img src="https://img.shields.io/badge/🤗-AWM4B-blue.svg" alt="HuggingFace"></a>
    <a href="https://huggingface.co/Snowflake/Arctic-AWM-8B"><img src="https://img.shields.io/badge/🤗-AWM8B-blue.svg" alt="HuggingFace"></a>
    <a href="https://huggingface.co/Snowflake/Arctic-AWM-14B"><img src="https://img.shields.io/badge/🤗-AWM14B-blue.svg" alt="HuggingFace"></a>
    <a href="https://github.com/meta-pytorch/OpenEnv/tree/main/envs/agent_world_model_env"><img src="https://img.shields.io/badge/⚓️-RL Infra-orange.svg" alt="RL Infra"></a>
</p>

<p align="left">
  <b>Agent World Model (AWM)</b> is a fully synthetic environment generation pipeline that synthesizes 1,000 executable, SQL database-backed tool-use environments exposed via unified MCP interface for large-scale multi-turn agentic reinforcement learning.
</p>

---

## 📣 News
- May 1, 2026: AWM got accepted to ICML 2026 🎉, and its infra got merged into [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) supporting large-scale agentic RL training! Have a live demo try at [huggingface space](https://huggingface.co/spaces/ChilleD/agent_world_model_env) 🤗!
- Mar 16, 2026: we added the verification demo, please refer to [Verification](#verification) section!
- Feb 10, 2026: we open-sourced the synthesis pipeline, 1,000 synthesized environments and RL trained agents at [Huggingface](https://huggingface.co/collections/Snowflake/agent-world-model)!


## 🔮 Resources
We released the syntheszied 1,000 executable environments and corresponding tasks, databases, and verification in huggingface. Please checkout huggingface repo at [Snowflake/AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K).  You can freely interact with these environments online at [HuggingFace Space](https://huggingface.co/spaces/ChilleD/agent_world_model_env).

| Resource | Link |
|----------|------|
| 📄 Paper | [📄 arxiv.org/abs/2602.10090](https://arxiv.org/abs/2602.10090) |
| 💻 Code | [💻 Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model) |
| ⚓️ RL Infra | [⚓️ meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/agent_world_model_env) |
| 🛜 Live Demo | [🤗 HuggingFace Space](https://huggingface.co/spaces/ChilleD/agent_world_model_env) |
| 📦 AgentWorldModel-1K | [🤗 Snowflake/AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) |
| 🤖 Arctic-AWM-4B | [🤗 Snowflake/Arctic-AWM-4B](https://huggingface.co/Snowflake/Arctic-AWM-4B) |
| 🤖 Arctic-AWM-8B | [🤗 Snowflake/Arctic-AWM-8B](https://huggingface.co/Snowflake/Arctic-AWM-8B) |
| 🤖 Arctic-AWM-14B | [🤗 Snowflake/Arctic-AWM-14B](https://huggingface.co/Snowflake/Arctic-AWM-14B) |

If you want to directly use our synthesized environments, please download by

```bash
hf download Snowflake/AgentWorldModel-1K --repo-type dataset --local-dir ./outputs/
```
Then you can skip to [Environment Management](#environment-management) and [Agent Demo](#agent-demo) to start using the environments locally.


## 📦 Setup
Run `uv sync` to setup the python environment. And set your LLM API credentials:

```bash
# OpenAI or any other compatible services
export AWM_SYN_LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-api-key"
# optional, if you are using a custom base url
export OPENAI_BASE_URL="http://xxxxxx"

# Azure OpenAI
export AWM_SYN_LLM_PROVIDER="azure"
export AZURE_ENDPOINT_URL="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"

# configure the model/LLM for synthesis
export AWM_SYN_OVERRIDE_MODEL="your-model-name such as gpt-5"
```

## 🔥 Synthesis

### AWM CLI

All synthesis is exposed through the `awm` command-line tool. Run `awm --help` to see available commands:

```
awm --help

Available commands:
  gen        Synthesis pipeline commands
  ├── scenario   Generate scenario names from seed set
  ├── task       Generate user tasks per scenario
  ├── db         Generate database schema and create SQLite databases
  ├── sample     Generate and insert sample data into databases
  ├── spec       Generate API specification for each scenario
  ├── env        Generate MCP environment code
  ├── verifier   Generate verification code for tasks
  └── all        Run the full synthesis pipeline
  env        Environment management commands
  ├── start      Start MCP server for a scenario
  ├── check      Check if an MCP server is running and list its tools
  ├── check_all  Check all generated environments
  └── reset_db   Reset databases to initial state
  agent      Run a tool-use agent to solve a task by interacting with the environment
  verify     Verify agent run outputs using code-augmented LLM-as-a-Judge or purely code-based Judge
  bench      Run evaluation on mcp-adapted-bench including bfclv3, tau2, and mcp-universe
```

Use `awm <command> --help` to see options for any command, e.g. `awm gen task --help`.

### Step 1: Scenario Generation
We start with a seed set of scenarios and generate 1,000 unique scenario descriptions. Note that only the names are used as seeds; the descriptions are included in the seed file for ease of use.

```bash
export EMBEDDING_OPENAI_API_KEY="your-api-key for the embedding model"

awm gen scenario \
    --input_path outputs/seed_scenario.jsonl \
    --output_path outputs/gen_scenario.jsonl \
    --target_count 1000
```

### Step 2: Task Generation
We generate 10 tasks per scenario, which are also serving as the requirements for building the environment.

```bash
awm gen task \
    --input outputs/gen_scenario.jsonl \
    --output outputs/gen_tasks.jsonl
```

### Step 3: Database Synthesis
We define the database schema and complete the initial state to fully support the generated tasks.

```bash
# database schema
awm gen db \
    --input outputs/gen_tasks.jsonl \
    --output outputs/gen_db.jsonl

# sample data for initial state
awm gen sample \
    --input_task outputs/gen_tasks.jsonl \
    --input_db outputs/gen_db.jsonl \
    --output outputs/gen_sample.jsonl
```

### Step 4: Interface Synthesis
We first generate API spec for better generating the Python code of the environment in MCP interface.

```bash
# API spec (interface schema)
awm gen spec \
    --input_task outputs/gen_tasks.jsonl \
    --input_db outputs/gen_db.jsonl \
    --output outputs/gen_spec.jsonl

# Environment code
awm gen env \
    --input_spec outputs/gen_spec.jsonl \
    --input_db outputs/gen_db.jsonl \
    --output outputs/gen_envs.jsonl
```

### Step 5: Verification Synthesis
We provide two options for verification:
1. code-augmented LLM-as-a-Judge (`sql`)
2. purely code-based Judge (`code`)

```bash
awm gen verifier \
    --mode sql \
    --input_task outputs/gen_tasks.jsonl \
    --output outputs/gen_verifier.jsonl
```

### Environment Management

Run and check each environment. The MCP endpoint will be available at `http://localhost:8001/mcp`.

```bash
# Reset databases to initial state
awm env reset_db \
    --input_db outputs/gen_db.jsonl \
    --input_sample outputs/gen_sample.jsonl

# Start MCP server for a scenario
awm env start \
    --scenario "scenario_name" \
    --envs_load_path outputs/gen_envs.jsonl \
    --port 8001

# Check if MCP server is running
awm env check --url http://localhost:8001/mcp

# Batch test all generated environments
awm env check_all --output outputs/gen_envs.jsonl
```

### Agent Demo

AWM includes a simple agent demo that connects to an MCP environment to solve tasks via multi-turn tool calling. Please start the environment and use [vLLM](https://github.com/vllm-project/vllm) to serve the model before running the agent.

```bash
# serve the model
vllm serve Snowflake/Arctic-AWM-4B --host 127.0.0.1 --port 8000

# start the environment, this will create an isolated folder outputs/servers/<timestamp> to save the environment related files such as initial.db, final.db, and etc.
awm env start --scenario e_commerce_33 --envs_load_path outputs/gen_envs.jsonl --port 8001

# run the agent
awm agent \
    --task "show me the top 10 most expensive products" \
    --mcp_url http://localhost:8001/mcp \
    --api_url http://localhost:8000/v1 \
    --model Snowflake/Arctic-AWM-4B
```

### Verification
AWM supports two types of verification:
1. `sql`, the recommended code-augmented LLM-as-a-Judge (requires LLM env vars, see Setup section)
2. `code`, purely code-based Judge

```bash
# launch an isolated environment and run the agent to finish the task of corresponding scenario
awm agent \
    --scenario e_commerce_33 \
    --task_id 0 \
    --api_url http://localhost:8000/v1 \
    --model Snowflake/Arctic-AWM-4B

# after interaction, the trajectory will be saved to outputs/agents/<timestamp>, we can verify it by
awm verify --input outputs/agents/<timestamp> --mode sql
```

## Benchmark
To align the evaluation with the trained tool-use protocol that requires agents to finish the task via unified MCP interface: `list_tools` and `call_tool`, we modified and adapated the evaluation harness of the following benchmarks: BFCLv3, τ²-bench-verified, and MCP-Universe. See more details at [mcp-adapted-bench](https://github.com/Raibows/mcp-adapted-bench). Remember to set the aforementioned env vars for the LLM agent API config before running the evaluation.

```bash
# install evaluation harness dependencies
git submodule update --init --recursive mcp-adapted-bench
uv sync --extra bench

# run evaluation, the tested agent LLM should be set by env vars as described in the Setup section
awm bench --mode bfcl --output_dir ./outputs/bfcl
awm bench --mode tau2 --output_dir ./outputs/tau2
awm bench --mode mcp_universe --output_dir ./outputs/mcp_universe
```


## Citation

If you find this work useful, please kindly cite:

```bibtex
@article{wang2026agentworldmodelinfinity,
      title={Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning}, 
      author={Zhaoyang Wang and Canwen Xu and Boyi Liu and Yite Wang and Siwei Han and Zhewei Yao and Huaxiu Yao and Yuxiong He},
      year={2026},
      eprint={2602.10090},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.10090}, 
}
```


