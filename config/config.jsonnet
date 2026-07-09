local agents = import "agents.libsonnet";

{
    agent_config: agents.NORAG_QWen7B,
    initial_prompt: "config/prompts/1003_norag.txt"
}