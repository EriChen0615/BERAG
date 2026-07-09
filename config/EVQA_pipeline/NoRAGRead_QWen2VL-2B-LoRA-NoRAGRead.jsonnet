local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local ops = import "../operators.libsonnet";

{
    agent_config:  {
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_2B_LoRA_NoRAGRead_Config,
    },
    initial_prompt: "config/prompts/1101_norag_answer.txt",
    op_config: [
        ops.VLMNoRAGRead_OP,
    ],
}