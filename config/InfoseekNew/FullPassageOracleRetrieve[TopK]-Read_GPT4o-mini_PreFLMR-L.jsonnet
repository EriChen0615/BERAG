local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local ops = import "../operators.libsonnet";

{
    agent_config:  {
        vlm_class: 'OpenAI_VLM',
        vlm_config: models.GPT4omini_Config,
        retriever_class: 'OracleRetriever', 
        retriever_config: retrievers.InfoseekNew_FullPassage_OracleRetriever,
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt",
    op_config: [
        ops.Retrieve_OP {
            kwargs: {
                ret_topk: 1,
            },
            use_gpu: true,
        },
        ops.VLMRead_OP,
    ],
}