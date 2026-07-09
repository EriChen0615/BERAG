local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local ops = import "../operators.libsonnet";

{
    agent_config:  {
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_LoRA_Config,
        retriever_class: 'CacheDatasetRetriever', 
        retriever_config: retrievers.Infoseek_test_CacheRerankDatasetRetriever_ViT_L {
            retrieval_field: 'retrieved_passage',
        },
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt",
    op_config: [
        ops.Retrieve_OP {
            kwargs: {
                ret_topk: 1,
                use_cache: false,
                cache_file: "",
            },
            use_gpu: true,
        },
        ops.VLMRead_OP,
    ],
}