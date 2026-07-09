local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local ops = import "../operators.libsonnet";

{
    agent_config:  {
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_LoRA_Config {
            generation_config: {
                do_sample: true,
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                max_new_tokens: 64,
                // min_length: 1,
                // early_stopping: true,
            },
        },
        retriever_class: 'CacheDatasetRetriever', 
        retriever_config: retrievers.InfoseekNew_test_CacheRerankDatasetRetriever_ViT_L,
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt",
    op_config: [
        ops.Retrieve_OP {
            // kwargs: {
            //     ret_topk: 1,
            //     use_cache: true,
            //     cache_file: "cache/InfoseekNew/test-256/Retrieve/Retrieve_histories.json"
            // },
            // use_gpu: true,
        },
        ops.VLM_MBRRead_OP,
    ],
}