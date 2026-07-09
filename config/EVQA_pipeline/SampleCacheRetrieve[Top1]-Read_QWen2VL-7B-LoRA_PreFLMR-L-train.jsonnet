local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local ops = import "../operators.libsonnet";

{
    agent_config:  {
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_LoRA_Config {
            generation_config: {
                temperature: 0.7,
                max_new_tokens: 512,
                do_sample: true,
                top_p: 1.0,
                top_k: 0.0,
            },
        },
        retriever_class: 'CacheDatasetRetriever', 
        retriever_config: retrievers.EVQA_train_CacheDatasetRetriever_ViT_L,
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt",
    op_config: [
        ops.Retrieve_OP {
            kwargs: {
                ret_topk: 1,
                use_cache: false,
                // use_cache: true,
                // cache_file: "cache/EVQA_train256/retrieve/Retrieve_histories.json"
            }
        },
       ops.VLMRead_OP,
    ],
}