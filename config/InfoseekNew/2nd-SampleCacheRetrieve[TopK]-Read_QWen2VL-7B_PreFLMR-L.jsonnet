local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local ops = import "../operators.libsonnet";

{
    agent_config:  {
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_Config {
            generation_config: {
                temperature: 1.0,
                max_new_tokens: 64,
                do_sample: true,
                top_p: 0.95,
                top_k: 50,
            },
        },
        retriever_class: 'CacheDatasetRetriever', 
        retriever_config: retrievers.InfoseekNew_train_CacheDatasetRetriever_ViT_L {
            ds_path: "outputs/0jingbiao_mei/InfoseekNew-train64000-with-retrieval-ds_seed=2025",
        },
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt",
    op_config: [
        ops.Retrieve_OP {
            kwargs: {
                ret_topk: 5,
                use_cache: false,
                cache_file: "",
            },
            use_gpu: true,
        },
        ops.VLMRead_OP,
    ],
}