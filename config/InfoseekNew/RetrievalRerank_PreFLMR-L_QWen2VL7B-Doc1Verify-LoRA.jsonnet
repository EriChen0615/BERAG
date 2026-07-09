local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local rerankers = import "../rerankers.libsonnet";

{
    retriever_class: 'FLMRRetriever', 
    retriever_config: retrievers.InfoseekNew_PreFLMRRetriever_ViT_L {
        use_gpu: true
    },
    
# Consider adding more arguments here for example Ks topk etc,.
#  Ks=[1,5,10], topk=500, query_batch_size=32, compute_pseudo_recall=True
    query_and_evaluate_ds_kwargs:
    {
        Ks: [1, 3, 5, 10, 50, 100, 500],
        topk: 500,
        query_batch_size: 32,
        compute_pseudo_recall: true,
    },

    reranker_class: 'QWen2Reranker',
    reranker_config: rerankers.InfoseekNew_QWen2VL_7B_LoRA_Config,
}