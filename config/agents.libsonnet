local models = import 'models.libsonnet';
local retrievers = import 'retrievers.libsonnet';

local A_RAVQA_QWen7B_Dummy = {
    class_name: 'RAG_VQA_Agent', 
    vlm_class: 'QWen2VLM',
    vlm_config: models.QWen2VLConfig,
    retriever_class: "DummyRetriever",
    retriever_config: retrievers.DummyRetriever,
};

local NORAG_QWen7B = {
    class_name: 'Plain_VQA_Agent', 
    vlm_class: 'QWen2VLM',
    vlm_config: models.QWen2VLConfig,
};

local NORAG_QWen72B = {
    class_name: 'Plain_VQA_Agent', 
    vlm_class: 'QWen2VLM',
    vlm_config: models.QWen2VL_72B_Config,
};

{
    A_RAVQA_QWen7B_Dummy: A_RAVQA_QWen7B_Dummy,
    NORAG_QWen7B: NORAG_QWen7B,
    NORAG_QWen72B: NORAG_QWen72B,
}