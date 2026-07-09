local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";

{
    vlm_class: 'QWen2VLM',
    vlm_config: models.QWen2VL_72B_8bitGPTQ_Config,
}