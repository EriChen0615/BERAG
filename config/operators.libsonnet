local VLMRead_OP = {
    module_name: "answer_op",
    class_name: "ReadEvidenceAndAnswer",
    name: "VLMReadEvidence",
    kwargs: {
        prompt_template_file: "config/prompts/1003_conventional_rag.txt",
    },
};

local VLM_MBRRead_OP = {
    module_name: "answer_op",
    class_name: "ReadEvidenceAndMBRAnswerWithVLM",
    name: "VLM_MBRReadEvidence",
    kwargs: {
        prompt_template_file: "config/prompts/1003_conventional_rag.txt",
        mbr_n_samples: 8,
        mbr_metric: "bleu",
    },
};
local VLMNoRAGRead_OP = {
    module_name: "answer_op",
    class_name: "ReadWithVLM",
    name: "VLMNoRAGRead",
    kwargs: {
        prompt_template_file: "config/prompts/1101_norag_answer.txt",
    },
};

local VLMParallelReadRerank_OP = {
    module_name: "answer_op",
    class_name: "ParallelReadEvidenceAndRerankWithVLM",
    name: "VLMParallelReadEvidenceAndRerank",
    kwargs: {
        prompt_template_file: "config/prompts/1003_conventional_rag.txt",
        add_retriever_score: false,
        normalize_length: false,
    },
};

local Retrieve_OP = {
    module_name: "retrieve_op",
    class_name: "Retrieve",
    name: "Retrieve",
    kwargs: {
        ret_topk: 1,
    }
};

local Rewrite_QR_OP = {
    module_name: "query_rewriting_op",
    class_name: "QueryRewrite",
    name: "Rewrite_QR",
    kwargs: {
        rewrite_method: "question_rewriting",
        prompt_template_file: "config/prompts/1016_question_rewriting.txt"
    },
};

local Rewrite_QRwCoT_OP = {
    module_name: "query_rewriting_op",
    class_name: "QueryRewrite",
    name: "Rewrite_QRwCoT",
    kwargs: {
        rewrite_method: "question_rewriting_with_cot",
        prompt_template_file: "config/prompts/1016_question_rewriting_with_cot.txt"
    },
};

local Rewrite_QRwDocGen_OP = {
    module_name: "query_rewriting_op",
    class_name: "QueryRewrite",
    name: "Rewrite_QRwDocGen",
    kwargs: {
        rewrite_method: "question_expansion_with_doc_gen",
        prompt_template_file: "config/prompts/1021_question_expansion_with_doc_gen.txt"
    },
};

local Verify_EntityMatch_OP = {
    module_name: "verifydoc_op",
    class_name: "VerifyDoc",
    name: "VerifyDoc_EntityMatch",
    kwargs: {
        prompt_template_file: "config/prompts/1023_verifydoc_entitymatch.txt",
    },
};

local StaticPlan_OP = {
    module_name: "plan_op",
    name: "StaticPlan",
};

{
    VLMRead_OP: VLMRead_OP,
    VLM_MBRRead_OP: VLM_MBRRead_OP,
    VLMNoRAGRead_OP: VLMNoRAGRead_OP,
    Retrieve_OP: Retrieve_OP,
    Rewrite_QR_OP: Rewrite_QR_OP,
    Rewrite_QRwCoT_OP: Rewrite_QRwCoT_OP,
    Rewrite_QRwDocGen_OP: Rewrite_QRwDocGen_OP,
    VLMParallelReadRerank_OP: VLMParallelReadRerank_OP,
    Verify_EntityMatch_OP: Verify_EntityMatch_OP,
}