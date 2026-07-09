local DummyRetriever = {
    'always_return': "Giant hogweed is a member of the carrot family and its resemblance to Queen Anne’s lace caused it to become a garden ornamental. It spreads easily and can establish along roadsides, ditches, and streams. Giant hogweed has a thick bright green stem (3-8 cm in diameter) with dark reddish-purple spots and coarse white hairs at the base of the leaf stock. The plant can be 2-5.5 m tall with broad leaves that are deeply-lobed and serrated. From late spring to mid-summer, giant hogweed produces a large upside-down umbrella-shaped head, up to 80 cm across, with clusters of tiny white flowers. Giant hogweed has a phototoxic sap that, when exposed to light, can cause severe burns on human skin. Removing hogweed can be dangerous because of this sap; it should also not be burned or composted for this reason. The easiest way to remove giant hogweed is to pull it when it is still very young and small and store all plant components in sealed black garbage bags until the plant is dried and seeds are no longer viable. Do not plant giant hogweed in gardens and report any sightings."
};

# EVQA
local EVQA_OrcaleRetriever = {
    ds_name: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
    ds_subset: "EVQA_data",
    use_split: "test",
};

local EVQA_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR',
    passage_subset: 'EVQA_passages',
    use_split: 'test',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    query_maxlen: 32,
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/EVQA",
        index_experiment_name: "",
        index_name: "EVQA_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
};

local EVQA_valid_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR',
    passage_subset: 'EVQA_passages',
    use_split: 'valid',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    use_gpu: true,
    query_maxlen: 32,
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/EVQA_valid",
        index_experiment_name: "",
        index_name: "EVQA_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
};

local EVQA_train_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR',
    passage_subset: 'EVQA_passages',
    use_split: 'train',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    use_gpu: true,
    query_maxlen: 32,
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/EVQA_train",
        index_experiment_name: "",
        index_name: "EVQA_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
};

local EVQA_train_CacheDatasetRetriever_ViT_L = {
    ds_path: "outputs/jinghong_chen/EVQA-with-retrieval",
    use_split: 'train',
    retrieval_field: 'retrieved_passage',
    passage_dataset_name: 'EVQA',
    ret_topk: 5,
};

local EVQA_test_CacheDatasetRetriever_ViT_L = {
    ds_path: "outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked",
    use_split: 'test',
    retrieval_field: 'reranked_passage',
    passage_dataset_name: 'EVQA',
    ret_topk: 5,
};

local EVQA_SE_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR',
    passage_subset: 'EVQA_passages',
    use_split: 'test',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/EVQA",
        index_experiment_name: "",
        index_name: "EVQA_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
    num_doc_to_return: 5,
    preview_max_wordcount: 20
};

# OKVQA
local OKVQA_OrcaleRetriever = {
    ds_name: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
    ds_subset: "OKVQA_data",
    use_split: "test",
};

local OKVQA_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR',
    passage_subset: 'OKVQA_passages',
    use_split: 'valid',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/OKVQA",
        index_experiment_name: "",
        index_name: "OKVQA_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
    use_gpu: true,
};

local OKVQA_train_CacheDatasetRetriever_ViT_L = {
    ds_path: "outputs/jinghong_chen/OKVQA-with-retrieval",
    use_split: 'train',
    retrieval_field: 'retrieved_passage',
    passage_dataset_name: 'OKVQA',
    ret_topk: 5,
};

local OKVQA_valid_CacheRerankDatasetRetriever_ViT_L = {
    ds_path: "outputs/jinghong_chen/OKVQA-valid-with-retrieval_post_reranked",
    use_split: 'valid',
    retrieval_field: 'reranked_passage',
    passage_dataset_name: 'OKVQA',
    ret_topk: 5,
};

# Infoseek
local Infoseek_OrcaleRetriever = {
    ds_name: "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
    ds_subset: "Infoseek_data",
    use_split: "test",
};

local Infoseek_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR',
    passage_subset: 'Infoseek_passages',
    use_split: 'test',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/Infoseek",
        index_experiment_name: "",
        index_name: "Infoseek_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
    use_gpu: true,
};

local InfoseekNew_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'Jingbiao/aravqa',
    passage_subset: 'Infoseek_passages',
    use_split: 'train',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/InfoseekNew",
        index_experiment_name: "",
        index_name: "InfoseekNew_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
    use_gpu: true,
};

local InfoseekNew_FullPassage_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'Jingbiao/aravqa',
    passage_subset: 'InfoseekFull_passages',
    use_split: 'valid',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/InfoseekNew",
        index_experiment_name: "",
        index_name: "InfoseekNew_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
    use_gpu: true,
};

local InfoseekNew_OracleRetriever = {
    ds_name: "Jingbiao/aravqa",
    ds_subset:  "Infoseek_data",
    use_split: "valid_m2kr"
};

local InfoseekNew_FullPassage_OracleRetriever = {
    ds_name: "Jingbiao/aravqa",
    ds_subset:  "Infoseek_data",
    use_split: "valid_m2kr",
    map_from_passage_set: true,
    passage_ds: "InfoseekNew_FullPassage",
    passage_split: "valid",
};

// local InfoseekNew_CacheRetriever_PreFLMR_L = {
//     ds_path: "outputs/jinghong_chen/InfoseekNew-valid-with-retrieval",
//     retrieval_field: "retrieved_passage",
//     passage_dataset_name: "InfoseekNew",
//     ret_topk: 5,
//     use_split: "valid",
// };

local Infoseek_test_CacheRerankDatasetRetriever_ViT_L = {
    ds_path: "outputs/jinghong_chen/Infoseek-test256-with-retrieval_post_reranked",
    use_split: 'test',
    retrieval_field: 'reranked_passage',
    passage_dataset_name: 'Infoseek',
    ret_topk: 5,
};

local InfoseekNew_test_CacheRerankDatasetRetriever_ViT_L = {
    ds_path: "outputs/0jingbiao_mei/InfoseekNew-test256-with-retrieval_post_reranked",
    use_split: 'valid',
    retrieval_field: 'reranked_passage',
    passage_dataset_name: 'InfoseekNew_FullPassage',
    ret_topk: 5,
};

local InfoseekNew_testfull_CacheRerankDatasetRetriever_ViT_L = {
    ds_path: "outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS2B_post_reranked",
    use_split: 'valid',
    retrieval_field: 'reranked_passage',
    passage_dataset_name: 'InfoseekNew_FullPassage',
    ret_topk: 5,
};

local InfoseekNew_testfull_CacheRerankDatasetRetriever_ViT_L_7BRerank = {
    ds_path: "outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked",
    use_split: 'valid',
    retrieval_field: 'reranked_passage',
    passage_dataset_name: 'InfoseekNew_FullPassage',
    ret_topk: 5,
};


local Infoseek_testfull_CacheRerankDatasetRetriever_ViT_L = {
    ds_path: "outputs/jinghong_chen/Infoseek-test_full-with-retrieval_post_reranked",
    use_split: 'test',
    retrieval_field: 'reranked_passage',
    passage_dataset_name: 'Infoseek',
    ret_topk: 5,
};

local Infoseek_train_CacheRerankDatasetRetriever_ViT_L = {
    ds_path: "outputs/jinghong_chen/Infoseek-train64000-with-retrieval",
    use_split: 'train',
    retrieval_field: 'retrieved_passage',
    passage_dataset_name: 'Infoseek',
    ret_topk: 5,
};

local InfoseekNew_train_CacheRerankDatasetRetriever_ViT_L = {
    ds_path: "outputs/0jingbiao_mei/InfoseekNew-train64000-with-retrieval",
    use_split: 'train',
    retrieval_field: 'reranked_passage',
    passage_dataset_name: 'InfoseekNew_FullPassage',
    ret_topk: 5,
};

local InfoseekNew_train_CacheDatasetRetriever_ViT_L = {
    ds_path: "outputs/0jingbiao_mei/InfoseekNew-train64000-with-retrieval",
    use_split: 'train',
    retrieval_field: 'retrieved_passage',
    passage_dataset_name: 'InfoseekNew_FullPassage',
    ret_topk: 5,
};

# EVQA2hop
local EVQA2hop_OrcaleRetriever = {
    ds_name: "Jingbiao/aravqa",
    ds_subset: "EVQA2hop_1013_data",
    use_split: "test",
};

local EVQA2hop_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'Jingbiao/aravqa',
    passage_subset: 'EVQA2hop_1013_passages',
    use_split: 'test',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/EVQA2hop",
        index_experiment_name: "",
        index_name: "EVQA2hop_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
};

local EVQA2hop_SE_PreFLMRRetriever_ViT_L = {
    ckpt_path: "LinWeizheDragon/PreFLMR_ViT-L",
    image_processor_name: "openai/clip-vit-large-patch14",
    passage_ds: 'Jingbiao/aravqa',
    passage_subset: 'EVQA2hop_1013_passages',
    use_split: 'test',
    instruction: "With the provided image, gather documents that offer a solution to the question:",
    searcher_kwargs: {
        index_root_path: "../vqa_data/Index/EVQA2hop",
        index_experiment_name: "",
        index_name: "EVQA2hop_PreFLMR_ViT-L",
        nbits: 8,
        use_gpu: true,
    },
    num_doc_to_return: 5,
    preview_max_wordcount: 20
};

{
    DummyRetriever: DummyRetriever,
    EVQA_PreFLMRRetriever_ViT_L: EVQA_PreFLMRRetriever_ViT_L,
    EVQA_valid_PreFLMRRetriever_ViT_L: EVQA_valid_PreFLMRRetriever_ViT_L,
    EVQA_train_PreFLMRRetriever_ViT_L: EVQA_train_PreFLMRRetriever_ViT_L,
    EVQA_train_CacheDatasetRetriever_ViT_L: EVQA_train_CacheDatasetRetriever_ViT_L,
    EVQA_test_CacheDatasetRetriever_ViT_L: EVQA_test_CacheDatasetRetriever_ViT_L, 
    EVQA_OrcaleRetriever: EVQA_OrcaleRetriever,
    OKVQA_PreFLMRRetriever_ViT_L: OKVQA_PreFLMRRetriever_ViT_L,
    OKVQA_OrcaleRetriever: OKVQA_OrcaleRetriever,
    OKVQA_train_CacheDatasetRetriever_ViT_L: OKVQA_train_CacheDatasetRetriever_ViT_L,
    OKVQA_valid_CacheRerankDatasetRetriever_ViT_L: OKVQA_valid_CacheRerankDatasetRetriever_ViT_L,
    Infoseek_PreFLMRRetriever_ViT_L: Infoseek_PreFLMRRetriever_ViT_L,
    InfoseekNew_PreFLMRRetriever_ViT_L: InfoseekNew_PreFLMRRetriever_ViT_L,
    InfoseekNew_FullPassage_PreFLMRRetriever_ViT_L: InfoseekNew_FullPassage_PreFLMRRetriever_ViT_L,
    InfoseekNew_train_CacheRerankDatasetRetriever_ViT_L: InfoseekNew_train_CacheRerankDatasetRetriever_ViT_L,
    InfoseekNew_train_CacheDatasetRetriever_ViT_L: InfoseekNew_train_CacheDatasetRetriever_ViT_L,
    InfoseekNew_OracleRetriever: InfoseekNew_OracleRetriever,
    InfoseekNew_FullPassage_OracleRetriever: InfoseekNew_FullPassage_OracleRetriever,
    InfoseekNew_test_CacheRerankDatasetRetriever_ViT_L: InfoseekNew_test_CacheRerankDatasetRetriever_ViT_L,
    InfoseekNew_testfull_CacheRerankDatasetRetriever_ViT_L: InfoseekNew_testfull_CacheRerankDatasetRetriever_ViT_L,
    InfoseekNew_testfull_CacheRerankDatasetRetriever_ViT_L_7BRerank: InfoseekNew_testfull_CacheRerankDatasetRetriever_ViT_L_7BRerank,
    Infoseek_OrcaleRetriever: Infoseek_OrcaleRetriever,
    Infoseek_test_CacheRerankDatasetRetriever_ViT_L: Infoseek_test_CacheRerankDatasetRetriever_ViT_L,
    Infoseek_testfull_CacheRerankDatasetRetriever_ViT_L: Infoseek_testfull_CacheRerankDatasetRetriever_ViT_L,
    Infoseek_train_CacheRerankDatasetRetriever_ViT_L: Infoseek_train_CacheRerankDatasetRetriever_ViT_L,
    EVQA2hop_PreFLMRRetriever_ViT_L: EVQA2hop_PreFLMRRetriever_ViT_L,
    EVQA2hop_OrcaleRetriever: EVQA2hop_OrcaleRetriever,
}