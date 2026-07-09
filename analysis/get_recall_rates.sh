
#!/bin/bash
# Infoseek PreFLMR-L retrieval
python analysis/compute_infoseek_recall.py \
    --input outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked \
    --retrieval_field retrieved_passage \
    --ks 1 2 3 5 10 20 50 100 150 200


# # Infoseek VLM+MLP
# python analysis/compute_infoseek_recall.py \
#     --input outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked \
#     --retrieval_field reranked_passage

# EVQA VLM-YesLogit
# python analysis/compute_infoseek_recall.py \
#     --input outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked \
#     --retrieval_field reranked_passage

# Infoseek VLM-YesLogit (x)
# python analysis/compute_infoseek_recall.py \
#     --input outputs/jinghong_chen/Infoseek-test_full-with-retrieval_post_reranked \
#     --retrieval_field reranked_passage

# Infoseek VLM+MLP
# python analysis/compute_infoseek_recall.py \
#     --input outputs/0jingbiao_mei/EVQA-testfull-with-retrieval-rerank7B-step4000_post_reranked \
#     --retrieval_field reranked_passage


# Infoseek VLM-YesLogit (x)
# python analysis/compute_infoseek_recall.py \
#     --input outputs/jinghong_chen/Infoseek-test_full-with-retrieval_post_reranked \
#     --retrieval_field reranked_passage
    