#!/bin/bash
# rsync -avh --progress cache/ "hpc:/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/cache/" 
# rsync -avh --progress outputs/jinghong_chen/EVQA-with-retrieval/ "hpc:/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/outputs/jinghong_chen/EVQA-with-retrieval/" 
rsync -avh --progress third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/rag1_answer/checkpoint-2500/ "hpc:/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/rag1_answer/checkpoint-2500/"