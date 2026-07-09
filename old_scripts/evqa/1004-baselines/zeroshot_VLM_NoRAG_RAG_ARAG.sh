#!/bin/bash
#NO-RAG
# sbatch scripts/evqa/run_NoRAG_QWen2VL-7B.sh
# sbatch scripts/evqa/run_NoRAG_QWen2VL-72B.sh

#conventional RAG
# sbatch scripts/evqa/run_RAG_QWen2VL-7B_PreFLMR-L.sh
# sbatch scripts/evqa/run_RAG_QWen2VL-72B_PreFLMR-L.sh

#ARAG
# sbatch scripts/evqa/run_ARAG_QWen2VL-7B_PreFLMR-L.sh
# sbatch scripts/evqa/run_ARAG_QWen2VL-72B_PreFLMR-L.sh
