import pandas as pd

df_arag = pd.read_csv("outputs/202410-baselines/20241005-23-EVQA_ARAG_QWen2VL-72B/marked_answers.csv")
df_rag = pd.read_csv("outputs/202410-baselines/20241006-11-EVQA_ConvRAG_QWen2VL-7B-top100/marked_answers.csv")

print("Total length:", len(df_arag))
print("Both hit:", sum((df_arag['hit']) & (df_rag['hit'])))
print("ARAG hit & ConvRAG miss:", sum((df_arag['hit']) & (~df_rag['hit'])))
print("ARAG miss & ConvRAG hit:", sum((~df_arag['hit']) & (df_rag['hit'])))
print("Both miss:", sum((~df_arag['hit']) & (~df_rag['hit'])))