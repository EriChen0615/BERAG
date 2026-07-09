import pandas as pd
INPUT_CSV="outputs/0922/EVQA-AttnRerank/SFT-Rerank/attn_rerank_results-mode=sum-n=512.csv"

def hit_at_1(row):
    return row['gt_doc_idx'] == 0

df = pd.read_csv(INPUT_CSV)
df['hit_at_1'] = df.apply(hit_at_1, axis=1)
print(df['hit_at_1'].mean())