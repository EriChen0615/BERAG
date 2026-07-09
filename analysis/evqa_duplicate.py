from datasets import load_from_disk
import sys
sys.path.append('./src')
from vqa_datasets import load_passages
ds = load_from_disk('outputs/0jingbiao_mei/EVQA-testfull-with-retrieval-rerank7B-step4000_post_reranked')
# field = 'retrieved_passage' 
field = 'reranked_passage'
_, pid_to_content_map = load_passages('EVQA', split='test')

topk = 5
dup_count = 0
for q in range(0, len(ds)):
    C = set()
    has_dup = False
    for r in range(0, topk):
        pid = ds[q][field][r]['passage_id']
        rp = pid_to_content_map[pid]
        if rp in C:
            print("DUP")
            print(str(q)+' '+str(r)+' | '+pid+'')
            # print(str(q)+' '+str(r)+' | '+pid+' | '+rp)
            if not has_dup:
                dup_count += 1
                has_dup = True
        C.add(rp)

print(f"TopK={topk}")
print(f"Examples containing duplicates in field `{field}` @Top{topk} = {dup_count} ({dup_count/len(ds)*100}%)")