import pandas as pd
from tabulate import tabulate

NoRAG_CSV="outputs/202410-baselines/20241004-14-EVQA_NoRAG_QWen2VL-72B/marked_answers.csv"
ConvRAG_CSV="outputs/202410-baselines/20241005-01-EVQA_ConvRAG_QWen2VL-72B/marked_answers.csv"

def tabulate_data(headers, cols, a, b, c, d):
    total = a+b+c+d
    table_data = ([
        (cols[0], a, b, f"{a+b} ({(a+b)/total*100:.1f}%)"),
        (cols[1], c, d, f"{c+d} ({(c+d)/total*100:.1f}%)"),
        ("Total", f"{a+c} ({(a+c)/total*100:.1f}%)", f"{b+d} ({(b+d)/total*100:.1f}%)", f"{total}")
    ])
    print(tabulate(table_data, headers=("", *headers, "Total")))


if __name__ == '__main__':
    norag_df = pd.read_csv(NoRAG_CSV)
    convrag_df = pd.read_csv(ConvRAG_CSV)

    both_correct = ((norag_df['score']>0.1) & (convrag_df['score']>0.1)).sum()
    rag_correct_norag_incorrect = ((norag_df['score']<0.1) & (convrag_df['score']>0.1)).sum()
    rag_incorrect_norag_correct = ((norag_df['score']>0.1) & (convrag_df['score']<0.1)).sum()
    both_incorrect = ((norag_df['score']<0.1) & (convrag_df['score']<0.1)).sum()

    tabulate_data(
        headers=("Correct w/o retrieval", "Wrong w/o retrieval"),
        cols=("Correct w/ retrieval", "Wrong w/ retrieval"),
        a=both_correct, b=rag_correct_norag_incorrect, c=rag_incorrect_norag_correct, d=both_incorrect
    )

    # table_data = [
    #     ("", "Correct w/o retrieval", "Wrong w/o retrieval", "Total"),
    #     ("Correct w/ retrieval", both_correct, rag_correct_norag_incorrect, both_correct+rag_correct_norag_incorrect),
    #     ("Wrong w/ retrieval", rag_incorrect_norag_correct, both_incorrect, rag_incorrect_norag_correct+both_incorrect),
    #     ("", both_correct + rag_incorrect_norag_correct, both_incorrect, rag_correct_norag_incorrect)
    # ]


