import pandas as pd
from tabulate import tabulate
import ast

# CSV_FILENAME="outputs/202410-baselines/20241006-11-EVQA_ConvRAG_QWen2VL-72B-top100/marked_answers.csv"
CSV_FILENAME="outputs/202410-baselines/20241005-01-EVQA_ConvRAG_QWen2VL-72B/marked_answers.csv"
# CSV_FILENAME="outputs/202410-baselines/20241005-23-EVQA_ARAG_QWen2VL-72B/marked_answers.csv"

def is_correct_retrieval(row):
    retrieved_doc = ast.literal_eval(row['retrieved_docs'])[0].rstrip('\n')
    gt_docs = ast.literal_eval(row['pos_item_contents'])
    if retrieved_doc in gt_docs:
        return True
    else:
        return False

def print_case(row):
    print(f"ID: {row['question_id']}")
    print(f"Image URL: {row['img_path']}")
    print(f"Question: {row['question']}")

    print(f"GT Doc: {row['pos_item_contents']}")
    print(f"GT Answer: {row['gold_answer']}")

    print(f"Query: {row['queries']}")
    print(f"Retrieved Doc: {row['retrieved_docs']}")
    print(f"Model Answer: {row['prediction']}")
    print(f"BEM Score: {row['score']}")
    print("="*50)

def tabulate_data(headers, cols, a, b, c, d):
    total = a+b+c+d
    table_data = ([
        (cols[0], a, b, f"{a+b} ({(a+b)/total*100:.1f}%)"),
        (cols[1], c, d, f"{c+d} ({(c+d)/total*100:.1f}%)"),
        ("Total", f"{a+c} ({(a+c)/total*100:.1f}%)", f"{b+d} ({(b+d)/total*100:.1f}%)", f"{total}")
    ])
    print(tabulate(table_data, headers=("", *headers, "Total")))


if __name__ == '__main__':
    df = pd.read_csv(CSV_FILENAME)
    df['correct_retrieval'] = df.apply(is_correct_retrieval, axis=1)

    ans_correct_retrieve_correct = len(df[(df['score']>0.1)&(df['correct_retrieval'])])
    ans_correct_retrieve_wrong = len(df[(df['score']>0.1)&(~df['correct_retrieval'])])
    ans_wrong_retrieve_correct = len(df[(df['score']<=0.1)&(df['correct_retrieval'])]) 
    ans_wrong_retrieve_wrong = len(df[(df['score']<=0.1)&(~df['correct_retrieval'])])

    tabulate_data(
        headers = ('Retrieve Correct', 'Retrieve Wrong'),
        cols = ('Answer Correct', 'Answer Wrong'),
        a=ans_correct_retrieve_correct,  b=ans_correct_retrieve_wrong, c=ans_wrong_retrieve_correct, d=ans_wrong_retrieve_wrong
    )

    # table_header = ['', 'Retrieve Correct', 'Retrieve Wrong', 'Total']
    # table_data = [
    #     ('Answer Correct', ans_correct_retrieve_correct, ans_correct_retrieve_wrong, ans_correct_retrieve_correct+ ans_correct_retrieve_wrong),
    #     ('Answer Wrong', ans_wrong_retrieve_correct, ans_wrong_retrieve_wrong, ans_wrong_retrieve_correct + ans_wrong_retrieve_wrong),
    #     ('Total', f"{ans_correct_retrieve_correct+ans_wrong_retrieve_correct} ({(ans_correct_retrieve_correct+ans_wrong_retrieve_correct)/len(df):.2f})", f"{ans_correct_retrieve_wrong+ans_wrong_retrieve_wrong} ({(ans_correct_retrieve_wrong+ans_wrong_retrieve_wrong)/len(df):.2f})", len(df)),
    # ]
    # print(tabulate(table_data, headers=table_header, tablefmt='grid'))

    # wrong_answer_wrong_retrieve_df = df[(df['score']<=0.1)&(~df['correct_retrieval'])]
    # for i, row in wrong_answer_wrong_retrieve_df[-5:].iterrows():
    #     print_case(row)

    # for i, row in df[[0]].iterrows():
    #     print_case(row)