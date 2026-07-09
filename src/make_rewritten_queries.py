import pandas as pd
import re
import argparse
import ast

def make_rewritten_query(row, method):
    rewritten_queries = []
    response = row['model_response']
    if method == 'question_rewriting':
        # Format: 
        # \n- Rewrite n: <rewrite>
        rewrites = re.findall(r'- Rewrite \d+: (.+)', response)
        rewritten_queries.extend(rewrites)
        row['rewritten_query'] = rewritten_queries
    elif method == 'question_rewritting_with_cot':
        # Format:
        # \n- Rewrite n: <rewrite>
        rewrites = re.findall(r'- Rewrite \d+: (.+)', response)
        rewritten_queries.extend(rewrites)
        row['rewritten_query'] = rewritten_queries
    elif method == 'question_expansion_with_answer':
        # Format
        # \n- Answer n: <answer>
        answers = re.findall(r'- Answer \d+: (.+)', response)
        rewritten_queries.extend(answers)
        row['rewritten_query'] = [q+' '+ans for q, ans in zip(row['rewritten_query'] + answers)]
    elif method == 'question+GTdoc':
        gt_docs = ast.literal_eval(row['pos_item_contents'])
        row['rewritten_query'] = [row['question'] + ' ' + gt_doc[:4096] for gt_doc in gt_docs] # cap at 4096 
    elif method == 'GTdoc':
        gt_docs = ast.literal_eval(row['pos_item_contents'])
        row['rewritten_query'] = [gt_doc[:4096] for gt_doc in gt_docs] # cap at 4096 
    elif method == 'question+GTentity':
        gt_doc_ids = ast.literal_eval(row['pos_item_ids'])
        row['rewritten_query'] = [row['question'] + gt_doc_id.split('_')[1] for gt_doc_id in gt_doc_ids]
    elif method == 'GTentity':
        gt_doc_ids = ast.literal_eval(row['pos_item_ids'])
        row['rewritten_query'] = [gt_doc_id.split('_')[1] for gt_doc_id in gt_doc_ids]
    elif method == 'question+PseudoDocGen':
        row['rewritten_query'] = [row['question'] + response]
    elif method == 'PseudoDocGen':
        row['rewritten_query'] = [response]
    elif method == 'question+PseudoDocGen-noformat':
        title = re.findall(r'- TITLE: (.+)', response)
        content = re.findall(r'- CONTENT: (.+)', response)
        if len(title) == 0 or len(content) == 0:
            row['rewritten_query'] = [row['question'] + response]
        else:
            title = title[0]
            content = content[0]
            row['rewritten_query'] = [row['question'] + ' ' + title + ' ' +content]
    elif method == 'question+EntityGen':
        title = re.findall(r'- TITLE: (.+)', response)
        content = re.findall(r'- CONTENT: (.+)', response)
        if len(title) == 0 or len(content) == 0:
            row['rewritten_query'] = [row['question'] + response]
        else:
            title = title[0]
            content = content[0]
            row['rewritten_query'] = [row['question'] + ' ' + title]
    else:
        raise NotImplementedError("make_rewritten_query")
    return row
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--rewrite_method", type=str, required=True)

    args = parser.parse_args()

    df_in = pd.read_csv(args.input_csv)
    # Apply make_rewritten_query to each row

    df_in = df_in.apply(lambda row: make_rewritten_query(row, args.rewrite_method), axis=1)

    # Explode the rewritten_query column (if the list has multiple items)
    df_exploded = df_in.explode('rewritten_query')

    # Save the result to the output CSV
    df_exploded.to_csv(args.output_csv, index=False)
    print("Query file saved to", args.output_csv)


