from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import os
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--file', required=True)
parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--json_dir', type=str)
args = parser.parse_args()

with open(args.file) as f:
    lines = f.readlines()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

out_file = args.file
if out_file.endswith('.tsv'):
    out_file = out_file[:-4]
if args.json_dir is not None:
    out_file = os.path.join(args.json_dir, os.path.split(out_file)[1])
out_file = out_file + '.json'

query_2_pos_docs = defaultdict(dict)
query_2_neg_docs = defaultdict(list)
queries = {}

for line in lines:
    qid, qry, pos_id, pos, neg_id, neg = line.strip().split('\t')

    qry, pos, neg = [
        tokenizer.encode(
            t, add_special_tokens=False, max_length=args.truncate, truncation=True)
        for t in (qry, pos, neg)
    ]

    pos_dict = {
        'pid': pos_id,
        'passage': pos,
    }

    neg_dict = {
        'pid': neg_id,
        'passage': neg,
    }

    query_dict = {
        'qid': qid,
        'query': qry,
    }
    if qid not in queries:
        queries[qid] = query_dict
    if qid not in query_2_pos_docs or pos_id not in query_2_pos_docs[qid]:
        query_2_pos_docs[qid][pos_id] = pos_dict
    query_2_neg_docs[qid].append(neg_dict)

with open(out_file, 'w') as f:
    for qid in queries.keys():
        item_set = {
            'qry': queries[qid],
            'pos': list(query_2_pos_docs[qid].values()),
            'neg': query_2_neg_docs[qid],
        }
        f.write(json.dumps(item_set) + '\n')