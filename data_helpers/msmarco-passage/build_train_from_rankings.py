from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import os
from collections import defaultdict
import datasets
import random
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--rank_file', required=True)
parser.add_argument('--query_split_file', type=str)
parser.add_argument('--qrel_file', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--sample_from_top', type=int, default=1000)
parser.add_argument('--n_sample', type=int, default=100)
parser.add_argument('--random', action='store_true')
parser.add_argument('--json_dir', type=str, required=True)
args = parser.parse_args()


def read_qrel():
    import csv
    qrel = {}
    with open(args.qrel_file, encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel


qrel = read_qrel()
qmap = {}
with open(args.query_split_file) as f:
    for l in f:
        qid, qry = l.strip().split('\t')
        qmap[qid] = qry

top_rankings = defaultdict(list)
with open(args.rank_file) as f:
    for l in f:
        qid, pid, rank = l.split()
        if pid in qrel[qid]:
            continue
        # append passage if & only if it is not juddged relevant but ranks high
        top_rankings[qid].append(pid)


collection = args.collectoin
collection = datasets.load_dataset(
    'csv',
    data_files=collection,
    column_names=['pid', 'psg'],
    delimiter='\t',
    ignore_verifications=True,
)['train']

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

out_file = args.rank_file
if out_file.endswith('.tsv') or out_file.endswith('.txt'):
    out_file = out_file[:-4]
out_file = os.path.join(args.json_dir, os.path.split(out_file)[1])

out_file = out_file + '.json'

queries = top_rankings.keys()
with open(out_file, 'w') as f:
    for qid in queries:
        for pos in qrel[qid]:
            pos_tokenized = tokenizer.encode(
                collection[int(pos)]['psg'],
                add_special_tokens=False,
                max_length=args.truncate,
                truncation=True
            )
            pos = {
                'passage': pos_tokenized,
                'pid': pos,
            }
            # pick from top of the full initial ranking
            negs = top_rankings[qid][:args.sample_from_top]
            # shuffle if random flag is on
            if args.random:
                random.shuffle(negs)
            # pick n samples
            negs = negs[:args.n_sample]

            neg_encoded = []
            for neg in negs:
                encoded_neg = tokenizer.encode(
                    collection[int(neg)]['psg'],
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True
                )
                neg_encoded.append({
                    'passage': encoded_neg,
                    'pid': neg,
                })

            query_dict = {
                'qid': qid,
                'query': tokenizer.encode(
                    qmap[qid],
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True),
            }

            item_set = {
                'qry': query_dict,
                'pos': [pos],
                'neg': neg_encoded,
            }
            f.write(json.dumps(item_set) + '\n')