from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--input_file', required=True)
parser.add_argument('--save_to', required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
os.makedirs(args.save_to, exist_ok=True)

with open(args.input_file, 'r') as f:
    lines = f.readlines()


def encode_one_entry(line: str):
    eid, e = line.strip().split('\t')
    encoded = tokenizer.encode(
        e,
        add_special_tokens=False,
        max_length=args.truncate,
        truncation=True
    )
    return json.dumps({
        'pid': eid,
        'psg': encoded
    })


file_name = os.path.split(args.input_file)[1] + '.json'
with open(os.path.join(args.save_to, file_name), 'w') as jfile:
    for x in tqdm(lines):
        e = encode_one_entry(x)
        jfile.write(e + '\n')
