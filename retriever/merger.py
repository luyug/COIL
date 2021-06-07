import os
import torch
import faiss
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List


def main():
    parser = ArgumentParser()
    parser.add_argument('--score_dir', required=True)
    parser.add_argument('--query_lookup', required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--num_query', type=int)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--marco_document', action='store_true')
    args = parser.parse_args()

    if args.num_query:
        rh = faiss.ResultHeap(args.num_query, args.depth)
    else:
        print("Inferring number of query from first input")
        rh = None

    partitions = os.listdir(args.score_dir)

    pbar = tqdm(partitions)

    for part_name in pbar:
        pbar.set_description_str(f'Processing {part_name}')
        scores, indices = torch.load(
            os.path.join(args.score_dir, part_name)
        )

        if rh is None:
            print(f'Initializing Heap. Assuming {scores.shape[0]} queries.')
            rh = faiss.ResultHeap(scores.shape[0], args.depth)
        rh.add_result(-scores.numpy(), indices.numpy())
    rh.finalize()

    corpus_scores, corpus_indices = (-rh.D).tolist(), rh.I.tolist()

    q_lookup: List[str] = torch.load(args.query_lookup).tolist()

    os.makedirs(os.path.split(args.save_ranking_to)[0], exist_ok=True)

    with open(args.save_ranking_to, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            _last = None
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                if args.marco_document:
                    _idx = f'D{idx}'
                else:
                    _idx = idx
                f.write(f'{qid}\t{_idx}\t{s}\n')


if __name__ == '__main__':
    main()