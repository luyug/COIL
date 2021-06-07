import argparse
import os
import pickle
from collections import defaultdict
from shutil import copyfile

import numpy as np
import torch


def pickle_file(file: str):
    with open(file, 'rb') as f:
        return pickle.load(f)


def rebuild_offsets(offset, query_ids):
    query_offsets = defaultdict(list)
    for tok_id in offset:
        start, n_tok = offset[tok_id]
        for off, qid in enumerate(query_ids[start: start+n_tok]):
            query_offsets[qid].append((tok_id, start+off))
    return dict(query_offsets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--as_torch', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.save_to, exist_ok=True)

    query_offset = pickle_file(os.path.join(args.dir, 'offsets.pkl'))
    query_reps = np.load(os.path.join(args.dir, 'tok_reps.npy'))
    query_pids = np.load(os.path.join(args.dir, 'tok_pids.npy'))
    query_cls_ids = np.load(os.path.join(args.dir, 'cls_pids.npy'))

    # exid_2_inid = {x: i for i, x in enumerate(query_cls_ids.tolist())}

    offset_by_query = rebuild_offsets(query_offset, query_pids.tolist())

    offsets = []
    curr = 0

    _index_order = []

    # reorder representations
    for qid in query_cls_ids.tolist():
        q_offset = []
        for tok_id, off in offset_by_query[qid]:
            q_offset.append((tok_id, curr))
            curr += 1
            _index_order.append(off)
        offsets.append(q_offset)

    assert len(_index_order) == len(query_reps)
    reps_by_query = query_reps[_index_order]

    # i/o
    if args.as_torch:
        torch.save(torch.tensor(reps_by_query), os.path.join(args.save_to, 'tok_reps.pt'))
    else:
        np.save(os.path.join(args.save_to, 'tok_reps'), reps_by_query)
    torch.save(offsets, os.path.join(args.save_to, 'offsets.pt'))

    del offsets
    del reps_by_query

    if args.as_torch:
        query_cls_reps = np.load(os.path.join(args.dir, 'cls_reps.npy'))
        torch.save(torch.tensor(query_cls_reps), os.path.join(args.save_to, 'cls_reps.pt'))
        torch.save(torch.tensor(query_cls_ids), os.path.join(args.save_to, 'cls_ex_ids.pt'))
    else:
        copyfile(os.path.join(args.dir, 'cls_reps.npy'), os.path.join(args.save_to, 'cls_reps.npy'))
        copyfile(os.path.join(args.dir, 'cls_pids.npy'), os.path.join(args.save_to, 'cls_ex_ids.npy'))


if __name__ == '__main__':
    main()