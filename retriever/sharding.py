import argparse
import os
import pickle
import numpy as np
import torch


def pickle_file(file: str):
    with open(file, 'rb') as f:
        return pickle.load(f)


def shard_splits(args):
    all_splits = os.listdir(args.dir)
    all_splits = sorted(all_splits, key=lambda x: int(x[5:]))
    n_splits = len(all_splits)
    if n_splits % args.n_shards == 0:
        shard_n_splits = int(n_splits / args.n_shards)
    else:
        shard_n_splits = int(n_splits / args.n_shards) + 1

    offset = args.shard_id * shard_n_splits
    splits = all_splits[offset: offset + shard_n_splits]

    return list(map(lambda x: os.path.join(args.dir, x), splits))


def load_ivl_one_split(split_dir: str):
    split_offset = pickle_file(os.path.join(split_dir, 'offsets.pkl'))
    split_reps = np.load(os.path.join(split_dir, 'tok_reps.npy'))
    split_pids = np.load(os.path.join(split_dir, 'tok_pids.npy'))

    return split_offset, split_reps, split_pids


def load_cls_one_split(split_dir: str):
    split_cls_reps = np.load(os.path.join(split_dir, 'cls_reps.npy'))
    split_cls_ids = np.load(os.path.join(split_dir, 'cls_pids.npy'))

    return split_cls_reps, split_cls_ids


def build_scatter_map(eid_list, eid_2_sid_map):
    _shard_scatter_map = [eid_list[0]]
    ivl_scatter_map = [0]

    for eid in eid_list[1:]:
        if eid != _shard_scatter_map[-1]:
            _shard_scatter_map.append(eid)
        ivl_scatter_map.append(len(_shard_scatter_map) - 1)

    ivl_scatter_map = np.array(ivl_scatter_map)

    shard_scatter_map = list(map(lambda x: eid_2_sid_map[x], _shard_scatter_map))
    shard_scatter_map = np.array(shard_scatter_map)

    return shard_scatter_map, ivl_scatter_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--n_shards', required=True, type=int)
    parser.add_argument('--shard_id', required=True, type=int)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--use_torch', action='store_true')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_to, f'shard_{args.shard_id:02d}'), exist_ok=True)

    splits_in_shard = shard_splits(args)

    # build cls shard first
    cls_reps = None
    cls_ids = None
    for sp in splits_in_shard:
        split_cls_reps, split_cls_ids = load_cls_one_split(sp)
        if cls_reps is None:
            cls_reps = split_cls_reps
            cls_ids = split_cls_ids
        else:
            cls_reps = np.concatenate((cls_reps, split_cls_reps), axis=0)
            cls_ids = np.concatenate((cls_ids, split_cls_ids), axis=0)

    if args.use_torch:
        torch.save(torch.tensor(cls_reps), os.path.join(args.save_to, f'shard_{args.shard_id:02d}', 'cls_reps.pt'))
        torch.save(torch.tensor(cls_ids), os.path.join(args.save_to, f'shard_{args.shard_id:02d}', 'cls_ex_ids.pt'))

    else:
        np.save(os.path.join(args.save_to, f'shard_{args.shard_id:02d}', 'cls_reps'), cls_reps)
        np.save(os.path.join(args.save_to, f'shard_{args.shard_id:02d}', 'cls_ex_ids'), cls_ids)

    del cls_reps

    cls_ids = cls_ids.tolist()
    ex_id_to_sid = dict(((x, i) for i, x in enumerate(cls_ids)))

    tok_id_2_rep = {}
    tok_ids_2_ex_id = {}
    for sp in splits_in_shard:
        split_offset, split_reps, split_pids = load_ivl_one_split(sp)
        for tok_id in split_offset:
            off, ivl_size = split_offset[tok_id]
            if tok_id not in tok_id_2_rep:
                tok_id_2_rep[tok_id] = split_reps[off: off+ivl_size]
                tok_ids_2_ex_id[tok_id] = split_pids[off: off+ivl_size]
            else:
                tok_id_2_rep[tok_id] = np.concatenate(
                    (tok_id_2_rep[tok_id], split_reps[off: off + ivl_size]), axis=0)
                tok_ids_2_ex_id[tok_id] = np.concatenate(
                    (tok_ids_2_ex_id[tok_id], split_pids[off: off + ivl_size]), axis=0)

    if args.use_torch:
        for tok_id in tok_id_2_rep:
            tok_id_2_rep[tok_id] = torch.tensor(tok_id_2_rep[tok_id])

    torch.save(tok_id_2_rep, os.path.join(args.save_to, f'shard_{args.shard_id:02d}', 'tok_reps.pt'))
    del tok_id_2_rep

    all_ivl_scatter_maps = {}
    all_shard_scatter_maps = {}

    for tok_id, tok_map in tok_ids_2_ex_id.items():
        shard_scatter_map, ivl_scatter_map = build_scatter_map(
            tok_map,
            ex_id_to_sid
        )

        if args.use_torch:
            shard_scatter_map = torch.tensor(shard_scatter_map)
            ivl_scatter_map = torch.tensor(ivl_scatter_map)

        all_ivl_scatter_maps[tok_id] = ivl_scatter_map
        all_shard_scatter_maps[tok_id] = shard_scatter_map

    torch.save(all_ivl_scatter_maps,
               os.path.join(args.save_to, f'shard_{args.shard_id:02d}', 'ivl_scatter_maps.pt'))
    torch.save(all_shard_scatter_maps,
               os.path.join(args.save_to, f'shard_{args.shard_id:02d}', 'shard_scatter_maps.pt'))


if __name__ == '__main__':
    main()