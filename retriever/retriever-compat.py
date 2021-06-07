import argparse
import torch
import os
from tqdm import tqdm
from torch_scatter import segment_max_coo as scatter_max


def dict_2_float(dd):
    for k in dd:
        dd[k] = dd[k].float()

    return dd


def build_full_tok_rep(dd):
    new_offsets = {}
    curr = 0
    reps = []
    for k in dd:
        rep = dd[k]
        reps.append(rep)

        new_offsets[k] = (curr, curr+len(rep))
        curr += len(rep)

    reps = torch.cat(reps).float()
    return reps, new_offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--doc_shard', required=True)
    parser.add_argument('--top', type=int, default=1000)
    parser.add_argument('--save_to', required=True)
    args = parser.parse_args()

    all_ivl_scatter_maps = torch.load(os.path.join(args.doc_shard, 'ivl_scatter_maps.pt'))
    all_shard_scatter_maps = torch.load(os.path.join(args.doc_shard, 'shard_scatter_maps.pt'))
    tok_id_2_reps = torch.load(os.path.join(args.doc_shard, 'tok_reps.pt'))
    doc_cls_reps = torch.load(os.path.join(args.doc_shard, 'cls_reps.pt')).float()
    cls_ex_ids = torch.load(os.path.join(args.doc_shard, 'cls_ex_ids.pt'))
    tok_id_2_reps = dict_2_float(tok_id_2_reps)

    print('Search index loaded', flush=True)

    query_tok_reps = torch.load(os.path.join(args.query, 'tok_reps.pt')).float()
    all_query_offsets = torch.load(os.path.join(args.query, 'offsets.pt'))
    query_cls_reps = torch.load(os.path.join(args.query, 'cls_reps.pt')).float()

    print('Query representations loaded', flush=True)

    all_query_match_scores = []
    all_query_inids = []

    shard_name = os.path.split(args.doc_shard)[1]

    for q_iid, q_offsets in enumerate(tqdm(all_query_offsets, desc=shard_name)):
        match_scores = torch.matmul(doc_cls_reps, query_cls_reps[q_iid])  # D
        for q_tok_id, q_tok_offset in q_offsets:
            if q_tok_id not in tok_id_2_reps:
                continue
            tok_reps = tok_id_2_reps[q_tok_id]
            ivl_scatter_map = all_ivl_scatter_maps[q_tok_id]
            shard_scatter_map = all_shard_scatter_maps[q_tok_id]

            tok_scores = torch.matmul(tok_reps, query_tok_reps[q_tok_offset])  # ntok
            tok_scores.relu_()

            ivl_maxed_scores, _ = scatter_max(tok_scores, ivl_scatter_map, dim_size=ivl_scatter_map[-1]+1)
            match_scores.scatter_add_(0, shard_scatter_map, ivl_maxed_scores)

        top_scores, top_iids = match_scores.topk(args.top)

        all_query_match_scores.append(top_scores)
        all_query_inids.append(top_iids)

    print('Search Done', flush=True)

    # post processing
    all_query_match_scores = torch.stack(all_query_match_scores, dim=0)
    all_query_exids = torch.stack([cls_ex_ids[inids] for inids in all_query_inids], dim=0)

    torch.save((all_query_match_scores, all_query_exids), args.save_to)


if __name__ == '__main__':
    main()

