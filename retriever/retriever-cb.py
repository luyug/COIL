import argparse
import torch
import os
from collections import defaultdict
from tqdm import tqdm, trange
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
    parser.add_argument('--batch_size', type=int, default=512)
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

    batch_size = args.batch_size

    for batch_start in trange(0, len(all_query_offsets), batch_size, desc=shard_name):
        batch_q_reps = query_cls_reps[batch_start: batch_start + batch_size]
        match_scores = torch.matmul(batch_q_reps, doc_cls_reps.transpose(0, 1))  # D * b

        batched_qtok_offsets = defaultdict(list)
        q_batch_offsets = defaultdict(list)
        for batch_offset, q_offsets in enumerate(all_query_offsets[batch_start: batch_start + batch_size]):
            for q_tok_id, q_tok_offset in q_offsets:
                if q_tok_id not in tok_id_2_reps:
                    continue
                batched_qtok_offsets[q_tok_id].append(q_tok_offset)
                q_batch_offsets[q_tok_id].append(batch_offset)

        batch_qtok_ids = list(batched_qtok_offsets.keys())
        batched_tok_scores = []

        for q_tok_id in batch_qtok_ids:
            q_tok_reps = query_tok_reps[batched_qtok_offsets[q_tok_id]]
            tok_reps = tok_id_2_reps[q_tok_id]
            tok_scores = torch.matmul(q_tok_reps, tok_reps.transpose(0, 1)).relu_()
            batched_tok_scores.append(tok_scores)

        for i, q_tok_id in enumerate(batch_qtok_ids):
            ivl_scatter_map = all_ivl_scatter_maps[q_tok_id]
            shard_scatter_map = all_shard_scatter_maps[q_tok_id]

            tok_scores = batched_tok_scores[i]
            ivl_maxed_scores = torch.empty(len(shard_scatter_map))

            for j in range(tok_scores.size(0)):
                ivl_maxed_scores.zero_()
                scatter_max(tok_scores[j], ivl_scatter_map, out=ivl_maxed_scores)
                boff = q_batch_offsets[q_tok_id][j]
                match_scores[boff].scatter_add_(0, shard_scatter_map, ivl_maxed_scores)

        top_scores, top_iids = match_scores.topk(args.top, dim=1)
        all_query_match_scores.append(top_scores)
        all_query_inids.append(top_iids)

    print('Search Done', flush=True)

    # post processing
    all_query_match_scores = torch.cat(all_query_match_scores, dim=0)
    all_query_exids = torch.cat([cls_ex_ids[inids] for inids in all_query_inids], dim=0)

    torch.save((all_query_match_scores, all_query_exids), args.save_to)


if __name__ == '__main__':
    main()

