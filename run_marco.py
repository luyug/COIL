# coding=utf-8
# Copyright 2021 COIL authors
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
from tqdm import tqdm

import numpy as np
import torch

from arguments import ModelArguments, DataArguments, COILTrainingArguments as TrainingArguments
from marco_datasets import GroupedMarcoTrainDataset, MarcoPredDataset, MarcoEncodeDataset
from modeling import COIL
from transformers import AutoConfig, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import (
    HfArgumentParser,
    set_seed,
)

from trainer import COILTrainer as Trainer

logger = logging.getLogger(__name__)

GLUE_PORTION = 0.0


@dataclass
class QryDocCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 16
    max_d_len: int = 128

    def __call__(
            self, features
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_d_len,
            return_tensors="pt",
        )

        return {'qry_input': q_collated, 'doc_input': d_collated}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model params %s", model_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = COIL.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if training_args.do_train:
        train_dataset = GroupedMarcoTrainDataset(
            data_args, data_args.train_path, tokenizer=tokenizer,
        )
    else:
        train_dataset = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QryDocCollator(
            tokenizer,
            max_q_len=data_args.q_max_len,
            max_d_len=data_args.p_max_len
        ),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        logging.info("*** Prediction ***")

        if os.path.exists(data_args.rank_score_path):
            if os.path.isfile(data_args.rank_score_path):
                raise FileExistsError(f'score file {data_args.rank_score_path} already exists')
            else:
                raise ValueError(f'Should specify a file name')
        else:
            score_dir = os.path.split(data_args.rank_score_path)[0]
            if not os.path.exists(score_dir):
                logger.info(f'Creating score directory {score_dir}')
                os.makedirs(score_dir)

        test_dataset = MarcoPredDataset(
            data_args.pred_path, tokenizer=tokenizer,
            q_max_len=data_args.q_max_len,
            p_max_len=data_args.p_max_len,
        )
        assert data_args.pred_id_file is not None

        pred_qids = []
        pred_pids = []
        with open(data_args.pred_id_file) as f:
            for l in f:
                q, p = l.split()
                pred_qids.append(q)
                pred_pids.append(p)

        pred_scores = trainer.predict(test_dataset=test_dataset).predictions

        if trainer.is_world_process_zero():
            assert len(pred_qids) == len(pred_scores)
            with open(data_args.rank_score_path, "w") as writer:
                for qid, pid, score in zip(pred_qids, pred_pids, pred_scores):
                    writer.write(f'{qid}\t{pid}\t{score}\n')

    if training_args.do_encode:
        if training_args.local_rank > -1:
            raise NotImplementedError('Encoding with multi processes is not implemented.')
        from torch.utils.data import DataLoader
        encode_dataset = MarcoEncodeDataset(
            data_args.encode_in_path, tokenizer, p_max_len=data_args.p_max_len
        )
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer,
                max_length=data_args.p_max_len,
                padding='max_length'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        encoded = []
        model = model.to(training_args.device)
        model.eval()

        for batch in tqdm(encode_loader):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                    cls, reps = model.encode(**batch)
                    encoded.append((cls.cpu(), reps.cpu()))

        all_cls = torch.cat([x[0] for x in encoded]).numpy()
        all_reps = torch.cat([x[1] for x in encoded]).numpy()

        all_pids = []
        tok_rep_dict = defaultdict(list)
        tok_pid_dict = defaultdict(list)

        for pos, entry in enumerate(tqdm(encode_dataset.nlp_dataset)):
            pid_str = entry['pid']
            if data_args.document:
                pid_str = pid_str[1:]  # remove the `D`
            pid, passage = int(pid_str), entry['psg']
            all_pids.append(pid)
            passage = passage[:data_args.p_max_len - 2]
            if not model_args.no_sep:
                # we record sep for models that use it
                passage = passage + [tokenizer.sep_token_id]

            rep_dict = defaultdict(list)
            for sent_pos, tok_id in enumerate(passage):
                rep_dict[tok_id].append(all_reps[pos][sent_pos + 1])  # skip cls
            for tok_id, tok_rep in rep_dict.items():
                tok_rep_dict[tok_id].extend(tok_rep)
                tok_pid_dict[tok_id].extend([pid for _ in range(len(tok_rep))])

        np.save(
            os.path.join(data_args.encoded_save_path, f'cls_pids'),
            np.array(all_pids)
        )
        np.save(
            os.path.join(data_args.encoded_save_path, f'cls_reps'),
            all_cls
        )
        offset_dict = {}
        tok_all_ids = []
        tok_all_reps = []
        _offset = 0
        for tok_id in tok_pid_dict:
            tok_rep = np.stack(tok_rep_dict[tok_id], axis=0)
            offset_dict[tok_id] = (_offset, tok_rep.shape[0])
            _offset += tok_rep.shape[0]
            tok_all_ids.append(np.array(tok_pid_dict[tok_id]))
            tok_all_reps.append(tok_rep)
        np.save(
            os.path.join(data_args.encoded_save_path, f'tok_pids'),
            np.concatenate(tok_all_ids, axis=0)
        )
        np.save(
            os.path.join(data_args.encoded_save_path, f'tok_reps'),
            np.concatenate(tok_all_reps, axis=0)
        )
        with open(os.path.join(data_args.encoded_save_path, f'offsets.pkl'), 'wb') as pf:
            pickle.dump(offset_dict, pf, protocol=pickle.HIGHEST_PROTOCOL)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
