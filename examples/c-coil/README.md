# C-COIL
C-COIL (working title) aims at improving vanilla COIL systems with improved model initialization.

It has zero added fine-tuninig/inference cost but only requires replacing BERT initializer.

As of 7/27/2021, C-COIL retrieval system is the top run on MS-MARCO passage ranking leaderboard.

## Resource 


| Configuration | MARCO DEV MRR@10 | TREC DL19 NDCG@5 | TREC DL19 NDCG@10 | Chekpoint | MARCO Train Ranking | MARCO Dev Ranking |
| --- | :---: |  :---: | :---: | :---: | :---: | :---: | 
| C-COIL w/ HN    | 0.3734       | 0.749      | 0.726       | checkpoint.tar.gz |  [train-ranking.tar.gz](http://boston.lti.cs.cmu.edu/luyug/c-coil/train-ranking.tar.gz)       | [dev-ranking.tsv](http://boston.lti.cs.cmu.edu/luyug/c-coil/dev-ranking.tsv)    

To save compute, we did the following,
- Use COIL-bm25 results as hard negatives
- Take the last training checkpoint 

If sufficient compute resource is available, you should consider mining hard negatives also with a C-COIL system and 
perform some validation on last few checkpoints.



The following sections covers C-COIL for msmarco passage ranking.

## Training Retriever
Get the data,
```
cd $DATA_DIR
wget http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/psg-train.tar.gz
wget http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg-hn/psg-train-hn.tar.gz

tar -xzf psg-train.tar.gz
tar -xzf psg-train-hn.tar.gz

mkdir train-data
cd train-data

# We use a mixture of BM25 and hard negatives following DPR setup
ln -s ../psg-train/* .
ln -s ../psg-train-hn/* .
```

Train,
```
python run_marco.py \  
  --output_dir $OUTDIR \  
  --model_name_or_path /path/to/c-coil-init \  
  --do_train \  
  --save_steps 4000 \  
  --train_dir $DATA_DIR/train-data \  
  --q_max_len 16 \  
  --p_max_len 128 \  
  --fp16 \  
  --per_device_train_batch_size 8 \  
  --train_group_size 8 \  
  --cls_dim 768 \  
  --token_dim 32 \  
  --warmup_ratio 0.1 \  
  --learning_rate 5e-6 \  
  --num_train_epochs 3 \  
  --overwrite_output_dir \  
  --dataloader_num_workers 16 \  
  --no_sep \  
  --pooling max 
```

## Training Reranker
We used the [reranker toolkit](https://github.com/luyug/Reranker). Details to be added.

## Inference  
C-COIL follws the same inference setup as COIL. You can find detailed explanation on the homepage.

```
for i in $(seq -f "%02g" 0 99)  
do  
  mkdir ${ENCODE_OUT_DIR}/split${i}  
  python run_marco.py \  
    --output_dir $ENCODE_OUT_DIR \  
    --model_name_or_path $CKPT_DIR \  
    --tokenizer_name bert-base-uncased \  
    --cls_dim 768 \  
    --token_dim 32 \  
    --do_encode \  
    --no_sep \  
    --p_max_len 128 \  
    --pooling max \  
    --fp16 \  
    --per_device_eval_batch_size 128 \  
    --dataloader_num_workers 12 \  
    --encode_in_path ${TOKENIZED_DIR}/split${i} \  
    --encoded_save_path ${ENCODE_OUT_DIR}/split${i}
done
```

For retrival, see the [retriever page](retriever).
