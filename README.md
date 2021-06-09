# COIL
Repo for our NAACL paper, [COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List](https://arxiv.org/abs/2104.07186). The code covers learning COIL models well as encoding and retrieving with COIL index. 

The code was refactored from our original experiment version to use the huggingface Trainer interface for future compatibility.

## Contextualized Exact Lexical Match
COIL systems are based on the idea of *contextualized exact lexical match*. It replaces term frequency based term matching in classical systems like BM25 with contextualized word representation similarities. It thereby gains the ability to model matching of context. Meanwhile COIL confines itself to comparing exact  lexical matched tokens and therefore can retrieve efficiently with inverted list form data structure.  Details can be found in our [paper](https://arxiv.org/abs/2104.07186).


## Dependencies
The code has been tested with,
```
pytorch==1.8.1
transformers==4.2.1
datasets==1.1.3
```
To use the retriever, you need in addition,
```
torch_scatter==2.0.6
faiss==1.7.0
```
## Resource
**MSMARCO Passage Ranking**

 Tokenized data and model checkpoint [link](http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/)
 
 Hard negative data and model checkpoint [link](http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg-hn/)
 
 *more to be added soon*
## Usage
The following sections will work through how to use this code base to train and retrieve over the MSMARCO passage ranking data set.
## Training
You can download the train file `psg-train.tar.gz` for BERT from our resource [link](http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/). Alternatively, you can run pre-processing by yourself following the pre-processing [instructions](data_helpers).

Extract the training set from the tar ball and run the following code to launch training for msmarco passage.
```
python run_marco.py \  
  --output_dir $OUTDIR \  
  --model_name_or_path bert-base-uncased \  
  --do_train \  
  --save_steps 4000 \  
  --train_dir /path/to/psg-train \  
  --q_max_len 16 \  
  --p_max_len 128 \  
  --fp16 \  
  --per_device_train_batch_size 8 \  
  --train_group_size 8 \  
  --cls_dim 768 \  
  --token_dim 32 \  
  --warmup_ratio 0.1 \  
  --learning_rate 5e-6 \  
  --num_train_epochs 5 \  
  --overwrite_output_dir \  
  --dataloader_num_workers 16 \  
  --no_sep \  
  --pooling max 
```


## Encoding
After training, you can encode the corpus splits and queries.

You can download pre-processed data for BERT, `corpus.tar.gz, queries.{dev, eval}.small.json` [here](http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/). 
```
for i in $(seq -f "%02g" 0 99)  
do  
  mkdir ${ENCODE_OUT_DIR}/split${i}  
  python run_marco.py \  
    --output_dir $ENCODE_OUT_DIR \  
    --model_name_or_path $CKPT_DIR \  
    --tokenizer_name bert-base-uncased \  
    --token_dim 768 \  
    --cls_dim 32 \  
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
If on a cluster, the encoding loop can be paralellized. For example, say if you are on a SLURM cluster, use `srun`,
```
for i in $(seq -f "%02g" 0 99)  
do  
  mkdir ${ENCODE_OUT_DIR}/split${i}  
  srun --ntasks=1 -c4 --mem=16000 -t0 --gres=gpu:1 python run_marco.py \  
    --output_dir $ENCODE_OUT_DIR \  
    --model_name_or_path $CKPT_DIR \  
    --tokenizer_name bert-base-uncased \  
    --token_dim 768 \  
    --cls_dim 32 \  
    --do_encode \  
    --no_sep \  
    --p_max_len 128 \  
    --pooling max \  
    --fp16 \  
    --per_device_eval_batch_size 128 \  
    --dataloader_num_workers 12 \  
    --encode_in_path ${TOKENIZED_DIR}/split${i} \  
    --encoded_save_path ${ENCODE_OUT_DIR}/split${i}&
done
```


Then encode the queries,
```
python run_marco.py \  
  --output_dir $ENCODE_QRY_OUT_DIR \  
  --model_name_or_path $CKPT_DIR \  
  --tokenizer_name bert-base-uncased \  
  --token_dim 768 \  
  --cls_dim 32 \  
  --do_encode \  
  --p_max_len 16 \  
  --fp16 \  
  --no_sep \  
  --pooling max \  
  --per_device_eval_batch_size 128 \  
  --dataloader_num_workers 12 \  
  --encode_in_path $TOKENIZED_QRY_PATH \  
  --encoded_save_path $ENCODE_QRY_OUT_DIR
```
Note that here `p_max_len` always controls the maximum length of the encoded text, regardless of the input type.

## Retrieval
To do retrieval, run the following steps, 

(Note that there is no dependency in the for loop within each step, meaning that if you are on a cluster, you can distribute the jobs across nodes using `srun` or `qsub`.)

1) build document index shards
```
for i in $(seq 0 9)  
do  
 python retriever/sharding.py \  
   --n_shards 10 \  
   --shard_id $i \  
   --dir $ENCODE_OUT_DIR \  
   --save_to $INDEX_DIR \  
   --use_torch
done  
```
2) reformat encoded query
```
python retriever/format_query.py \  
  --dir $ENCODE_QRY_OUT_DIR \  
  --save_to $QUERY_DIR \  
  --as_torch
```

3) retrieve from each shard
```
for i in $(seq -f "%02g" 0 9)  
do  
  python retriever/retriever-compat.py \  
      --query $QUERY_DIR \  
      --doc_shard $INDEX_DIR/shard_${i} \  
      --top 1000 \  
      --save_to ${SCORE_DIR}/intermediate/shard_${i}.pt
done 
```
4) merge scores from all shards
```
python retriever/merger.py \  
  --score_dir ${SCORE_DIR}/intermediate/ \  
  --query_lookup  ${QUERY_DIR}/cls_ex_ids.pt \  
  --depth 1000 \  
  --save_ranking_to ${SCORE_DIR}/rank.txt

python data_helpers/msmarco-passage/score_to_marco.py \  
  --score_file ${SCORE_DIR}/rank.txt
```

Note that this compat(ible) version of retriever differs from our internal retriever. It relies on `torch_scatter` package for scatter operation so that we can have a pure python code that can easily work across platforms.  We do notice that on our system `torch_scatter` does not scale very well with number of cores. We may in the future release another faster version of retriever that requires some compiling work. 

##  Data Format
For both training and encoding,  the core code expects pre-tokenized data.
### Training Data
Training data is grouped by query into one or several json files where each line has a query, its corresponding positives and negatives.
```
{
    "qry": {
        "qid": str,
        "query": List[int],
    },
    "pos": List[
        {
            "pid": str,
            "passage": List[int],
        }
    ],
    "neg": List[
        {
            "pid": str,
            "passage": List[int]
        }
    ]
}
```
### Encoding Data
Encoding data is also formatted into one or several json files. Each line corresponds to an entry item.
```
{"pid": str, "psg": List[int]}
```
Note that for code simplicity, we share this format for query/passage/document encoding.
