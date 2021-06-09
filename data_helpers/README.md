# Preprocessing Data
- We currently support building training data from two forms of data, triplet file and ranking file. 
- Encoding data is built from a one entry per line format. 
- Inputs are expected to be tsv files.

## Train Data From Triplets Files

A triplet file has the following format,
```
qid1 qry_text pos_did1 pos_doc_text neg_did1 neg_doc_text
qid2 qry_text pos_did2 pos_doc_text neg_did2 neg_doc_text
...
```
In other words, file used typically in triplet/hinge loss training. Run pre-processing with following command, 
```
python msmarco-passage/build_train_from_triplets.py \
    --tokenizer_name bert-base-uncased \
    --file $TRIPLET_FILE \
    --truncate {typically 128 for short passage, 512 for long document} \
    --json_dir $OUTPUT_DIRECTORY
```
This will create a train file in `$OUTPUT_DIRECTORY` that is tokenized for BERT.
## Train Data From Ranking Files
Alternatively, you can build training data directly from ranking. 

```
qid did1 score1
qid did2 score2
...
```

This requires a few other files specifying the actual text content and relevance information.

A query file,
```
qid1 qry_text
qid2 qry_text
...
```
A collection file,
```
did1 doc_text
did2 doc_text
...
```
Note that for *multi-field collection*, you can first fuse the fields into the `doc_text` field and then run the pre-processing script.
  
A query relevance file such as,
```
1185869 0       0       1
1185868 0       16      1
597651  0       49      1
403613  0       60      1
1183785 0       389     1
312651  0       616     1
80385   0       723     1
645590  0       944     1
645337  0       1054    1
186154  0       1160    1
...
```
Run pre-processing with following command, 
```
python msmarco-passage/build_train_from_rankings.py \
    --tokenizer_name bert-base-uncased \
    --rank_file $RANKING_FILE \
    --query_split_file $QUERY_FILE \
    --qrel_file $QREL_FILE \
    --collection $COLLECTION_FILE \
    --truncate 128 \
    --n_sample {number of negatives to sample per query} \
    --random \
    --json_dir $OUTPUT_DIRECTORY \
```
## Encode Data
Query or document for encoding is formatted as,
```
entry_id1 entry_text
entry_id2 entry_text
...
```
Run the following command,
```
python common/encode_entry.py \
    --tokenizer_name bert-base-uncased \
    --truncate {16 for query, 128 for short passage, 512 for long document} \
    --input_file $INPUT \
    --save_to $SAVE_DIRECTORY
```

## Split/Shard Files
For training data and corpus encoding data, we recommend first splitting them into shards for easier handling and parallelism. Note that all pre-processing code do not perform duplication detection, which means that duplicated queries in different train split will be treated as different queries. The same is also true for duplicated entries in encoding data. 
