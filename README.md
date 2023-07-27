# STKGR-PR

## Requirements

- python3 (tested on 3.6.6)
- pytorch (tested on 1.5.0)

## Datasets

there are six datasets under folder `data`.

``` bash
# dataset ICEWS14-2%
data/ICEWS14-2

# dataset ICEWS14-3%
data/ICEWS14-3

# dataset ICEWS14-5%
data/ICEWS14-5

# dataset ICEWS05-15-0.3%
data/ICEWS515-03

# dataset ICEWS05-15-0.4%
data/ICEWS515-04

# dataset ICEWS05-15-0.5%
data/ICEWS515-05
```

## Data Processing

``` bash
./experiment.sh configs/<dataset>.sh --process_data <gpu-ID>
```

`dataset` is the name of datasets. In our experiments, `dataset` could be `ICEWS14-2`, `ICEWS14-3`, `ICEWS14-5`, `ICEWS515-03`, `ICEWS515-04` and `ICEWS515-05`. `<gpu-ID>` denotes a non-negative integer that serves as the index for identifying a specific GPU.

## Pretrain Temporal Knowledge Graph Embedding

``` bash
./experiment-emb.sh configs/<dataset>-<model>.sh --train <gpu-ID>
```

`dataset` is the name of datasets and `model` is the name of temporal knowledge graph embedding model. In our experiments, `dataset` could be `ICEWS14-2`, `ICEWS14-3`, `ICEWS14-5`, `ICEWS515-03`, `ICEWS515-04` and `ICEWS515-05`, `model` could be `conve`. `<gpu-ID>` denotes a non-negative integer that serves as the index for identifying a specific GPU.

## Train

``` bash
# take ICEWS05-15-0.3% for example
./experiment-rs.sh configs/icews515-03-rs.sh --train <gpu-ID> 
```

## Test

``` bash
# take ICEWS05-15-0.3% for example
./experiment-rs.sh configs/icews515-03-rs.sh --inference <gpu-ID> 
```

## Acknowledgement

We refer to the code of [DacKGR](Xin Lv, Xu Han, Lei Hou, Juanzi Li, Zhiyuan Liu, Wei Zhang, Yichi Zhang, Hao Kong, Suhui Wu. Dynamic Anticipation and Completion for Multi-Hop Reasoning over Sparse Knowledge Graph. *The Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)*.). Thanks for their contributions.

