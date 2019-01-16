# Part Convolutional Baseline

This project implements PCB (Part-based Convolutional Baseline) of paper [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349) using [pytorch](https://github.com/pytorch/pytorch).

## Dependency

* Python 3.6
* Pytorch 1.0
* ignite 0.1

## Usage
*Note: We use multi-step learning scheduler at 20th, 40th epoch instead single step at 40th epoch in the original paper.*

```bash
python3 train.py
```

```bash
python3 eval.py [gpu-id] [chekpoint-path] 
```



#### Structure

```
├── configs
│   ├── config.yml
│   └── default.py
├── data
│   ├── dataset.py
│   └── __init__.py
├── engine
│   ├── create_reid_engine.py
│   ├── __init__.py
│   └── scalar_metric.py
├── eval.py
├── extract.py
├── layers
│   ├── am_softmax.py
│   ├── norm_linear.py
│   └── random_walk_layer.py
├── models
│   ├── __init__.py
│   └── pcb.py
├── README.md
├── solver
│   └── lr_scheduler.py
├── train.py
├── transform
│   └── random_erase.py
└── utils
    └── evaluation.py

```



## Performance

#### DukeMTMC-reID

| setting | mAP   | Rank-1 |
| ------- | ----- | ------ |
| paper-1536  | 65.30 | 81.90  |
| share-embed-1536 | 67.95  | 82.76   |
| independent-embed-1536  | 71.70 |  84.91  |
| paper-12288  | 66.10  |   81.70  |
| share-embed-12288 | 62.85 |  81.28   |
| independent-embed-12288  | 65.51 |  83.39   |

#### Market-1501
| setting | mAP   | Rank-1 |
| ------- | ----- | ------ |
| paper-1536  | 77.30 | 92.40  |
| share-embed-1536 | 78.03 | 92.93 |
| independent-embed-1536  | 80.30 | 93.26  |
| paper-12288  | 77.40 | 92.30  |
| share-embed-12288 | 72.29  | 91.74   |
| independent-embed-12288  | 72.29  | 91.95  |

#### MSMT17
| setting | mAP   | Rank-1 |
| ------- | ----- | ------ |
| share-embed-1536 | 42.45   |  70.35   |
| independent-embed-1536 | 46.75    |  72.96  |
