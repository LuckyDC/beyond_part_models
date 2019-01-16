# Part Convolutional Baseline

This project implements PCB (Part-based Convolutional Baseline) of paper [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349) using [pytorch](https://github.com/pytorch/pytorch).

## Dependency

* Python 3.6
* Pytorch 1.0
* ignite 0.1

## Usage

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
| paper  | 81.90 | 65.30  |
| share-embed | 67.95  | 82.76   |
| independent-embed  | 71.70 |  84.91  |

#### Market-1501
| setting | mAP   | Rank-1 |
| ------- | ----- | ------ |
| paper  | 92.40 | 77.30  |
| share-embed | 78.03 | 92.93 |
| independent-embed  | 80.30 | 93.26  |

#### MSMT17
| setting | mAP   | Rank-1 |
| ------- | ----- | ------ |
| share-embed | 42.45   |  70.35   |
| independent-embed | 46.75    |  72.96  |
