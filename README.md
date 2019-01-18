# Part Convolutional Baseline

This project implements PCB (Part-based Convolutional Baseline) of paper [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349) using [pytorch](https://github.com/pytorch/pytorch).

## Dependency

* python 3.6
* pytorch 1.0
* torchvision 
* ignite
* yacs  

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
│   └── norm_linear.py
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
    ├── evaluation.py
    └── initializer.py

```



## Performance

#### DukeMTMC-reID

| setting | mAP   | Rank-1 |
| ------- | ----- | ------ |
| paper-1536  | 65.30 | 81.90  |
| share-embed-1536 | 70.15   | 84.38   |
| independent-embed-1536  | 71.48  |  85.10   |
| paper-12288  | 66.10  |   81.70  |
| share-embed-12288 | 64.91  |  82.90   |
| independent-embed-12288  | 65.26  |  83.98   |

#### Market-1501
| setting | mAP   | Rank-1 |
| ------- | ----- | ------ |
| paper-1536  | 77.30 | 92.40  |
| share-embed-1536 |  78.68 | 92.96 |
| independent-embed-1536  | 79.79 | 93.08  |
| paper-12288  | 77.40 | 92.30  |
| share-embed-12288 | 73.09   | 92.36   |
| independent-embed-12288  | 72.72  |  92.07  |

#### MSMT17
| setting | mAP   | Rank-1 |
| ------- | ----- | ------ |
| share-embed-1536 | 42.39   |  69.59   |
| independent-embed-1536 |  47.20   |  74.05  |


*We also evaluate the original setting with a decay step-size of 20 which is adopted in some re-implementations.*

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

## Ablation Studies

#### Experiments with shared 1x1 conv on DukeMTMC-reID

| num parts | mAP-h | R1-h  | mAP-g | R1-g  |
| --------- | ----- | ----- | ----- | ----- |
| 1         | 58.38 | 76.88 | 59.80 | 80.20 |
| 2         | 66.68 | 82.54 | 62.84 | 81.41 |
| 3         | 66.66 | 83.48 | 63.15 | 81.86 |
| 4         | 66.96 | 82.40 | 62.21 | 81.28 |
| 5         | 67.17 | 82.67 | 61.57 | 81.10 |
| 6         | 70.09 | 84.20 | 64.90 | 83.21 |

#### Experiments with unshared 1x1 conv on DukeMTMC-reID

| num parts | mAP-h | R1-h  | mAP-g | R1-g  |
| --------- | ----- | ----- | ----- | ----- |
| 1         | 58.38 | 76.88 | 59.80 | 80.20 |
| 2         | 66.34 | 83.03 | 61.84 | 81.32 |
| 3         | 67.80 | 82.89 | 60.60 | 80.38 |
| 4         | 70.07 | 84.42 | 63.40 | 82.40 |
| 5         | 69.74 | 84.38 | 63.92 | 82.63 |
| 6         | 71.48 | 85.10 | 65.26 | 83.98 |




