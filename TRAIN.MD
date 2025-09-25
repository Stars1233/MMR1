# MMR1 Variance-Aware Sampling

This guide provides comprehensive instructions for reproduce VAS in MMR1.

## Overview

EasyR1 supports three main sampling strategies for training:

-**Sequential**: Process samples in order

-**Curriculum**: Variance-Aware Sampling

## Quick Start

### Basic Training with Shuffle Strategy

```bash

JOB_NAME=my_training\

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct\

bashexamples/mmr1/train_qwen2_5_vl_3b.sh\

    data.train_files=/path/to/training/data \

data.sampling_strategy=shuffle\

    trainer.total_episodes=10

```

### Variance-Aware Sampling with Outcome Variance Score (OVS)

```bash

JOB_NAME=curriculum_training\

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct\

bashexamples/mmr1/train_qwen2_5_vl_3b.sh\

    data.train_files=/path/to/training/data \

data.sampling_strategy=curriculum\

    'data.curriculum_metrics=[learnability]' \

'data.curriculum_metric_weights=[1.0]'\

    data.curriculum_mixture_ratio=0.5 \

trainer.total_episodes=10

```

## Variance-Aware Sampling Features

### Available Metrics

-**Outcome Variance Score (OVS)**:`learnability` Measures sample difficulty based on model performance. Given the accuracy p, we define the learnability here as p(1-p) **Outcome Variance Score (OVS)**

-**Trajectory Diversity Score (TDS)**: `self-bleu` Similarity between generated responses

### Key Parameters

| Parameter               | Description                                | Default  |
|--------------------------|--------------------------------------------|----------|
| `curriculum_metrics`     | List of metrics for weighting              | `[learnability]` |
| `curriculum_metric_weights` | Weights for combining metrics            | `[1.0]`  |
| `curriculum_mixture_ratio`  | Ratio of weighted vs random sampling (0.0–1.0) | `0.5`    |
| `curriculum_update_freq` | Update weights every N steps (0 for epoch-level) | `4`      |
| `curriculum_rollout_n`   | Number of rollouts for metric calculation | `8`      |
| `curriculum_momentum`    | Momentum for weight updates               | `0.0`    |


## Advanced Examples

### Multi-Metric Variance-Aware Sampling

Combine OVS and TDS metrics:

```bash

JOB_NAME=advanced_variance_aware_sampling\

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct\

bashexamples/mmr1/train_qwen2_5_vl_3b.sh\

    data.train_files=/path/to/data \

data.sampling_strategy=curriculum\

    'data.curriculum_metrics=[learnability,self-bleu]' \

'data.curriculum_metric_weights=[0.8,0.2]'\

    data.curriculum_mixture_ratio=0.8 \

data.curriculum_update_freq=8\

    trainer.total_episodes=10

```

## Configuration via YAML

Create a custom configuration file:

```yaml

# custom_curriculum.yaml

defaults:

-mmr1_b200


data:

sampling_strategy: curriculum

curriculum_metrics:

-learnability

-self-bleu

curriculum_metric_weights:

-0.7

-0.3

curriculum_mixture_ratio: 0.6

curriculum_update_freq: 10

curriculum_rollout_n: 12

```

Use with:

```bash

python-mverl.trainer.mainconfig=custom_curriculum.yaml

```


### For Quick Experimentation

- Start with single metric: `'data.curriculum_metrics=[learnability]'`
- Use `curriculum_mixture_ratio=0.5` as baseline
- Set `curriculum_rollout_n=4` for faster iteration


## Monitoring Curriculum Learning

Track these metrics in wandb/tensorboard:

-`curriculum/mean_weight`: Average sample weight

-`curriculum/std_weight`: Weight distribution spread

-`curriculum/min_weight`, `curriculum/max_weight`: Weight range

-`curriculum/consumed_batches`: Training progress

-`curriculum/random_position`: Mix of weighted vs random samples
