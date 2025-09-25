# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Dict, List

import numpy as np
import ray
import torch
from Levenshtein import distance
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from ..protocol import DataProto
from ..workers.reward import CustomRewardManager


def calculate_distinct_n(tokenized_sequences, n):
    """Calculate distinct-n metric using model's tokenized sequences."""
    all_ngrams = []
    for seq in tokenized_sequences:
        all_ngrams.extend(list(zip(*[seq[i:] for i in range(n)])))

    if not all_ngrams:
        return 0.0

    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def calculate_self_bleu_n(tokenized_sequences, n):
    """Calculate self-BLEU score for specific n-gram using model's tokenized sequences."""
    if len(tokenized_sequences) < 2:
        return 0.0

    weights = tuple([1.0] if n == 1 else [0.0] * (n - 1) + [1.0])  # Only use n-gram
    scores = []

    for i, hyp in enumerate(tokenized_sequences):
        refs = tokenized_sequences[:i] + tokenized_sequences[i + 1 :]
        score = sentence_bleu(
            refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1
        )
        scores.append(score)

    return np.mean(scores)


def calculate_self_bleu_123(tokenized_sequences):
    """Calculate self-BLEU-123 score with uniform weights using model's tokenized sequences."""
    if len(tokenized_sequences) < 2:
        return 0.0

    weights = (1 / 3, 1 / 3, 1 / 3)  # Uniform weights for 1,2,3-grams
    scores = []

    for i, hyp in enumerate(tokenized_sequences):
        refs = tokenized_sequences[:i] + tokenized_sequences[i + 1 :]
        score = sentence_bleu(
            refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1
        )
        scores.append(score)

    return np.mean(scores)


def calculate_pairwise_edit_distance(tokenized_sequences):
    """Calculate average pairwise edit distance between all tokenized sequences."""
    if len(tokenized_sequences) < 2:
        return 0.0

    distances = []
    for i in range(len(tokenized_sequences)):
        for j in range(i + 1, len(tokenized_sequences)):
            dist = distance(tokenized_sequences[i], tokenized_sequences[j])
            distances.append(dist)

    return np.mean(distances) if distances else 0.0


def calculate_learnability_metric(reward_fn, batch, batch_size, curriculum_rollout_n):
    """Remote task for calculating learnability metric."""
    reward_tensor, reward_metrics = reward_fn(batch)
    # reshape to (batch_size, n, max_len)
    acc_reward_tensor = torch.tensor(
        reward_metrics["accuracy"], dtype=torch.float32
    ).view(batch_size, curriculum_rollout_n, -1)

    # Calculate pass rate and learnability
    pass_rate = acc_reward_tensor.mean(dim=-1).mean(
        dim=-1
    )  # sequence-level average over rollouts
    learnability = pass_rate * (1 - pass_rate)
    return learnability


def combine_metric_results(
    results: list[torch.Tensor], weights: list[float], metrics: list[str]
) -> torch.Tensor:
    """Combine metric results with their weights."""
    weighted_sum = None
    for i, (result, metric) in enumerate(zip(results, metrics)):
        weight = weights[i]
        if weighted_sum is None:
            weighted_sum = result * weight
        else:
            weighted_sum += result * weight
    return weighted_sum


@ray.remote
def calculate_single_bleu_score(tokens: List, start_idx: int, end_idx: int):
    """Ray task for calculating single BLEU score."""
    item_tokens = tokens[start_idx:end_idx]
    return calculate_self_bleu_123(item_tokens)


@ray.remote
def calculate_single_edit_distance(tokens: List, start_idx: int, end_idx: int):
    """Ray task for calculating single edit distance."""
    item_tokens = tokens[start_idx:end_idx]
    return calculate_pairwise_edit_distance(item_tokens)


@ray.remote(max_retries=3)
def calculate_learnability_metric_with_batch_data(
    reward_fn: CustomRewardManager,
    batch_data: Dict,
    gen_batch_output: DataProto,
    batch_size: int,
    curriculum_rollout_n: int,
):
    """Remote task for calculating learnability metric with pre-extracted batch data.

    Args:
        reward_fn: Can be either CustomRewardManager object or ray.ObjectRef to it
        batch_data: Pre-extracted batch data dictionary
        gen_batch_output: Generated batch output
        batch_size: Size of the batch
        curriculum_rollout_n: Number of curriculum rollouts
    """
    batch = DataProto.from_single_dict(batch_data)

    if "multi_modal_inputs" in batch.non_tensor_batch.keys():
        _ = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "multi_modal_data",
                "multi_modal_inputs",
            ],
        )
    else:
        _ = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],
        )

    batch = batch.repeat(repeat_times=curriculum_rollout_n, interleave=True)
    batch = batch.union(gen_batch_output)

    # If reward_fn is an ObjectRef, Ray will automatically dereference it
    # No need for explicit ray.get() as it's a top-level argument
    return calculate_learnability_metric(
        reward_fn=reward_fn,
        batch=batch,
        batch_size=batch_size,
        curriculum_rollout_n=curriculum_rollout_n,
    )


@ray.remote(max_retries=3)
def calculate_single_distinct_n(shared_data_ref, start_idx: int, end_idx: int, n: int):
    """Ray task for calculating distinct-n score for a single batch item."""
    tokenized_sequences = ray.get(shared_data_ref)
    sequences_slice = tokenized_sequences[start_idx:end_idx]
    return calculate_distinct_n(sequences_slice, n=n)


@ray.remote(max_retries=3)
def calculate_distinct_n_metric(responses, batch_size, curriculum_rollout_n, n=3):
    """Remote task for calculating distinct-n metric with optimized centralized serialization."""
    tokenized_sequences = responses.tolist()

    # 1. 数据预处理和共享 - 一次序列化，多次使用
    shared_data_ref = ray.put(tokenized_sequences)

    # 2. 批量任务参数准备
    task_params = [
        (shared_data_ref, i * curriculum_rollout_n, (i + 1) * curriculum_rollout_n, n)
        for i in range(batch_size)
    ]

    # 3. 分块批量提交（避免调度器过载）
    chunk_size = min(100, batch_size)
    all_futures = []

    for i in range(0, len(task_params), chunk_size):
        chunk = task_params[i:i+chunk_size]
        chunk_futures = [
            calculate_single_distinct_n.remote(data_ref, start_idx, end_idx, n_val)
            for data_ref, start_idx, end_idx, n_val in chunk
        ]
        all_futures.extend(chunk_futures)

    distinct_scores = ray.get(all_futures)
    return torch.tensor(distinct_scores, dtype=torch.float32)


@ray.remote(max_retries=3)
def calculate_self_bleu_metric(responses, batch_size, curriculum_rollout_n):
    """Remote task for calculating self-BLEU-123 metric using optimized Ray tasks."""
    tokenized_sequences = responses.tolist()

    # 1. 数据预处理和共享 - 一次序列化，多次使用
    shared_data_ref = ray.put(tokenized_sequences)

    # 2. 批量任务参数准备
    task_params = [
        (shared_data_ref, i * curriculum_rollout_n, (i + 1) * curriculum_rollout_n)
        for i in range(batch_size)
    ]

    # 3. 分块批量提交（避免调度器过载）
    chunk_size = min(8, batch_size)  # 动态调整chunk大小
    all_futures = []

    for i in range(0, len(task_params), chunk_size):
        chunk = task_params[i:i+chunk_size]

        # 批量创建这个chunk的任务
        chunk_futures = [
            calculate_single_bleu_score.remote(data_ref, start_idx, end_idx)
            for data_ref, start_idx, end_idx in chunk
        ]

        all_futures.extend(chunk_futures)

    # 4. 批量收集结果
    bleu_scores = ray.get(all_futures)
    return torch.tensor(bleu_scores, dtype=torch.float32)


@ray.remote(max_retries=3)
def calculate_edit_distance_metric(responses, batch_size, curriculum_rollout_n):
    """Remote task for calculating edit distance metric using optimized Ray tasks."""
    tokenized_sequences = responses.tolist()

    # 1. 数据预处理和共享 - 一次序列化，多次使用
    shared_data_ref = ray.put(tokenized_sequences)

    # 2. 批量任务参数准备
    task_params = [
        (shared_data_ref, i * curriculum_rollout_n, (i + 1) * curriculum_rollout_n)
        for i in range(batch_size)
    ]

    # 3. 分块批量提交（避免调度器过载）
    chunk_size = min(100, batch_size)  # 动态调整chunk大小
    all_futures = []

    for i in range(0, len(task_params), chunk_size):
        chunk = task_params[i:i+chunk_size]

        # 批量创建这个chunk的任务
        chunk_futures = [
            calculate_single_edit_distance.remote(data_ref, start_idx, end_idx)
            for data_ref, start_idx, end_idx in chunk
        ]

        all_futures.extend(chunk_futures)

    # 4. 批量收集结果
    edit_scores = ray.get(all_futures)
    return torch.tensor(edit_scores, dtype=torch.float32)