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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import os.path as osp
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Type, Tuple

import numpy as np
import ray
import ray.exceptions
import torch
from codetiming import Timer
from Levenshtein import distance
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from ray.experimental.tqdm_ray import tqdm
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str
from ..utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from ..workers.reward import CustomRewardManager
from ..curriculum import (
    CurriculumWeightedSampler,
    MixedCurriculumSampler,
    PaddedSequentialSampler,
    calculate_distinct_n_metric,
    calculate_edit_distance_metric,
    calculate_learnability_metric_with_batch_data,
    calculate_self_bleu_metric,
    combine_metric_results,
    pad_list,
)


# Allow very large images
Image.MAX_IMAGE_PIXELS = None


WorkerType = Type[Worker]


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}."
            )


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"
):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    if "ref_log_probs" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_probs"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = VF.masked_mean(
        kld, mask=response_mask, dim=-1
    )  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards,
            reward_baselines=reward_baselines,
            eos_mask=response_mask,
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@contextmanager
def _timer(name: str, timing_raw: dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last





class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn: Callable = None,
        val_reward_fn: Callable = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"ActorRollout should be included in {role_worker_mapping.keys()}."
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print(
                "KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics."
            )

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(
                f"Unknown advantage estimator: {config.algorithm.adv_estimator}."
            )

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError(
                "Rollout batch size must be divisible by global batch size."
            )

        if (
            self.use_critic
            and config.data.rollout_batch_size % config.worker.critic.global_batch_size
            != 0
        ):
            raise ValueError(
                "Rollout batch size must be divisible by global batch size."
            )

        self.checkpoint_path = self.config.load_checkpoint_path = (
            self._get_checkpoint_path()
        )

        # Initialize workers first
        self.init_workers()

        self.global_step = 0
        # Load checkpoint for workers and dataloader
        self._load_worker_state()

        # Then create dataloader which depends on workers
        self._create_dataloader()

        # Curriculum related variables
        self.logger = Tracker(
            loggers=self.config.trainer.logger, config=self.config.to_dict()
        )

    def _create_dataloader(self) -> None:
        # --- Create Train Dataset ---
        self.train_dataset = RLHFDataset(
            data_path=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=self.config.data.system_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )

        # --- Create Sampler ---

        if self.config.data.sampling_strategy == "curriculum":
            curriculum_state = self._load_curriculum_state()

            # Load or compute curriculum weights
            # print(f"DEBUG: always compute curriculum weights")
            if curriculum_state is not None:
                self.train_dataset.curriculum_weights = curriculum_state[
                    "curriculum_weights"
                ]
                print(
                    f"Loaded curriculum weights to self.train_dataset.curriculum_weights"
                )
            else:
                print("No curriculum state found, will compute curriculum weights")
                self._update_curriculum_weights()

            # Initialize curriculum sampler (After self.train_dataset.curriculum_weights is initialized)
            sampler = self.curriculum_sampler = MixedCurriculumSampler(
                dataset=self.train_dataset,
                weights=self.train_dataset.curriculum_weights,
                batch_size=self.config.data.rollout_batch_size,
                mixture_ratio=self.config.data.curriculum_mixture_ratio,
                replacement=True,  # Always use replacement for weighted sampling
                generator=torch.Generator().manual_seed(self.config.data.seed),
            )

            # Load curriculum sampler state if available
            if curriculum_state is not None:
                self.curriculum_sampler.load_state_dict(
                    curriculum_state["sampler_state"]
                )

            self.curriculum_sampler.update_weights(self.train_dataset.curriculum_weights)

            assert (
                self.train_dataset.curriculum_weights is not None
            ), "curriculum_weights should be initialized"

        elif self.config.data.sampling_strategy == "shuffle":
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:  # sequential
            sampler = SequentialSampler(data_source=self.train_dataset)
        

        # --- Create Train Dataloader ---


        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        self.val_dataset = RLHFDataset(
            data_path=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=self.config.data.system_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=(
                len(self.val_dataset)
                if self.config.data.val_batch_size == -1
                else self.config.data.val_batch_size
            ),
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = (
                len(self.train_dataloader) * self.config.trainer.total_episodes
            )

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _calculate_curriculum_metric(
        self, dataset: RLHFDataset, indices: List[int]
    ) -> list[ray.ObjectRef]:
        """Calculate the curriculum metric based on the configured strategy.

        Returns:
            A list containing futures for the remote metric calculations.
        """

        # --- Prepare gen batch - extract data on driver to preserve memory mapping ---
        batch_data = collate_fn([dataset[i] for i in indices])
        batch = DataProto.from_single_dict(batch_data)
        batch_size = len(indices)

        # Generate responses for the batch - this has to be done sequentially
        if "multi_modal_inputs" in batch.non_tensor_batch.keys():
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    "raw_prompt_ids",
                    "multi_modal_data",
                    "multi_modal_inputs",
                ],
            )
        else:
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

        # --- Generate responses ---
        gen_batch.meta_info["n"] = self.config.data.curriculum_rollout_n
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        gen_batch.meta_info.pop("n")

        # --- Submit future for remote metric calculations ---
        metric_futures = []

        for metric_name in self.config.data.curriculum_metrics:
            if metric_name == "learnability":
                future = calculate_learnability_metric_with_batch_data.remote(
                    reward_fn=self.reward_fn,
                    batch_data=batch_data,  # Pass pre-extracted data instead of dataset
                    gen_batch_output=gen_batch_output,
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "distinct_3":
                future = calculate_distinct_n_metric.remote(
                    responses=gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                    n=3,
                )
            elif metric_name == "self_bleu_123":
                future = calculate_self_bleu_metric.remote(
                    responses=gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "edit_distance":
                future = calculate_edit_distance_metric.remote(
                    responses=gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            else:
                raise ValueError(f"Unknown curriculum metric: {metric_name}")

            metric_futures.append(future)

        # Return the futures
        return metric_futures

    def _calculate_curriculum_metric_with_batch(
        self, batch_data: Dict, reward_fn_ref: ray.ObjectRef = None
    ) -> list[ray.ObjectRef]:
        """Calculate curriculum metrics using pre-loaded batch data from dataloader."""
        
        batch = DataProto.from_single_dict(batch_data)
        batch_size = len(batch)

        # Generate responses for the batch - this has to be done sequentially
        if "multi_modal_inputs" in batch.non_tensor_batch.keys():
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    "raw_prompt_ids",
                    "multi_modal_data",
                    "multi_modal_inputs",
                ],
            )
        else:
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

        # --- Generate responses ---
        gen_batch.meta_info["n"] = self.config.data.curriculum_rollout_n
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        gen_batch.meta_info.pop("n")

        # --- Submit future for remote metric calculations ---
        metric_futures = []

        for metric_name in self.config.data.curriculum_metrics:
            if metric_name == "learnability":
                # Use reward_fn_ref if provided, otherwise fall back to self.reward_fn
                reward_fn_to_use = reward_fn_ref if reward_fn_ref is not None else self.reward_fn
                future = calculate_learnability_metric_with_batch_data.remote(
                    reward_fn=reward_fn_to_use,
                    batch_data=batch_data,  # Pass the original batch_data
                    gen_batch_output=gen_batch_output,
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "distinct_3":
                future = calculate_distinct_n_metric.remote(
                    responses=gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                    n=3,
                )
            elif metric_name == "self_bleu_123":
                future = calculate_self_bleu_metric.remote(
                    responses=gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "edit_distance":
                future = calculate_edit_distance_metric.remote(
                    responses=gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            else:
                raise ValueError(f"Unknown curriculum metric: {metric_name}")

            metric_futures.append(future)

        return metric_futures

    def _save_curriculum_weights(self, weights: torch.Tensor, step: int = None) -> None:
        """Save curriculum weights to a file."""
        # Create directory if it doesn't exist
        weights_dir = os.path.join(
            self.config.trainer.save_checkpoint_path, "curriculum_weights"
        )
        os.makedirs(weights_dir, exist_ok=True)

        # Create filename based on step (if provided) or use 'initial'
        filename = (
            f"weights_step_{step}.pt" if step is not None else "initial_weights.pt"
        )
        weights_path = os.path.join(weights_dir, filename)

        # Save weights
        torch.save(weights, weights_path)
        print(f"Saved curriculum weights to {weights_path}")

    def _compute_curriculum_weights(self) -> torch.Tensor:
        """Compute curriculum weights for each sample in the dataset using dataloader for efficient loading."""

        curriculum_weights = torch.zeros(len(self.train_dataset), dtype=torch.float32)
        
        # Put reward_fn into Ray object store once to avoid repeated serialization
        reward_fn_ref = ray.put(self.reward_fn)

        # Create a temporary dataloader with padded sequential sampling
        # Use PaddedSequentialSampler to ensure all batches have consistent size
        padded_sampler = PaddedSequentialSampler(
            dataset=self.train_dataset,
            batch_size=self.config.data.curriculum_rollout_batch_size
        )
        
        curriculum_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.curriculum_rollout_batch_size,
            sampler=padded_sampler,
            num_workers=8,  # Same as train_dataloader for prefetch
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,  # PaddedSequentialSampler handles the padding
        )

        # Process batches sequentially but don't wait for metric calculations
        metric_futures_with_batch_idx = []  # Store (batch_idx, futures_dict) tuples

        print("Generating sequences and launching metric calculations...")
        for batch_idx, batch_data in enumerate(curriculum_dataloader):
            # Calculate metrics using pre-loaded batch_data
            metric_futures_cur_batch = self._calculate_curriculum_metric_with_batch(
                batch_data=batch_data,
                reward_fn_ref=reward_fn_ref  # Pass the ObjectRef instead of the object
            )

            # Store batch index and futures for later collection
            metric_futures_with_batch_idx.append((batch_idx, metric_futures_cur_batch))

            print(
                f"Launched {len(metric_futures_cur_batch)} metric calculations for batch {batch_idx}"
            )

        # Now process all the pending futures
        for batch_idx, futures in metric_futures_with_batch_idx:
            # Collect results from futures
            metric_results = ray.get(futures)

            # Combine the metric results for this batch
            combined_metric = combine_metric_results(
                metric_results,
                weights=self.config.data.curriculum_metric_weights,
                metrics=self.config.data.curriculum_metrics,
            )

            # Store the combined metric, handling potential padding
            start_idx = batch_idx * self.config.data.curriculum_rollout_batch_size
            end_idx = min(
                start_idx + self.config.data.curriculum_rollout_batch_size,
                len(self.train_dataset),
            )
            
            # Only assign weights to valid (non-padded) samples
            valid_samples = end_idx - start_idx
            curriculum_weights[start_idx:end_idx] = combined_metric.detach()[:valid_samples]

            # Log progress
            print(
                f"Processed metrics for {batch_idx + 1}/{len(metric_futures_with_batch_idx)} batches"
            )

        print(f"DEBUG: succesful! curriculum weights shape: {curriculum_weights.shape}")

        # Normalize weights using min-max scaling
        min_weight = curriculum_weights.min()
        max_weight = curriculum_weights.max()
        curriculum_weights = (curriculum_weights - min_weight) / (
            max_weight
            - min_weight
            + 1e-8  # Add small epsilon to avoid division by zero
        )

        return curriculum_weights

    def _update_curriculum_weights(self) -> None:
        """Update curriculum learning weights based on current model performance."""
        print(f"Updating curriculum weights at step {self.global_step}...")
        # Create a temporary dataloader for weight estimation
        new_weights = self._compute_curriculum_weights()

        # Update weights with configured momentum
        momentum = self.config.data.curriculum_momentum

        if getattr(self.train_dataset, "curriculum_weights", None) is None:
            self.train_dataset.curriculum_weights = new_weights
        else:
            self.train_dataset.curriculum_weights = (
                momentum * self.train_dataset.curriculum_weights
                + (1 - momentum) * new_weights
            )

        # Save updated weights
        self._save_curriculum_weights(
            self.train_dataset.curriculum_weights, step=self.global_step
        )

        # self.curriculum_sampler.update_weights(self.train_dataset.curriculum_weights)

        # Update dataloader iterator for current epoch
        # self.dataloader_iterator = iter(self.train_dataloader)
    

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_scores = [], [], []
        reward_metrics_lst = defaultdict(list)
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=[
                        "raw_prompt_ids",
                        "multi_modal_data",
                        "multi_modal_inputs",
                    ],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(
                test_gen_batch
            )
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {
            f"val/{key}_reward": value
            for key, value in reduce_metrics(reward_metrics_lst).items()
        }
        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.worker,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.worker,
                role="critic",
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.worker,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel],
                config=self.config.worker,
                role="reward",
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the reference of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # Initialize fault tolerance monitoring
        self.actor_failure_counts = defaultdict(int)
        self.actor_restart_counts = defaultdict(int)

    def _handle_actor_error(
        self, error: ray.exceptions.RayActorError, actor_type: str
    ) -> None:
        """Handle actor death and restart scenarios"""
        import logging

        logger = logging.getLogger(__name__)

        logger.warning(f"{actor_type} actor died: {error}")

        # Track failure metrics
        self.actor_failure_counts[actor_type] += 1

        # Log metrics if logger is available
        if hasattr(self, "logger") and self.logger is not None:
            try:
                metrics = {
                    f"actor_failures/{actor_type}": 1,
                    f"actor_failures/total": 1,
                    f"actor_failures/{actor_type}_cumulative": self.actor_failure_counts[
                        actor_type
                    ],
                }
                self.logger.log(metrics, step=getattr(self, "global_step", 0))
            except Exception as e:
                logger.warning(f"Failed to log actor failure metrics: {e}")

    def _safe_actor_call(
        self, actor_method_ref, actor_type: str = "unknown", timeout: float = 300.0
    ):
        """Safely call actor method with error handling and timeout"""
        import logging

        logger = logging.getLogger(__name__)

        try:
            return ray.get(actor_method_ref, timeout=timeout)
        except ray.exceptions.RayActorError as e:
            self._handle_actor_error(e, actor_type)
            raise
        except ray.exceptions.GetTimeoutError as e:
            logger.warning(f"Actor call timeout for {actor_type}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in actor call for {actor_type}: {e}")
            raise

    def _check_actor_health(self) -> dict[str, str]:
        """Check health of all actors"""
        import logging

        logger = logging.getLogger(__name__)

        health_status = {}

        try:
            # Check actor_rollout health
            if hasattr(self, "actor_rollout_wg") and self.actor_rollout_wg is not None:
                try:
                    # Try a simple health check - accessing the worker group should work
                    # If actors are healthy, this should not raise an exception
                    world_size = self.actor_rollout_wg.world_size
                    health_status["actor_rollout"] = "healthy"
                except Exception as e:
                    health_status["actor_rollout"] = f"unhealthy: {str(e)}"
                    logger.warning(f"Actor rollout health check failed: {e}")

            # Check critic health
            if hasattr(self, "critic_wg") and self.critic_wg is not None:
                try:
                    world_size = self.critic_wg.world_size
                    health_status["critic"] = "healthy"
                except Exception as e:
                    health_status["critic"] = f"unhealthy: {str(e)}"
                    logger.warning(f"Critic health check failed: {e}")

            # Check reference policy health
            if hasattr(self, "ref_policy_wg") and self.ref_policy_wg is not None:
                try:
                    world_size = self.ref_policy_wg.world_size
                    health_status["ref_policy"] = "healthy"
                except Exception as e:
                    health_status["ref_policy"] = f"unhealthy: {str(e)}"
                    logger.warning(f"Reference policy health check failed: {e}")

        except Exception as e:
            logger.error(f"Error during actor health check: {e}")
            health_status["health_check"] = f"error: {str(e)}"

        return health_status

    def _log_actor_restart(self, actor_type: str) -> None:
        """Log actor restart events"""
        import logging

        logger = logging.getLogger(__name__)

        self.actor_restart_counts[actor_type] += 1
        logger.info(
            f"Actor {actor_type} restarted (restart count: {self.actor_restart_counts[actor_type]})"
        )

        # Log metrics if logger is available
        if hasattr(self, "logger") and self.logger is not None:
            try:
                metrics = {
                    f"actor_restarts/{actor_type}": 1,
                    f"actor_restarts/total": 1,
                    f"actor_restarts/{actor_type}_cumulative": self.actor_restart_counts[
                        actor_type
                    ],
                }
                self.logger.log(metrics, step=getattr(self, "global_step", 0))
            except Exception as e:
                logger.warning(f"Failed to log actor restart metrics: {e}")

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(
            self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}"
        )
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        # Save curriculum learning state if using curriculum strategy
        if self.config.data.sampling_strategy == "curriculum" and hasattr(
            self, "curriculum_sampler"
        ):
            curriculum_state = {
                "curriculum_weights": self.train_dataset.curriculum_weights,
                "sampler_state": self.curriculum_sampler.state_dict(),
            }
            curriculum_path = os.path.join(folder_path, "curriculum_state.pt")
            torch.save(curriculum_state, curriculum_path)
            print(f"Saved curriculum state to {curriculum_path}")

        last_global_step_path = os.path.join(
            self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER
        )
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _get_checkpoint_path(self) -> str:
        if self.config.trainer.load_checkpoint_path is not None:
            return None
        if not os.path.exists(self.config.trainer.save_checkpoint_path):
            print(
                f"No checkpoint found at {self.config.trainer.save_checkpoint_path}, will start from scratch."
            )
            return None

        ckpt_list = [
            _
            for _ in os.listdir(self.config.trainer.save_checkpoint_path)
            if _.startswith("global_step_")
        ]
        if len(ckpt_list) == 0:
            print(
                f"No checkpoint found at {self.config.trainer.save_checkpoint_path}, will start from scratch."
            )
            return None

        ckpt_list.sort(key=lambda x: int(x.split("global_step_")[-1]))
        return os.path.join(self.config.trainer.save_checkpoint_path, ckpt_list[-1])

    def _load_curriculum_state(self) -> bool:
        """Load curriculum state from checkpoint, set self.curriculum_weights and sampler state
        Must be called after self.train_dataset, self.curriculum_sampler is initialized
        """
        if self.checkpoint_path is None:
            return None

        curriculum_path = os.path.join(self.checkpoint_path, "curriculum_state.pt")
        if os.path.exists(curriculum_path):
            curriculum_state = torch.load(curriculum_path, weights_only=False)
            return curriculum_state
        else:
            print(
                f"No curriculum state found at {curriculum_path}, will initialize from scratch."
            )
            return None

    def _load_dataloader_state(self) -> None:
        if self.checkpoint_path is None:
            return

        dataloader_path = os.path.join(self.checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f"No dataloader state found at {dataloader_path}, will start from scratch."
            )

    def _load_worker_state(self) -> None:
        if self.checkpoint_path is None:
            return

        print(f"Load from checkpoint: {self.checkpoint_path}.")
        self.global_step = int(
            self.checkpoint_path.strip(os.path.sep).split("global_step_")[-1]
        )
        print(f"Set global_step to {self.global_step}")

        actor_path = os.path.join(self.checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

    def _balance_batch(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        logging_prefix: str = "global_seqlen",
    ) -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # breakpoint()
        val_metrics: dict[str, Any] | None = None

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for epoch in tqdm(
            range(self.config.trainer.total_episodes), desc="Episode", position=0
        ):
            # Reset cached indices to ensure new indices are generated with the correct random position
            if self.config.data.sampling_strategy == "curriculum":
                self.curriculum_sampler.cached_mixed_indices = None
                print(
                    f"Reset cached indices for epoch {epoch}, random position: {self.curriculum_sampler.get_training_random_position()}"
                )

            # Create a new iterator for each epoch
            self.dataloader_iterator = iter(self.train_dataloader)

            # Loop until we've processed all batches or need to refresh the iterator
            while True:
                try:
                    batch_dict = next(self.dataloader_iterator)

                    self.global_step += 1
                    if self.global_step > self.training_steps:
                        break

                    metrics, timing_raw = {}, {}
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    # breakpoint()
                    # pop those keys for generation
                    if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                        gen_batch = batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=[
                                "raw_prompt_ids",
                                "multi_modal_data",
                                "multi_modal_inputs",
                            ],
                        )
                    else:
                        gen_batch = batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids"],
                        )

                    with _timer("step", timing_raw):
                        # generate a batch
                        with _timer("gen", timing_raw):  # wg: worker group
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                gen_batch
                            )

                        if self.config.algorithm.adv_estimator == "remax":
                            with _timer("gen_max", timing_raw):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["temperature"] = 0.0
                                gen_baseline_output = (
                                    self.actor_rollout_wg.generate_sequences(
                                        gen_baseline_batch
                                    )
                                )

                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor, _ = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(
                                    dim=-1
                                )

                                batch.pop(
                                    batch_keys=list(gen_baseline_output.batch.keys())
                                )
                                batch.batch["reward_baselines"] = reward_baseline_tensor
                                del gen_baseline_batch, gen_baseline_output

                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                            dtype=object,
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(
                            repeat_times=self.config.worker.rollout.n, interleave=True
                        )
                        batch = batch.union(gen_batch_output)

                        # compute reward
                        with _timer("reward", timing_raw):
                            if self.use_reward_model:
                                raise NotImplementedError(
                                    "Reward model is not supported yet."
                                )

                            # we combine with rule-based rm
                            reward_tensor, reward_metrics = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                            reward_metrics = {
                                f"reward/{key}": value
                                for key, value in reduce_metrics(reward_metrics).items()
                            }
                            metrics.update(reward_metrics)

                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        # recompute old_log_probs
                        with _timer("old", timing_raw):
                            old_log_probs = self.actor_rollout_wg.compute_log_probs(
                                batch
                            )
                            batch = batch.union(old_log_probs)

                        # compute ref_log_probs
                        if self.use_reference_policy:
                            with _timer("ref", timing_raw):
                                ref_log_probs = (
                                    self.ref_policy_wg.compute_ref_log_probs(batch)
                                )
                                batch = batch.union(ref_log_probs)

                        # compute values
                        if self.use_critic:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with _timer("adv", timing_raw):
                            # apply kl penalty if available
                            if (
                                not self.config.algorithm.use_kl_loss
                                and self.use_reference_policy
                            ):  # apply kl penalty to reward
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch[
                                    "token_level_scores"
                                ]

                            # compute advantages, executed on the driver process
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )

                        # update critic
                        if self.use_critic:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)

                            critic_metrics = reduce_metrics(
                                critic_output.non_tensor_batch
                            )
                            metrics.update(critic_metrics)

                        # update actor
                        if self.config.trainer.critic_warmup <= self.global_step:
                            with _timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)

                            actor_metrics = reduce_metrics(
                                actor_output.non_tensor_batch
                            )
                            metrics.update(actor_metrics)

                        # validate
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.val_freq > 0
                            and self.global_step % self.config.trainer.val_freq == 0
                        ):
                            with _timer("validation", timing_raw):
                                val_metrics = self._validate()

                            metrics.update(val_metrics)

                        if (
                            self.config.trainer.save_freq > 0
                            and self.global_step % self.config.trainer.save_freq == 0
                        ):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                        # Update curriculum weights if configured for step-level updates
                        if (
                            self.config.data.sampling_strategy == "curriculum"
                            and self.config.data.curriculum_update_freq > 0
                            and self.global_step
                            % self.config.data.curriculum_update_freq
                            == 0
                        ):
                            with _timer("update_curriculum", timing_raw):
                                # Update Curriculum Weights to self.train_dataset.curriculum_weights
                                self._update_curriculum_weights()

                                # Replace Internal Sampler of Curriculum Sampler with the new weights
                                self.curriculum_sampler.update_weights(self.train_dataset.curriculum_weights)

                                # Refresh Dataloader Iterator
                                self.dataloader_iterator = iter(self.train_dataloader)

                                # Log curriculum learning metrics
                                curriculum_metrics = {
                                    "curriculum/mean_weight": self.train_dataset.curriculum_weights.mean().item(),
                                    "curriculum/std_weight": self.train_dataset.curriculum_weights.std().item(),
                                    "curriculum/min_weight": self.train_dataset.curriculum_weights.min().item(),
                                    "curriculum/max_weight": self.train_dataset.curriculum_weights.max().item(),
                                    "curriculum/consumed_batches": self.curriculum_sampler.consumed_batches,
                                    "curriculum/random_position": self.curriculum_sampler.get_training_random_position(),
                                }
                                metrics.update(curriculum_metrics)

                    # collect metrics
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(
                        compute_data_metrics(batch=batch, use_critic=self.use_critic)
                    )
                    metrics.update(
                        compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                    )
                    metrics.update(
                        compute_throughout_metrics(
                            batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                        )
                    )

                    self.logger.log(data=metrics, step=self.global_step)

                    # Periodic health monitoring
                    if (
                        self.config.fault_tolerance.enable_health_monitoring
                        and self.config.fault_tolerance.health_check_interval > 0
                        and self.global_step
                        % max(1, int(self.config.fault_tolerance.health_check_interval))
                        == 0
                    ):
                        health_status = self._check_actor_health()
                        health_metrics = {
                            f"actor_health/{actor_type}": (
                                1 if status == "healthy" else 0
                            )
                            for actor_type, status in health_status.items()
                            if not status.startswith("error")
                        }
                        if health_metrics:
                            self.logger.log(data=health_metrics, step=self.global_step)

                    # Update the random indices position to properly track the samples seen
                    if self.config.data.sampling_strategy == "curriculum":
                        self.curriculum_sampler.update_position_after_batch()

                except StopIteration:
                    # We've reached the end of the current iterator
                    break

            # Exit the epoch loop if we've exceeded training steps
            if self.global_step > self.training_steps:
                break

            # Update curriculum weights at the end of each epoch if configured for epoch-level updates
            if (
                self.config.data.sampling_strategy == "curriculum"
                and self.config.data.curriculum_update_freq == 0
            ):
                with _timer("update_curriculum", timing_raw):
                    # Update Curriculum Weights to self.train_dataset.curriculum_weights
                    self._update_curriculum_weights()

                    # Replace Internal Sampler of Curriculum Sampler with the new weights
                    self.curriculum_sampler.update_weights(self.train_dataset.curriculum_weights)

                    # Refresh Dataloader Iterator
                    self.dataloader_iterator = iter(self.train_dataloader)

                    # Log curriculum learning metrics
                    curriculum_metrics = {
                        "curriculum/mean_weight": self.train_dataset.curriculum_weights.mean().item(),
                        "curriculum/std_weight": self.train_dataset.curriculum_weights.std().item(),
                        "curriculum/min_weight": self.train_dataset.curriculum_weights.min().item(),
                        "curriculum/max_weight": self.train_dataset.curriculum_weights.max().item(),
                        "curriculum/consumed_batches": self.curriculum_sampler.consumed_batches,
                        "curriculum/random_position": self.curriculum_sampler.get_training_random_position(),
                    }
                    self.logger.log(data=curriculum_metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if (
            self.config.trainer.save_freq <= 0
            or self.global_step % self.config.trainer.save_freq != 0
        ):
            self._save_checkpoint()
