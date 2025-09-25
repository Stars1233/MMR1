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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List, Optional, Tuple

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    val_batch_size: int = -1
    system_prompt: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    max_pixels: int = 4194304
    min_pixels: int = 262144
    sampling_strategy: str = "shuffle"  # Options: "shuffle", "sequential", "curriculum"
    curriculum_momentum: float = (
        1.0  # Only used when sampling_strategy is "curriculum", 1 means no momentum
    )
    curriculum_metrics: List[str] = field(default_factory=list)
    curriculum_metric_weights: List[float] = field(default_factory=list)
    curriculum_update_freq: int = (
        1  # Update curriculum weights every N steps (0 means epoch-level updates)
    )
    curriculum_mixture_ratio: float = (
        0.5  # Ratio of weighted samples in each batch (0.0 to 1.0)
    )
    curriculum_rollout_n: int = 8
    curriculum_rollout_batch_size: int = 8192


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "grpo"
    disable_kl: bool = False
    use_kl_loss: bool = False
    kl_penalty: str = "kl"
    kl_coef: float = 1e-3
    kl_type: str = "fixed"
    kl_horizon: float = 0.0
    kl_target: float = 0.0


@dataclass
class FaultToleranceConfig:
    """Configuration for Ray actor fault tolerance"""
    enable_fault_tolerance: bool = True
    max_restarts: int = 3  # Maximum number of times to restart an actor when it crashes
    max_task_retries: int = 2  # Maximum number of times to retry a task on actor death
    enable_health_monitoring: bool = True  # Enable actor health monitoring
    health_check_interval: float = 30.0  # Health check interval in seconds


@dataclass
class TrainerConfig:
    total_episodes: int = 10
    max_steps: Optional[int] = None
    project_name: str = "easy_r1"
    experiment_name: str = "demo"
    logger: Tuple[str] = ("console", "wandb")
    nnodes: int = 1
    n_gpus_per_node: int = 8
    critic_warmup: int = 0
    val_freq: int = -1
    val_before_train: bool = True
    val_only: bool = False
    val_generations_to_log: int = 0
    save_freq: int = -1
    save_limit: int = -1
    save_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join(
                "checkpoints", self.project_name, self.experiment_name
            )


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
