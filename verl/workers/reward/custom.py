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


from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, TypedDict

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import (
    math_compute_score,
    openr1_compute_score_batch,
    openr1_compute_score_batch_vllm,
    openr1_compute_score_batch_wo_LMM,
    r1v_compute_score,
)


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


class CustomRewardManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        compute_score: str,
        validation: bool,
        response_length: int = None,
        batch_processing: bool = False,
        cos_len_reward_config: list = None,
        provider: str = "azure",
        base_urls: list = None,
        model_name: str = None,
        api_key: str = None,
        workflow_id: str = None,
    ):
        self.tokenizer = tokenizer
        self.validation = validation
        self.response_length = response_length
        self.batch_processing = batch_processing
        self.cos_len_reward_config = cos_len_reward_config
        self.provider = provider
        self.base_urls = base_urls
        self.model_name = model_name
        self.api_key = api_key
        self.workflow_id = workflow_id
        if compute_score == "math":
            self.compute_score: Callable[[str, str], RewardScore] = math_compute_score
        elif compute_score == "r1v":
            self.compute_score: Callable[[str, str], RewardScore] = r1v_compute_score
        elif compute_score == "openr1":
            if self.batch_processing:
                self.compute_score = openr1_compute_score_batch
            else:
                # self.compute_score = openr1_compute_score
                raise NotImplementedError(
                    "openr1_compute_score is not adapted to the new channel-wise reward computation, use batch_processing if openr1_reward is needed"
                )
        elif compute_score == "openr1_wo_LMM":
            if self.batch_processing:
                self.compute_score = openr1_compute_score_batch_wo_LMM
            else:
                raise NotImplementedError(
                    "openr1_compute_score_wo_LMM is not adapted to the new channel-wise reward computation, use batch_processing if openr1_reward is needed"
                )
        elif compute_score == "openr1_vllm":
            if self.batch_processing:
                self.compute_score = openr1_compute_score_batch_vllm
            else:
                raise NotImplementedError(
                    "openr1_compute_score_vllm is not adapted to the new channel-wise reward computation, use batch_processing if openr1_reward is needed"
                )
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # breakpoint()
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)

        if not self.batch_processing:
            # Collect accuracy scores for batch-level variance
            batch_accuracy_scores = []
            
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=True
                )
                ground_truth = data_item.non_tensor_batch["ground_truth"]

                score = self.compute_score(response_str, ground_truth)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                
                # Collect accuracy for variance computation
                batch_accuracy_scores.append(score["accuracy"])
                
                for key, value in score.items():
                    reward_metrics[key].append(value)
            
            # Compute batch-level accuracy statistics
            if batch_accuracy_scores:
                reward_metrics["accuracy_variance"].append(np.var(batch_accuracy_scores))
                reward_metrics["accuracy_std"].append(np.std(batch_accuracy_scores))
                reward_metrics["accuracy_min"].append(np.min(batch_accuracy_scores))
                reward_metrics["accuracy_max"].append(np.max(batch_accuracy_scores))
            
            return reward_tensor, reward_metrics
        else:
            return self.batch_process(data, reward_tensor, reward_metrics)

    def batch_process(
        self,
        data: DataProto,
        reward_tensor: torch.Tensor,
        reward_metrics: Dict[str, float],
    ):
        # breakpoint()
        prompt_strs = []
        response_strs = []
        ground_truths = []
        valid_response_lengths = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )

            prompt_strs.append(prompt_str)
            response_strs.append(response_str)
            ground_truths.append(data_item.non_tensor_batch["ground_truth"])
            valid_response_lengths.append(valid_response_length)

        # Pass provider configuration to the compute_score function
        kwargs = {
            "validation": self.validation,
            "response_length": self.response_length,
            "cos_len_reward_config": self.cos_len_reward_config,
        }

        # Add provider settings for vLLM if needed
        if self.provider == "vllm":
            kwargs.update(
                {
                    "provider": self.provider,
                    "base_urls": self.base_urls,
                    "model_name": self.model_name,
                    "api_key": self.api_key,
                    "workflow_id": self.workflow_id,
                }
            )
        print("compure_score kwargs", kwargs)

        scores = self.compute_score(response_strs, ground_truths, prompt_strs, **kwargs)
        
        # Collect accuracy scores for batch-level variance
        batch_accuracy_scores = []
        
        for i in range(len(data)):
            reward_tensor[i, valid_response_lengths[i] - 1] = scores[i]["overall"]
            
            # Collect accuracy for variance computation
            batch_accuracy_scores.append(scores[i]["accuracy"])

            for key, value in scores[i].items():
                reward_metrics[key].append(value)
        
        # Compute batch-level accuracy statistics
        if batch_accuracy_scores:
            reward_metrics["accuracy_variance"].append(np.var(batch_accuracy_scores))
            reward_metrics["accuracy_std"].append(np.std(batch_accuracy_scores))
            reward_metrics["accuracy_min"].append(np.min(batch_accuracy_scores))
            reward_metrics["accuracy_max"].append(np.max(batch_accuracy_scores))

        return reward_tensor, reward_metrics
