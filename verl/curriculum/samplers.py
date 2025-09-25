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

from typing import List, Optional
import torch
from torch.utils.data import WeightedRandomSampler


class CurriculumWeightedSampler(WeightedRandomSampler):
    """Custom weighted sampler that supports dynamic weight updates."""

    def __init__(self, weights, num_samples, replacement=True, generator=None):
        super().__init__(weights, num_samples, replacement, generator)
        self.weights = weights

    def update_weights(self, new_weights):
        """Update the sampling weights."""
        self.weights = new_weights


class MixedCurriculumSampler(torch.utils.data.Sampler):
    """Custom sampler that mixes random and weighted sampling."""

    def __init__(
        self,
        dataset,
        weights,
        batch_size,
        mixture_ratio=0.5,
        replacement=True,
        generator=None,
    ):
        self.dataset = dataset
        self.weights = weights
        self.batch_size = batch_size
        self.mixture_ratio = mixture_ratio
        self.replacement = replacement
        self.generator = generator
        self.dataset_size = len(dataset)

        # Calculate number of samples of each type per batch
        self.n_weighted_per_batch = int(batch_size * mixture_ratio)
        self.n_random_per_batch = batch_size - self.n_weighted_per_batch

        # Track total number of random indices needed for the entire dataset
        self.total_random_indices_needed = (
            self.dataset_size // self.batch_size
        ) * self.n_random_per_batch
        if self.dataset_size % self.batch_size > 0:
            self.total_random_indices_needed += self.dataset_size % self.batch_size

        # Create weighted sampler for the entire dataset
        self.weighted_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=self.dataset_size,  # Sample as many as the dataset size
            replacement=replacement,
            generator=generator,
        )

        # Initialize random indices pool and tracking
        self._init_random_indices()

        # Cache for mixed indices to handle multiple calls to __iter__
        self.cached_mixed_indices = None

        # Separate tracking variable for actual consumed batches in training
        self.consumed_batches = 0

    def _init_random_indices(self):
        """Initialize random indices - creating a pool of random indices for the dataset."""
        # Create a random permutation of all indices
        if self.generator is not None:
            rand_indices = torch.randperm(
                len(self.dataset), generator=self.generator
            ).tolist()
        else:
            import random

            rand_indices = list(range(len(self.dataset)))
            random.shuffle(rand_indices)

        # Store the entire random permutation
        self.random_indices_pool = rand_indices
        # Position tracker for random indices used for initial index creation
        self.random_indices_position = 0

    def _get_next_random_indices(self, n):
        """Get next n random indices from our pre-shuffled pool, cycling if needed."""
        # If we're near the end of our pool, we need to cycle back
        if self.random_indices_position + n > len(self.random_indices_pool):
            # Just wrap around to the beginning without reshuffling
            self.random_indices_position = 0

        # Get the indices
        indices = self.random_indices_pool[
            self.random_indices_position : self.random_indices_position + n
        ]
        self.random_indices_position += n
        return indices

    def get_training_random_position(self):
        """Get the current position in random indices used for actual training.
        This is separate from the position used during index generation.
        """
        # Simply use modulo to handle wrap-around consistently with _get_next_random_indices
        return (self.consumed_batches * self.n_random_per_batch) % len(
            self.random_indices_pool
        )

    def update_position_after_batch(self):
        """Update tracking of consumed batches after each batch is processed in training.
        This should be called after each batch is processed in the training loop.
        """
        # Increment the number of consumed batches
        self.consumed_batches += 1

    def __iter__(self):
        # Only compute indices if they haven't been cached
        if self.cached_mixed_indices is None:
            # Get all indices from weighted sampler
            weighted_indices = list(self.weighted_sampler)

            # Shuffle the weighted indices
            if self.generator is not None:
                rand_tensor = torch.randperm(
                    len(weighted_indices), generator=self.generator
                )
                weighted_indices = [weighted_indices[i] for i in rand_tensor]
            else:
                import random

                random.shuffle(weighted_indices)

            # Create batches that maintain the mixture ratio
            mixed_indices = []
            num_batches = self.dataset_size // self.batch_size

            # Synchronize the position counter with the actual training position
            # This ensures we continue from where training left off rather than restarting
            self.random_indices_position = self.get_training_random_position()
            print(
                f"Starting index generation with random_indices_position: {self.random_indices_position}"
            )

            for batch_idx in range(num_batches):
                # Calculate start indices for weighted part
                w_start = batch_idx * self.n_weighted_per_batch

                # Get indices for this batch
                batch_weighted = weighted_indices[
                    w_start : w_start + self.n_weighted_per_batch
                ]
                # Get random indices from our tracked pool
                batch_random = self._get_next_random_indices(self.n_random_per_batch)

                # Interleave the indices to maintain diversity within the batch
                batch_indices = []
                for i in range(max(len(batch_weighted), len(batch_random))):
                    if i < len(batch_weighted):
                        batch_indices.append(batch_weighted[i])
                    if i < len(batch_random):
                        batch_indices.append(batch_random[i])

                mixed_indices.extend(batch_indices)

            # Handle remaining samples if dataset size is not divisible by batch size
            remaining = self.dataset_size % self.batch_size
            if remaining > 0:
                w_start = num_batches * self.n_weighted_per_batch

                n_remaining_weighted = int(remaining * self.mixture_ratio)
                n_remaining_random = remaining - n_remaining_weighted

                batch_weighted = weighted_indices[
                    w_start : w_start + n_remaining_weighted
                ]
                batch_random = self._get_next_random_indices(n_remaining_random)

                batch_indices = []
                for i in range(max(len(batch_weighted), len(batch_random))):
                    if i < len(batch_weighted):
                        batch_indices.append(batch_weighted[i])
                    if i < len(batch_random):
                        batch_indices.append(batch_random[i])

                mixed_indices.extend(batch_indices)

            # Store the computed indices
            self.cached_mixed_indices = mixed_indices

        # Return iterator of cached indices
        return iter(self.cached_mixed_indices)

    def __len__(self):
        return self.dataset_size

    def update_weights(self, new_weights):
        """Update the sampling weights."""
        self.weights = new_weights
        self.weighted_sampler = WeightedRandomSampler(
            weights=new_weights,
            num_samples=self.dataset_size,
            replacement=self.replacement,
            generator=self.generator,
        )
        # Reset the cached indices to force regeneration with new weights
        self.cached_mixed_indices = None
        # Note: We don't reset random indices - this preserves their state between updates

    def state_dict(self):
        """Return the state dictionary for checkpointing."""
        return {
            "consumed_batches": self.consumed_batches,
            "random_indices_pool": self.random_indices_pool,
            "random_indices_position": self.random_indices_position,
            "weights": (
                self.weights.clone() if torch.is_tensor(self.weights) else self.weights
            ),
            "generator_state": (
                self.generator.get_state() if self.generator is not None else None
            ),
        }

    def load_state_dict(self, state_dict):
        """Load state from a state dictionary."""
        self.consumed_batches = state_dict.get("consumed_batches", 0)
        self.random_indices_pool = state_dict.get(
            "random_indices_pool", list(range(len(self.dataset)))
        )
        self.random_indices_position = state_dict.get("random_indices_position", 0)
        self.weights = state_dict.get("weights", self.weights)

        # Restore generator state if available
        if state_dict.get("generator_state") is not None and self.generator is not None:
            try:
                self.generator.set_state(state_dict["generator_state"])
            except Exception as e:
                print(f"Warning: Could not restore generator state: {e}")

        # Update the weighted sampler with restored weights
        self.weighted_sampler = WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.dataset_size,
            replacement=self.replacement,
            generator=self.generator,
        )

        # Reset cached indices to force regeneration with restored state
        self.cached_mixed_indices = None

        print(
            f"Restored curriculum sampler state: consumed_batches={self.consumed_batches}, "
            f"random_indices_position={self.random_indices_position}"
        )


class PaddedSequentialSampler(torch.utils.data.BatchSampler):
    """Custom sampler that pads each batch to the same batch size."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        pad_to_length = (len(dataset) + batch_size - 1) // batch_size * batch_size
        self.indices = list(range(pad_to_length))
        self.indices[len(dataset) :] = [len(dataset) - 1] * (
            pad_to_length - len(dataset)
        )

    def __iter__(self):
        """Generate padded batches of the same size."""
        for i in self.indices:
            yield i

    def __len__(self):
        return len(self.indices)