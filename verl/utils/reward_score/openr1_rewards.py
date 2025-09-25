"""Reward functions for GRPO training."""

import math
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .r1v import r1v_accuracy_reward, r1v_format_reward


def accuracy_reward(content, sol, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        content = content.strip()
        if re.search(r"<answer>(.*?)</answer>", content, re.DOTALL):
            content = (
                re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                .group(1)
                .strip()
            )
        else:
            content = content
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
            reward = 0.0
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward


def get_format_reward(completion_content, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    match = re.match(pattern, completion_content, re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(
    completions: list[Dict[str, str]], solution: list[str], **kwargs
) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 4096,
):
    def cosine_scaled_reward(content, sol, acc_reward, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        # gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        # if len(gold_parsed) == 0:
        #     reward = 1.0  # Treat as correct to avoid penalizing
        #     print("Failed to parse gold solution: ", sol)
        #     return reward

        # answer_parsed = parse(
        #     content,
        #     extraction_config=[
        #         LatexExtractionConfig(
        #             normalization_config=NormalizationConfig(
        #                 nits=False,
        #                 malformed_operators=False,
        #                 basic_latex=True,
        #                 equations=True,
        #                 boxed=True,
        #                 units=True,
        #             ),
        #             boxed_match_priority=0,
        #             try_extract_without_anchor=False,
        #         )
        #     ],
        #     extraction_mode="first_match",
        # )

        # is_correct = verify(answer_parsed, gold_parsed)
        if acc_reward == 1.0:
            is_correct = True
        else:
            is_correct = False

        gen_len = len(content)

        # Apply cosine scaling based on length
        progress = gen_len / max_len
        cosine = math.cos(progress * math.pi)

        if is_correct:
            min_value = min_value_correct
            max_value = max_value_correct
        else:
            # Swap min/max for incorrect answers
            min_value = max_value_wrong
            max_value = min_value_wrong

        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        reward = float(reward)

        return reward

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int = 40, max_penalty: float = -0.5):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completion, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        if completion == "":
            reward = 0.0
            return reward
        if len(completion.split()) < ngram_size:
            reward = 0.0
            return reward

        ngrams = set()
        total = 0
        for ng in zipngram(completion, ngram_size):
            ngrams.add(ng)
            total += 1

        scaling = 1 - len(ngrams) / total
        reward = scaling * max_penalty

        return reward

    return repetition_penalty_reward


def openr1_compute_score(
    predict_str: str, ground_truth: str, validation: bool = False, response_length=None
) -> float:
    """Compute reward score based on the completion and ground truth.

    Args:
        predict_str: The completion string
        ground_truth: The ground truth string
    """
    acc_reward = r1v_accuracy_reward(predict_str, ground_truth, response_length)
    format_reward = r1v_format_reward(predict_str)
    cosine_len_reward = get_cosine_scaled_reward()(
        predict_str, ground_truth, acc_reward
    )
    repetition_penalty_reward = get_repetition_penalty_reward()(predict_str)
    if validation:
        reward = acc_reward if acc_reward == 1.0 else 0.0
    else:
        reward = (
            acc_reward + format_reward + cosine_len_reward + repetition_penalty_reward
        )
    return reward
