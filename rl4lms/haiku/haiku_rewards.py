
from .count_syllables import count_syllables_in_haiku
from rl4lms.envs.text_generation.registry import RewardFunctionRegistry, MetricRegistry
from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from rl4lms.envs.text_generation.metric import BaseMetric
from typing import Dict, Type, Any, Union, List
from transformers import AutoTokenizer, PreTrainedModel
import torch
from collections import defaultdict
import numpy as np

def compute_meter_score(sample, max_violations=4., parseable=0.2, min_for_correct_lines=0.2, return_explanation=False):
    score = 0.
    info = {}
    # return 5 if number of lines is wrong
    info['violations_1'] = 5
    info['violations_2'] = 5
    info['violations_3'] = 5
    info['syntax_error'] = True
    explanation = ''  # could also be more detailed
    if '|' in sample:
        explanation = 'triggered phoneme mode'
    elif sample != '<syntax_error>':
        info['syntax_error'] = False
        # Is at least parseable
        score += parseable
        syllables = count_syllables_in_haiku(sample, fast=False)
        #print(syllables, sample)
        is_three_lines = len(syllables) == 3
        info['is_three_lines'] = is_three_lines
        if is_three_lines:
            # Got three lines
            score += min_for_correct_lines
            info['violations_1'] = min(abs(syllables[0]-5), 5)
            info['violations_2'] = min(abs(syllables[1]-7), 5)
            info['violations_3'] = min(abs(syllables[2]-5), 5)
            violations = abs(syllables[0]-5) + abs(syllables[1]-7) + abs(syllables[2]-5)
            if syllables[0]>5:
                explanation += 'line 0 is longer than 5,'
            elif syllables[0]<5:
                explanation += 'line 0 is shorter than 5,'
            if syllables[1]>7:
                explanation += 'line 1 is longer than 7,'
            elif syllables[1]<7:
                explanation += 'line 1 is shorter than 7,'
            if syllables[2]>5:
                explanation += 'line 2 is longer than 5,'
            elif syllables[2]<5:
                explanation += 'line 2 is shorter than 5,'
            # Total score
            score += (1-min_for_correct_lines-parseable)*(1. - min(1., violations/max_violations))
            explanation = 'perfect'
        else:
            explanation = 'number of lines is not 3'
    else:
        explanation = 'syntax error'
        info['syntax_error'] = True
    info['explanation'] = explanation
    if return_explanation:
        return score, explanation
    else:
        return score

def process_output(sample):
    # Stop at the first closing parenthesis
    parts = sample.split(')')
    if len(parts)>=1:
        tokens = parts[0].split(' = ')
        if len(tokens) == 2:
            return tokens[1]
    # in any other case return the empty string
    return '<syntax_error>'

def compute_score(sample):
    processed = process_output(sample)
    score, info = compute_meter_score(processed, return_explanation=True)
    return score, info


# Custom reward function
class HaikuMeterRewardFunction(RewardFunction):
   def __init__(self, *args) -> None:
       super().__init__()

   def __call__(self, prev_observation: Observation,
                action: int,
                current_observation: Observation,
                done: bool,
                meta_info: Dict[str, Any] = None) -> float:
        if done:
            reward, info = compute_score(current_observation.context_text)
            return reward
        return 0

RewardFunctionRegistry.add('haiku_meter', HaikuMeterRewardFunction)


class HaikuMeterMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(self,
                prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = defaultdict()
        for gen_text in generated_texts:
            # prompt is <BOS> token (unconditional generation?)
            reward, info = compute_score(gen_text)
            for k, v in info.items():
                if k != 'explanation':
                    all_rewards[k].append(v)

        metric_dict = {
            f"semantic/{k}": (v, np.mean(v)) for k, v in all_rewards
        }
        return metric_dict

MetricRegistry.add('haiku_meter', HaikuMeterMetric)
