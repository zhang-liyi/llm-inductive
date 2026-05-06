#
# Dataset builder: train on single_scenario, val/test on probabilistic_reasoning.
#
# Use this dataset config to fine-tune on the single-scenario dataset (standard CE
# loss, no ground-truth distributions) while evaluating against the probabilistic
# reasoning val/test splits (which do carry ground-truth distributions).
#
from torchtune.modules.tokenizers import ModelTokenizer
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from probabilistic_reasoning_utils import SingleScenarioDataset, ProbabilisticReasoningDataset


def data(tokenizer: ModelTokenizer, packed: bool = False, max_seq_length: int = 2048):
    """
    Dataset builder that trains on single_scenario and evaluates on probabilistic_reasoning.

    Training:   data/single_scenario_dataset.json  (no bins — standard CE loss)
    Validation: data/probabilistic_reasoning_val.json  (with bins)
    Test:       data/probabilistic_reasoning_test.json (with bins)

    Args:
        tokenizer: Model tokenizer
        packed: Unused (not implemented)
        max_seq_length: Maximum sequence length (default: 2048)

    Returns:
        [ds_list_train, ds_list_dev, ds_list_test]
    """
    ds_train = SingleScenarioDataset(
        data_path='data/single_scenario_dataset.json',
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    ds_dev = ProbabilisticReasoningDataset(
        data_path='data/probabilistic_reasoning_val.json',
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    ds_test = ProbabilisticReasoningDataset(
        data_path='data/probabilistic_reasoning_test.json',
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    if packed:
        return None  # Packed mode not implemented

    return [[ds_train], [ds_dev], [ds_test]]
