#
# Dataset builder for probabilistic reasoning with ground truth distributions
#
from torchtune.modules.tokenizers import ModelTokenizer
import sys
import os

# Add parent directory to path to import ProbabilisticReasoningDataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from probabilistic_reasoning_utils import ProbabilisticReasoningDataset


def data(tokenizer: ModelTokenizer, packed: bool = False, max_seq_length: int = 2048):
    """
    Dataset builder for probabilistic reasoning task.

    Uses ProbabilisticReasoningDataset which includes:
    - tokens: Tokenized input + output
    - labels: Labels with input masked (-100)
    - ground_truth_bins: Posterior distributions [num_queries, 101]

    Args:
        tokenizer: Model tokenizer
        packed: Whether to pack sequences (not implemented, returns None)
        max_seq_length: Maximum sequence length (default: 2048)

    Returns:
        [ds_list_train, ds_list_dev, ds_list_test] where each is a list with one dataset
    """

    ds_list_train, ds_list_dev, ds_list_test = [], [], []

    # Training dataset
    ds_train = ProbabilisticReasoningDataset(
        data_path='data/probabilistic_reasoning_train.json',
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    ds_list_train.append(ds_train)

    # Validation/dev dataset
    ds_dev = ProbabilisticReasoningDataset(
        data_path='data/probabilistic_reasoning_val.json',
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    ds_list_dev.append(ds_dev)

    # Test dataset
    ds_test = ProbabilisticReasoningDataset(
        data_path='data/probabilistic_reasoning_test.json',
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    ds_list_test.append(ds_test)

    ds = [ds_list_train, ds_list_dev, ds_list_test]

    if packed:
        return None  # Packed mode not implemented
    else:
        return ds
    