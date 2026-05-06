#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune
def mc(tokenizer: ModelTokenizer, packed: bool = True):
    """
    Custom dataset for the counting task. Instruct and code response pairs.
    """

    ds_wino = SFTDataset(
        model_transform=tokenizer,
        source="json",
        message_transform=InputOutputToMessages(
            column_map={"input": "input", "output": "output"},
        ),
        #filter_fn=lambda x: x["language"] == "python",
        filter_fn=None,
        split="train",
        data_files='../data/winogrande_s_inputoutput_train.json'
    )
    ds_wino_v = SFTDataset(
        model_transform=tokenizer,
        source="json",
        message_transform=InputOutputToMessages(
            column_map={"input": "input", "output": "output"},
        ),
        #filter_fn=lambda x: x["language"] == "python",
        filter_fn=None,
        split="train",
        data_files='../data/winogrande_s_inputoutput_train.json'
    )

    ds_arc = SFTDataset(
        model_transform=tokenizer,
        source="json",
        message_transform=InputOutputToMessages(
            column_map={"input": "input", "output": "output"},
        ),
        #filter_fn=lambda x: x["language"] == "python",
        filter_fn=None,
        split="train",
        data_files='../data/arc_c_inputoutput_train.json'
    )
    ds_arc_v = SFTDataset(
        model_transform=tokenizer,
        source="json",
        message_transform=InputOutputToMessages(
            column_map={"input": "input", "output": "output"},
        ),
        #filter_fn=lambda x: x["language"] == "python",
        filter_fn=None,
        split="train",
        data_files='../data/arc_c_inputoutput_train.json'
    )

    ds_obqa = SFTDataset(
        model_transform=tokenizer,
        source="json",
        message_transform=InputOutputToMessages(
            column_map={"input": "input", "output": "output"},
        ),
        #filter_fn=lambda x: x["language"] == "python",
        filter_fn=None,
        split="train",
        data_files='../data/obqa_inputoutput_train.json'
    )
    ds_obqa_v = SFTDataset(
        model_transform=tokenizer,
        source="json",
        message_transform=InputOutputToMessages(
            column_map={"input": "input", "output": "output"},
        ),
        #filter_fn=lambda x: x["language"] == "python",
        filter_fn=None,
        split="train",
        data_files='../data/obqa_inputoutput_train.json'
    )

    if packed:
        return 
    else:
        return ds_wino, ds_wino_v, ds_arc, ds_arc_v, ds_obqa, ds_obqa_v
    