#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune
def acronym(tokenizer: ModelTokenizer, packed: bool = True):
    """
    Custom dataset for the counting task. Instruct and code response pairs.
    """
    ds = SFTDataset(
        model_transform=tokenizer,
        source="json",
        message_transform=InputOutputToMessages(
            column_map={"input": "input", "output": "output"},
        ),
        #filter_fn=lambda x: x["language"] == "python",
        filter_fn=None,
        split="train",
        data_files='acronym_inputoutput_train.json'
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds
    