#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune
def counting1(tokenizer: ModelTokenizer, packed: bool = True):
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
        data_files='counting_chars_common_inputoutput1_train.json'
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds
    
def counting2(tokenizer: ModelTokenizer, packed: bool = True):
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
        data_files='counting_chars_common_inputoutput2_train.json'
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds
    
def counting3(tokenizer: ModelTokenizer, packed: bool = True):
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
        data_files='counting_chars_common_inputoutput3_train.json'
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds
    
def counting4(tokenizer: ModelTokenizer, packed: bool = True):
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
        data_files='counting_chars_common_inputoutput4_train.json'
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds
    
def counting5(tokenizer: ModelTokenizer, packed: bool = True):
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
        data_files='counting_chars_common_inputoutput5_train.json'
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds
    
def counting6(tokenizer: ModelTokenizer, packed: bool = True):
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
        data_files='counting_chars_common_inputoutput6_train.json'
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds
    
def counting_extrains(tokenizer: ModelTokenizer, packed: bool = True):
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
        data_files='counting_chars_common_extrains_inputoutput1_train.json'
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds
    