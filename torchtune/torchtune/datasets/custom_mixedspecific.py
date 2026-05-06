#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune

train_filelist = []
train_filelist.extend(['sick', 'scitail', 'glue-rte', 'anli', 'superglue-cb', 'glue-qnli'])
train_filelist.extend(["glue-mrpc", "paws"])
train_filelist.extend(['codah',
 'cosmos_qa',
 'dream',
 'hellaswag',
 'openbookqa',
 'quartz-no_knowledge',
 'quartz-with_knowledge',
 'sciq',
 'swag',
 'wiqa'])
test_filelist = ['glue-mnli', 'glue-wnli', "glue-qqp", "medical_questions_pairs", 'wino_grande', 'quarel']

def mc(tokenizer: ModelTokenizer, packed: bool = True):
    """
    Custom dataset for the counting task. Instruct and code response pairs.
    """

    ds_list_train, ds_list_dev, ds_list_test = [], [], []

    for f in train_filelist:
        ds_train = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_inputoutput_train.json'
        )
        ds_list_train.append(ds_train)

    for f in test_filelist:
        ds_dev = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_inputoutput_dev.json'
        )
        ds_list_dev.append(ds_dev)
        ds_test = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_inputoutput_test.json'
        )
        ds_list_test.append(ds_test)
    
    ds = [ds_list_train, ds_list_dev, ds_list_test]

    if packed:
        return 
    else:
        return ds
    