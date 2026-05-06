#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer
import os

# dev_list = ['enkavi2019adaptivenback',
#  'frey2017risk',
#  'lefebvre2017behavioural',
#  'levering2020revisiting',
#  'wulff2018sampling']

# train_list = ['bahrami2020four',
#  'enkavi2019recentprobes',
#  'gershman2018deconstructing',
#  'hebart2023things',
#  'hilbig2014generalized',
#  'ruggeri2022globalizability',
#  'steingroever2015data',
#  'wu2018generalisation',
#  'wulff2018description']

dev_list = ['enkavi2019adaptivenback',
 'frey2017risk',
 'lefebvre2017behavioural',
 'levering2020revisiting',
 'wulff2018sampling',
 'bahrami2020four',]

train_list = ['enkavi2019recentprobes',
 'gershman2018deconstructing',
 'hebart2023things',
 'hilbig2014generalized',
 'ruggeri2022globalizability',
 'steingroever2015data',
 'wu2018generalisation',
 'wulff2018description']

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune
def mc(tokenizer: ModelTokenizer, packed: bool = True):
    """
    Custom dataset for the counting task. Instruct and code response pairs.
    """

    ds_list_train, ds_list_dev, ds_list_test = [], [], []
    
    for subdir, dirs, files in os.walk('../data/psych101/'):
        for f in files:
            task = f.split('_inputoutput')[0]
            if task in train_list or task in dev_list:
                ds = SFTDataset(
                    model_transform=tokenizer,
                    source="json",
                    message_transform=InputOutputToMessages(
                        column_map={"input": "input", "output": "output"},
                    ),
                    #filter_fn=lambda x: x["language"] == "python",
                    filter_fn=None,
                    split="train",
                    data_files=f'../data/psych101/{f}'
                )
                if task in train_list and 'TRAIN' in f: 
                    ds_list_train.append(ds)
                if task in dev_list and 'TRAIN' not in f:
                    if 'DEV' in f:
                        ds_list_dev.append(ds)
                    elif 'TEST' in f:
                        ds_list_test.append(ds)

    ds_list = [ds_list_train, ds_list_dev, ds_list_test]

    if packed:
        return 
    else:
        return ds_list
    