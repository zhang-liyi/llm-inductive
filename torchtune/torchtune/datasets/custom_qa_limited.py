#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune

train_filelist = ['hate_speech_offensive',
 'google_wellformed_query',
 'glue-sst2',
 'scitail',
 'ag_news',
 'art',
 'paws',
 'glue-qnli',
 'ade_corpus_v2-classification',
 'hatexplain',
 'glue-qqp',
 'kilt_fever',
 'glue-mnli',
 'tab_fact',
 'tweet_eval-offensive',
 'imdb',
 'anli',
 'yelp_polarity']
test_filelist = ['codah',
 'cosmos_qa',
 'dream',
 'hellaswag',
 'openbookqa',
 'quarel',
 'quartz-no_knowledge',
 'quartz-with_knowledge',
 'sciq',
 'swag',
 'wino_grande',
 'wiqa']

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
            data_files=f'../data/{f}_ins_inputoutput_train_30.json'
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
            data_files=f'../data/{f}_ins_inputoutput_dev.json'
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
            data_files=f'../data/{f}_ins_inputoutput_test.json'
        )
        ds_list_test.append(ds_test)
    
    ds = [ds_list_train, ds_list_dev, ds_list_test]

    if packed:
        return 
    else:
        return ds
    