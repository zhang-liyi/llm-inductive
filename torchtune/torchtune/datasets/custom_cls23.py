#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune

train_filelist = ['blimp-ellipsis_n_bar_2',
 'blimp-sentential_negation_npi_scope',
 'crows_pairs',
 'hellaswag',
 'openbookqa',
 'piqa',
 'quartz-no_knowledge',
 'sciq',
 'ethos-disability',
 'ethos-sexual_orientation',
 'glue-cola',
 'glue-mnli',
 'glue-mrpc',
 'glue-qqp',
 'glue-rte',
 'glue-wnli',
 'hatexplain',
 'health_fact',
 'imdb',
 'paws',
 'sick',
 'sms_spam',
 'superglue-rte',
 'superglue-wsc',
 'tweet_eval-emotion',
 'tweet_eval-offensive',
 'tweet_eval-sentiment',
 'tweet_eval-stance_hillary']
test_filelist = ['tweet_eval-stance_feminist', 'ethos-national_origin', 'tweet_eval-hate', 'ag_news', 'anli', 'hate_speech18', 'poem_sentiment', 'climate_fever', 'medical_questions_pairs', 'tweet_eval-stance_atheism', 'ethos-race', 'ethos-religion', 'superglue-cb', 'wiki_qa', 'yelp_polarity']

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
    