#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune

train_filelist = ['ade_corpus_v2-classification',
 'ag_news',
 'climate_fever',
 'ethos-directed_vs_generalized',
 'ethos-disability',
 'ethos-gender',
 'ethos-national_origin',
 'ethos-race',
 'ethos-religion',
 'ethos-sexual_orientation',
 'glue-cola',
 'glue-mrpc',
 'glue-qqp',
 'glue-sst2',
 'google_wellformed_query',
 'hate_speech18',
 'hate_speech_offensive',
 'hatexplain',
 'health_fact',
 'imdb',
 'kilt_fever',
 'medical_questions_pairs',
 'paws',
 'poem_sentiment',
 'rotten_tomatoes',
 'sick',
 'sms_spam',
 'superglue-wic',
 'superglue-wsc',
 'tab_fact',
 'tweet_eval-emotion',
 'tweet_eval-hate',
 'tweet_eval-irony',
 'tweet_eval-offensive',
 'tweet_eval-sentiment',
 'tweet_eval-stance_abortion',
 'tweet_eval-stance_atheism',
 'tweet_eval-stance_climate',
 'tweet_eval-stance_feminist',
 'tweet_eval-stance_hillary',
 'wiki_qa',
 'yelp_polarity']
test_filelist = ['sick', 'glue-mnli', 'glue-wnli', 'scitail', 'glue-rte', 'anli', 'superglue-cb', 'glue-qnli']

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
    