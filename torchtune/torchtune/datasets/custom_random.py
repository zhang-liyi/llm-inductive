#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune

train_filelist = ['glue-mrpc',
 'quarel',
 'tweet_eval-stance_atheism',
 'tab_fact',
 'glue-wnli',
 'codah',
 'tweet_eval-offensive',
 'wiki_qa',
 'openbookqa',
 'sms_spam',
 'ethos-national_origin',
 'hellaswag',
 'superglue-wsc',
 'blimp-ellipsis_n_bar_2',
 'google_wellformed_query',
 'wiqa',
 'tweet_eval-stance_abortion',
 'ethos-religion',
 'ethos-race',
 'glue-qqp',
 'paws',
 'ethos-directed_vs_generalized',
 'glue-sst2',
 'tweet_eval-hate',
 'glue-rte',
 'blimp-anaphor_number_agreement',
 'hate_speech_offensive',
 'superglue-wic',
 'boolq',
 'quartz-no_knowledge',
 'sick',
 'tweet_eval-stance_climate',
 'tweet_eval-sentiment',
 'crows_pairs',
 'glue-mnli',
 'medical_questions_pairs',
 'imdb',
 'ethos-gender',
 'swag',
 'scitail',
 'tweet_eval-stance_feminist',
 'social_i_qa',
 'anli',
 'cosmos_qa',
 'race-middle',
 'sciq',
 'wino_grande',
 'rotten_tomatoes',
 'superglue-cb',
 'poem_sentiment',
 'piqa',
 'climate_fever',
 'mc_taco',
 'quartz-with_knowledge',
 'superglue-copa',
 'hate_speech18']

test_filelist = ['kilt_fever',
 'art',
 'tweet_eval-stance_hillary',
 'tweet_eval-emotion',
 'dream',
 'ade_corpus_v2-classification',
 'health_fact',
 'ethos-disability',
 'yelp_polarity',
 'superglue-rte',
 'glue-cola',
 'ethos-sexual_orientation',
 'blimp-sentential_negation_npi_scope',
 'blimp-sentential_negation_npi_licensor_present',
 'tweet_eval-irony',
 'glue-qnli',
 'hatexplain',
 'ag_news']

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
    