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
    Uses ICL (in-context learning) versions for dev and test splits.
    """

    ds_list_train, ds_list_dev_wino, ds_list_dev_cls45, ds_list_test_wino, ds_list_test_cls45 = [], [], [], [], []
    filelist = ['superglue-rte', 'tweet_eval-sentiment', 'glue-rte', 'superglue-wsc', 'glue-mrpc', 'tweet_eval-stance_hillary', 'tweet_eval-offensive', 'hatexplain', 'glue-cola', 'sick', 'paws', 'ethos-sexual_orientation', 'glue-qqp', 'tweet_eval-emotion', 'sms_spam', 'health_fact', 'glue-mnli', 'imdb', 'ethos-disability', 'glue-wnli', 'scitail', 'glue-sst2', 'tweet_eval-stance_abortion', 'tweet_eval-stance_climate', 'glue-qnli', 'ethos-directed_vs_generalized', 'ade_corpus_v2-classification', 'hate_speech_offensive', 'superglue-wic', 'google_wellformed_query', 'tweet_eval-irony', 'ethos-gender', 'rotten_tomatoes', 'kilt_fever']
    cls45_test = ['tweet_eval-stance_feminist', 'ethos-national_origin', 'tweet_eval-hate', 'ag_news', 'anli', 'hate_speech18', 'poem_sentiment', 'climate_fever', 'medical_questions_pairs', 'tweet_eval-stance_atheism', 'ethos-race', 'ethos-religion', 'superglue-cb', 'wiki_qa', 'yelp_polarity']
    mcqa_test = ['codah', 'cosmos_qa', 'dream', 'hellaswag', 'openbookqa', 'quarel', 'quartz-no_knowledge', 'quartz-with_knowledge', 'sciq', 'swag', 'wino_grande', 'wiqa']

    for f in filelist:
        ds_train = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_ins_inputoutput_train.json'
        )
        ds_list_train.append(ds_train)
    # for f in filelist:
    #     ds_dev = SFTDataset(
    #         model_transform=tokenizer,
    #         source="json",
    #         message_transform=InputOutputToMessages(
    #             column_map={"input": "input", "output": "output"},
    #         ),
    #         #filter_fn=lambda x: x["language"] == "python",
    #         filter_fn=None,
    #         split="train",
    #         data_files=f'../data/{f}_inputoutput_dev.json'
    #     )
    #     ds_list_dev.append(ds_dev)
    # for f in filelist:
    #     ds_test = SFTDataset(
    #         model_transform=tokenizer,
    #         source="json",
    #         message_transform=InputOutputToMessages(
    #             column_map={"input": "input", "output": "output"},
    #         ),
    #         #filter_fn=lambda x: x["language"] == "python",
    #         filter_fn=None,
    #         split="train",
    #         data_files=f'../data/{f}_inputoutput_test.json'
    #     )
    #     ds_list_test.append(ds_test)
    # for f in mcqa_test:
    #     ds = SFTDataset(
    #         model_transform=tokenizer,
    #         source="json",
    #         message_transform=InputOutputToMessages(
    #             column_map={"input": "input", "output": "output"},
    #         ),
    #         #filter_fn=lambda x: x["language"] == "python",
    #         filter_fn=None,
    #         split="train",
    #         data_files=f'../data/{f}_inputoutput_dev.json'
    #     )
    #     ds_list_dev_wino.append(ds)

    # for f in mcqa_test:
    #     ds = SFTDataset(
    #         model_transform=tokenizer,
    #         source="json",
    #         message_transform=InputOutputToMessages(
    #             column_map={"input": "input", "output": "output"},
    #         ),
    #         #filter_fn=lambda x: x["language"] == "python",
    #         filter_fn=None,
    #         split="train",
    #         data_files=f'../data/{f}_inputoutput_test.json'
    #     )
    #     ds_list_test_wino.append(ds)
    ds_list_dev_wino.append(SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/winogrande/winogrande_m_inputoutput_train.json'
        ))
    ds_list_test_wino.append(SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/winogrande/winogrande_m_inputoutput_val.json'
        ))
    ds_list_dev_wino.append(SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/openbookqa_inputoutput_dev.json'
        ))
    ds_list_test_wino.append(SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/openbookqa_inputoutput_test.json'
        ))
    for f in cls45_test:
        ds_dev = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_ins_inputoutput_icl_dev.json'
        )
        ds_list_dev_cls45.append(ds_dev)
        ds_test = SFTDataset(
            model_transform=tokenizer,
            source="json",
            message_transform=InputOutputToMessages(
                column_map={"input": "input", "output": "output"},
            ),
            #filter_fn=lambda x: x["language"] == "python",
            filter_fn=None,
            split="train",
            data_files=f'../data/{f}_ins_inputoutput_icl_test.json'
        )
        ds_list_test_cls45.append(ds_test)
    ds = [ds_list_train, ds_list_dev_wino, ds_list_dev_cls45, ds_list_test_wino, ds_list_test_cls45]

    if packed:
        return
    else:
        return ds

