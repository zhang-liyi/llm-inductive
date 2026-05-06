#
# No-op cross-evaluation dataset: returns empty dev/test lists so the pyro
# recipe does not instantiate any probabilistic_reasoning WebPPL dataloader.
# Used when pyro_train_json is the single source of truth for train/val/test.
#
from torchtune.modules.tokenizers import ModelTokenizer


def data(tokenizer: ModelTokenizer, packed: bool = False, max_seq_length: int = 2048):
    """Return the dataset triple ([train], [dev], [test]) with all three empty."""
    return [[], [], []]
