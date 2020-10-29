import pickle
from dataclasses import dataclass
from os.path import exists
from typing import Dict


SEED = 7

# sequence service tokens
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"

# path context keys
FROM_TOKEN = "from_token"
PATH_TYPES = "path_types"
TO_TOKEN = "to_token"

# vocabulary keys
TOKEN_TO_ID = "token_to_id"
TYPE_TO_ID = "type_to_id"
LABEL_TO_ID = "label_to_id"

# dataset keys
DATA_FOLDER = "data"
VOCABULARY_NAME = "vocabulary.pkl"
TRAIN_HOLDOUT = "train"
VAL_HOLDOUT = "val"
TEST_HOLDOUT = "test"
HOLDOUTS = [TRAIN_HOLDOUT, VAL_HOLDOUT, TEST_HOLDOUT]


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    type_to_id: Dict[str, int]
    label_to_id: Dict[str, int]


def load_vocabulary(vocabulary_path: str) -> Vocabulary:
    assert exists(vocabulary_path), f"Can't find file with data: {vocabulary_path}"
    with open(vocabulary_path, "rb") as vocabulary_file:
        vocabulary_dicts = pickle.load(vocabulary_file)
    token_to_id = vocabulary_dicts[TOKEN_TO_ID]
    type_to_id = vocabulary_dicts[TYPE_TO_ID]
    label_to_id = vocabulary_dicts[LABEL_TO_ID]
    return Vocabulary(token_to_id=token_to_id, type_to_id=type_to_id, label_to_id=label_to_id)
