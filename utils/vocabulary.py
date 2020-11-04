import pickle
from dataclasses import dataclass
from os.path import exists
from typing import Dict

from utils.common import TOKEN_TO_ID, TYPE_TO_ID, LABEL_TO_ID


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    type_to_id: Dict[str, int]
    label_to_id: Dict[str, int]

    @staticmethod
    def load_vocabulary(vocabulary_path: str) -> "Vocabulary":
        assert exists(vocabulary_path), f"Can't find file with data: {vocabulary_path}"
        with open(vocabulary_path, "rb") as vocabulary_file:
            vocabulary_dicts = pickle.load(vocabulary_file)
        token_to_id = vocabulary_dicts[TOKEN_TO_ID]
        type_to_id = vocabulary_dicts[TYPE_TO_ID]
        label_to_id = vocabulary_dicts[LABEL_TO_ID]
        return Vocabulary(token_to_id=token_to_id, type_to_id=type_to_id, label_to_id=label_to_id)

    def dump_vocabulary(self, vocabulary_path: str):
        with open(vocabulary_path, "wb") as vocabulary_file:
            pickle.dump(
                {TOKEN_TO_ID: self.token_to_id, TYPE_TO_ID: self.type_to_id, LABEL_TO_ID: self.label_to_id},
                vocabulary_file,
            )
