from dataclasses import dataclass, field
from typing import Dict, Counter, List


@dataclass
class Vocabulary:
    token_to_id: Dict = field(default_factory=dict)
    type_to_id: Dict = field(default_factory=dict)
    label_to_id: Dict = field(default_factory=dict)

    def add_from_counter(
        self, target_field: str, counter: Counter, n_most_values: int = -1, add_values: List[str] = None,
    ):
        if not hasattr(self, target_field):
            raise ValueError(f"There is no {target_field} attribute in vocabulary class")
        if add_values is None:
            add_values = []
        if n_most_values == -1:
            add_values += list(zip(*counter.most_common()))[0]
        else:
            add_values += list(zip(*counter.most_common(n_most_values - len(add_values))))[0]
        attr = {value: i for i, value in enumerate(add_values)}
        setattr(self, target_field, attr)
