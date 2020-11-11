from dataclasses import dataclass

from configs import Code2SeqConfig, Code2SeqTestConfig
from configs.parts import TypedPathContextConfig, ContextDescription


@dataclass(frozen=True)
class TypedCode2SeqConfig(Code2SeqConfig):
    data_processing = TypedPathContextConfig(
        token_description=ContextDescription(max_parts=5, is_wrapped=False, is_splitted=True, vocab_size=190000),
        path_description=ContextDescription(max_parts=9, is_wrapped=False, is_splitted=True,),
        target_description=ContextDescription(max_parts=7, is_wrapped=False, is_splitted=True, vocab_size=27000),
        type_description=ContextDescription(max_parts=5, is_wrapped=False, is_splitted=True,),
    )


@dataclass(frozen=True)
class TypedCode2SeqTestConfig(Code2SeqTestConfig):
    data_processing = TypedPathContextConfig(
        token_description=ContextDescription(max_parts=5, is_wrapped=False, is_splitted=True, vocab_size=190000),
        path_description=ContextDescription(max_parts=9, is_wrapped=False, is_splitted=True,),
        target_description=ContextDescription(max_parts=7, is_wrapped=False, is_splitted=True, vocab_size=27000),
        type_description=ContextDescription(max_parts=5, is_wrapped=False, is_splitted=True,),
    )
