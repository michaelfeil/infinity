from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast
from typing import Literal, cast






T = TypeVar("T", bound="EmbeddingObject")



@_attrs_define
class EmbeddingObject:
    """ 
        Attributes:
            embedding (list[float] | list[list[float]] | str):
            index (int):
            object_ (Literal['embedding'] | Unset):  Default: 'embedding'.
     """

    embedding: list[float] | list[list[float]] | str
    index: int
    object_: Literal['embedding'] | Unset = 'embedding'
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        embedding: list[float] | list[list[float]] | str
        if isinstance(self.embedding, list):
            embedding = self.embedding


        elif isinstance(self.embedding, list):
            embedding = []
            for embedding_type_2_item_data in self.embedding:
                embedding_type_2_item = embedding_type_2_item_data


                embedding.append(embedding_type_2_item)


        else:
            embedding = self.embedding

        index = self.index

        object_ = self.object_


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "embedding": embedding,
            "index": index,
        })
        if object_ is not UNSET:
            field_dict["object"] = object_

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        def _parse_embedding(data: object) -> list[float] | list[list[float]] | str:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                embedding_type_0 = cast(list[float], data)

                return embedding_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, list):
                    raise TypeError()
                embedding_type_2 = []
                _embedding_type_2 = data
                for embedding_type_2_item_data in (_embedding_type_2):
                    embedding_type_2_item = cast(list[float], embedding_type_2_item_data)

                    embedding_type_2.append(embedding_type_2_item)

                return embedding_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[float] | list[list[float]] | str, data)

        embedding = _parse_embedding(d.pop("embedding"))


        index = d.pop("index")

        object_ = cast(Literal['embedding'] | Unset , d.pop("object", UNSET))
        if object_ != 'embedding'and not isinstance(object_, Unset):
            raise ValueError(f"object must match const 'embedding', got '{object_}'")

        embedding_object = cls(
            embedding=embedding,
            index=index,
            object_=object_,
        )


        embedding_object.additional_properties = d
        return embedding_object

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
