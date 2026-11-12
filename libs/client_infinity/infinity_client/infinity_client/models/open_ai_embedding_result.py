from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast
from typing import Literal, cast

if TYPE_CHECKING:
  from ..models.embedding_object import EmbeddingObject
  from ..models.usage import Usage





T = TypeVar("T", bound="OpenAIEmbeddingResult")



@_attrs_define
class OpenAIEmbeddingResult:
    """ 
        Attributes:
            data (list[EmbeddingObject]):
            model (str):
            usage (Usage):
            object_ (Literal['list'] | Unset):  Default: 'list'.
            id (str | Unset):
            created (int | Unset):
     """

    data: list[EmbeddingObject]
    model: str
    usage: Usage
    object_: Literal['list'] | Unset = 'list'
    id: str | Unset = UNSET
    created: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        from ..models.embedding_object import EmbeddingObject
        from ..models.usage import Usage
        data = []
        for data_item_data in self.data:
            data_item = data_item_data.to_dict()
            data.append(data_item)



        model = self.model

        usage = self.usage.to_dict()

        object_ = self.object_

        id = self.id

        created = self.created


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "data": data,
            "model": model,
            "usage": usage,
        })
        if object_ is not UNSET:
            field_dict["object"] = object_
        if id is not UNSET:
            field_dict["id"] = id
        if created is not UNSET:
            field_dict["created"] = created

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.embedding_object import EmbeddingObject
        from ..models.usage import Usage
        d = dict(src_dict)
        data = []
        _data = d.pop("data")
        for data_item_data in (_data):
            data_item = EmbeddingObject.from_dict(data_item_data)



            data.append(data_item)


        model = d.pop("model")

        usage = Usage.from_dict(d.pop("usage"))




        object_ = cast(Literal['list'] | Unset , d.pop("object", UNSET))
        if object_ != 'list'and not isinstance(object_, Unset):
            raise ValueError(f"object must match const 'list', got '{object_}'")

        id = d.pop("id", UNSET)

        created = d.pop("created", UNSET)

        open_ai_embedding_result = cls(
            data=data,
            model=model,
            usage=usage,
            object_=object_,
            id=id,
            created=created,
        )


        open_ai_embedding_result.additional_properties = d
        return open_ai_embedding_result

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
