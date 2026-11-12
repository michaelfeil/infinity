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
  from ..models.classify_object import ClassifyObject
  from ..models.usage import Usage





T = TypeVar("T", bound="ClassifyResult")



@_attrs_define
class ClassifyResult:
    """ Result of classification.

        Attributes:
            data (list[list[ClassifyObject]]):
            model (str):
            usage (Usage):
            object_ (Literal['classify'] | Unset):  Default: 'classify'.
            id (str | Unset):
            created (int | Unset):
     """

    data: list[list[ClassifyObject]]
    model: str
    usage: Usage
    object_: Literal['classify'] | Unset = 'classify'
    id: str | Unset = UNSET
    created: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        from ..models.classify_object import ClassifyObject
        from ..models.usage import Usage
        data = []
        for data_item_data in self.data:
            data_item = []
            for data_item_item_data in data_item_data:
                data_item_item = data_item_item_data.to_dict()
                data_item.append(data_item_item)


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
        from ..models.classify_object import ClassifyObject
        from ..models.usage import Usage
        d = dict(src_dict)
        data = []
        _data = d.pop("data")
        for data_item_data in (_data):
            data_item = []
            _data_item = data_item_data
            for data_item_item_data in (_data_item):
                data_item_item = ClassifyObject.from_dict(data_item_item_data)



                data_item.append(data_item_item)

            data.append(data_item)


        model = d.pop("model")

        usage = Usage.from_dict(d.pop("usage"))




        object_ = cast(Literal['classify'] | Unset , d.pop("object", UNSET))
        if object_ != 'classify'and not isinstance(object_, Unset):
            raise ValueError(f"object must match const 'classify', got '{object_}'")

        id = d.pop("id", UNSET)

        created = d.pop("created", UNSET)

        classify_result = cls(
            data=data,
            model=model,
            usage=usage,
            object_=object_,
            id=id,
            created=created,
        )


        classify_result.additional_properties = d
        return classify_result

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
