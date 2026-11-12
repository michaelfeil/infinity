from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast






T = TypeVar("T", bound="ClassifyInput")



@_attrs_define
class ClassifyInput:
    """ 
        Attributes:
            input_ (list[str]):
            model (str | Unset):  Default: 'default/not-specified'.
            raw_scores (bool | Unset):  Default: False.
     """

    input_: list[str]
    model: str | Unset = 'default/not-specified'
    raw_scores: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        input_ = self.input_



        model = self.model

        raw_scores = self.raw_scores


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "input": input_,
        })
        if model is not UNSET:
            field_dict["model"] = model
        if raw_scores is not UNSET:
            field_dict["raw_scores"] = raw_scores

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        input_ = cast(list[str], d.pop("input"))


        model = d.pop("model", UNSET)

        raw_scores = d.pop("raw_scores", UNSET)

        classify_input = cls(
            input_=input_,
            model=model,
            raw_scores=raw_scores,
        )


        classify_input.additional_properties = d
        return classify_input

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
