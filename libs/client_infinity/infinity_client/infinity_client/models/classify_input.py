from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ClassifyInput")


@_attrs_define
class ClassifyInput:
    """
    Attributes:
        input_ (List[str]):
        model (Union[Unset, str]):  Default: 'default/not-specified'.
        raw_scores (Union[Unset, bool]):  Default: False.
    """

    input_: List[str]
    model: Union[Unset, str] = "default/not-specified"
    raw_scores: Union[Unset, bool] = False
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        input_ = self.input_

        model = self.model

        raw_scores = self.raw_scores

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input": input_,
            }
        )
        if model is not UNSET:
            field_dict["model"] = model
        if raw_scores is not UNSET:
            field_dict["raw_scores"] = raw_scores

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        input_ = cast(List[str], d.pop("input"))

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
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
