from typing import TYPE_CHECKING, Any, Dict, List, Literal, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.re_rank_object import ReRankObject
    from ..models.usage import Usage


T = TypeVar("T", bound="ReRankResult")


@_attrs_define
class ReRankResult:
    """Following the Cohere protocol for Rerankers.

    Attributes:
        results (List['ReRankObject']):
        model (str):
        usage (Usage):
        object_ (Union[Literal['rerank'], Unset]):  Default: 'rerank'.
        id (Union[Unset, str]):
        created (Union[Unset, int]):
    """

    results: List["ReRankObject"]
    model: str
    usage: "Usage"
    object_: Union[Literal["rerank"], Unset] = "rerank"
    id: Union[Unset, str] = UNSET
    created: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        results = []
        for results_item_data in self.results:
            results_item = results_item_data.to_dict()
            results.append(results_item)

        model = self.model

        usage = self.usage.to_dict()

        object_ = self.object_

        id = self.id

        created = self.created

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "results": results,
                "model": model,
                "usage": usage,
            }
        )
        if object_ is not UNSET:
            field_dict["object"] = object_
        if id is not UNSET:
            field_dict["id"] = id
        if created is not UNSET:
            field_dict["created"] = created

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.re_rank_object import ReRankObject
        from ..models.usage import Usage

        d = src_dict.copy()
        results = []
        _results = d.pop("results")
        for results_item_data in _results:
            results_item = ReRankObject.from_dict(results_item_data)

            results.append(results_item)

        model = d.pop("model")

        usage = Usage.from_dict(d.pop("usage"))

        object_ = cast(Union[Literal["rerank"], Unset], d.pop("object", UNSET))
        if object_ != "rerank" and not isinstance(object_, Unset):
            raise ValueError(f"object must match const 'rerank', got '{object_}'")

        id = d.pop("id", UNSET)

        created = d.pop("created", UNSET)

        re_rank_result = cls(
            results=results,
            model=model,
            usage=usage,
            object_=object_,
            id=id,
            created=created,
        )

        re_rank_result.additional_properties = d
        return re_rank_result

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
