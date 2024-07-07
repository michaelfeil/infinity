from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ReRankObject")


@_attrs_define
class ReRankObject:
    """
    Attributes:
        relevance_score (float):
        index (int):
        document (Union[None, Unset, str]):
    """

    relevance_score: float
    index: int
    document: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        relevance_score = self.relevance_score

        index = self.index

        document: Union[None, Unset, str]
        if isinstance(self.document, Unset):
            document = UNSET
        else:
            document = self.document

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "relevance_score": relevance_score,
                "index": index,
            }
        )
        if document is not UNSET:
            field_dict["document"] = document

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        relevance_score = d.pop("relevance_score")

        index = d.pop("index")

        def _parse_document(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        document = _parse_document(d.pop("document", UNSET))

        re_rank_object = cls(
            relevance_score=relevance_score,
            index=index,
            document=document,
        )

        re_rank_object.additional_properties = d
        return re_rank_object

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
