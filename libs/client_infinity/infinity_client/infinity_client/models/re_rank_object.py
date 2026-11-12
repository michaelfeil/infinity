from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast






T = TypeVar("T", bound="ReRankObject")



@_attrs_define
class ReRankObject:
    """ 
        Attributes:
            relevance_score (float):
            index (int):
            document (None | str | Unset):
     """

    relevance_score: float
    index: int
    document: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)





    def to_dict(self) -> dict[str, Any]:
        relevance_score = self.relevance_score

        index = self.index

        document: None | str | Unset
        if isinstance(self.document, Unset):
            document = UNSET
        else:
            document = self.document


        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
            "relevance_score": relevance_score,
            "index": index,
        })
        if document is not UNSET:
            field_dict["document"] = document

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        relevance_score = d.pop("relevance_score")

        index = d.pop("index")

        def _parse_document(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        document = _parse_document(d.pop("document", UNSET))


        re_rank_object = cls(
            relevance_score=relevance_score,
            index=index,
            document=document,
        )


        re_rank_object.additional_properties = d
        return re_rank_object

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
