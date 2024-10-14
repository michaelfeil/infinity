from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="RerankInput")


@_attrs_define
class RerankInput:
    """Input for reranking

    Attributes:
        query (str):
        documents (List[str]):
        return_documents (Union[Unset, bool]):  Default: False.
        raw_scores (Union[Unset, bool]):  Default: False.
        model (Union[Unset, str]):  Default: 'default/not-specified'.
        top_n (Union[None, Unset, int]):
    """

    query: str
    documents: List[str]
    return_documents: Union[Unset, bool] = False
    raw_scores: Union[Unset, bool] = False
    model: Union[Unset, str] = "default/not-specified"
    top_n: Union[None, Unset, int] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        query = self.query

        documents = self.documents

        return_documents = self.return_documents

        raw_scores = self.raw_scores

        model = self.model

        top_n: Union[None, Unset, int]
        if isinstance(self.top_n, Unset):
            top_n = UNSET
        else:
            top_n = self.top_n

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "query": query,
                "documents": documents,
            }
        )
        if return_documents is not UNSET:
            field_dict["return_documents"] = return_documents
        if raw_scores is not UNSET:
            field_dict["raw_scores"] = raw_scores
        if model is not UNSET:
            field_dict["model"] = model
        if top_n is not UNSET:
            field_dict["top_n"] = top_n

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        query = d.pop("query")

        documents = cast(List[str], d.pop("documents"))

        return_documents = d.pop("return_documents", UNSET)

        raw_scores = d.pop("raw_scores", UNSET)

        model = d.pop("model", UNSET)

        def _parse_top_n(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        top_n = _parse_top_n(d.pop("top_n", UNSET))

        rerank_input = cls(
            query=query,
            documents=documents,
            return_documents=return_documents,
            raw_scores=raw_scores,
            model=model,
            top_n=top_n,
        )

        rerank_input.additional_properties = d
        return rerank_input

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
