from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.embedding_object_object import EmbeddingObjectObject
from ..types import UNSET, Unset

T = TypeVar("T", bound="EmbeddingObject")


@_attrs_define
class EmbeddingObject:
    """
    Attributes:
        embedding (List[float]):
        index (int):
        object_ (Union[Unset, EmbeddingObjectObject]):  Default: EmbeddingObjectObject.EMBEDDING.
    """

    embedding: List[float]
    index: int
    object_: Union[Unset, EmbeddingObjectObject] = EmbeddingObjectObject.EMBEDDING
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        embedding = self.embedding

        index = self.index

        object_: Union[Unset, str] = UNSET
        if not isinstance(self.object_, Unset):
            object_ = self.object_.value

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "embedding": embedding,
                "index": index,
            }
        )
        if object_ is not UNSET:
            field_dict["object"] = object_

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        embedding = cast(List[float], d.pop("embedding"))

        index = d.pop("index")

        _object_ = d.pop("object", UNSET)
        object_: Union[Unset, EmbeddingObjectObject]
        if isinstance(_object_, Unset):
            object_ = UNSET
        else:
            object_ = EmbeddingObjectObject(_object_)

        embedding_object = cls(
            embedding=embedding,
            index=index,
            object_=object_,
        )

        embedding_object.additional_properties = d
        return embedding_object

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
