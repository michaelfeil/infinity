from io import BytesIO
from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.embedding_object_object import EmbeddingObjectObject
from ..types import UNSET, File, FileJsonType, Unset

T = TypeVar("T", bound="EmbeddingObject")


@_attrs_define
class EmbeddingObject:
    """
    Attributes:
        embedding (Union[File, List[List[float]], List[float]]):
        index (int):
        object_ (Union[Unset, EmbeddingObjectObject]):  Default: EmbeddingObjectObject.EMBEDDING.
    """

    embedding: Union[File, List[List[float]], List[float]]
    index: int
    object_: Union[Unset, EmbeddingObjectObject] = EmbeddingObjectObject.EMBEDDING
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        embedding: Union[FileJsonType, List[List[float]], List[float]]
        if isinstance(self.embedding, list):
            embedding = self.embedding

        elif isinstance(self.embedding, File):
            embedding = self.embedding.to_tuple()

        else:
            embedding = []
            for embedding_type_2_item_data in self.embedding:
                embedding_type_2_item = embedding_type_2_item_data

                embedding.append(embedding_type_2_item)

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

        def _parse_embedding(data: object) -> Union[File, List[List[float]], List[float]]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                embedding_type_0 = cast(List[float], data)

                return embedding_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, bytes):
                    raise TypeError()
                embedding_type_1 = File(payload=BytesIO(data))

                return embedding_type_1
            except:  # noqa: E722
                pass
            if not isinstance(data, list):
                raise TypeError()
            embedding_type_2 = []
            _embedding_type_2 = data
            for embedding_type_2_item_data in _embedding_type_2:
                embedding_type_2_item = cast(List[float], embedding_type_2_item_data)

                embedding_type_2.append(embedding_type_2_item)

            return embedding_type_2

        embedding = _parse_embedding(d.pop("embedding"))

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
