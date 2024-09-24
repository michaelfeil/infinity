from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.classify_result_object import ClassifyResultObject
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.classify_object import ClassifyObject
    from ..models.usage import Usage


T = TypeVar("T", bound="ClassifyResult")


@_attrs_define
class ClassifyResult:
    """Result of classification.

    Attributes:
        data (List[List['ClassifyObject']]):
        model (str):
        usage (Usage):
        object_ (Union[Unset, ClassifyResultObject]):  Default: ClassifyResultObject.CLASSIFY.
        id (Union[Unset, str]):
        created (Union[Unset, int]):
    """

    data: List[List["ClassifyObject"]]
    model: str
    usage: "Usage"
    object_: Union[Unset, ClassifyResultObject] = ClassifyResultObject.CLASSIFY
    id: Union[Unset, str] = UNSET
    created: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = []
        for data_item_data in self.data:
            data_item = []
            for data_item_item_data in data_item_data:
                data_item_item = data_item_item_data.to_dict()
                data_item.append(data_item_item)

            data.append(data_item)

        model = self.model

        usage = self.usage.to_dict()

        object_: Union[Unset, str] = UNSET
        if not isinstance(self.object_, Unset):
            object_ = self.object_.value

        id = self.id

        created = self.created

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "data": data,
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
        from ..models.classify_object import ClassifyObject
        from ..models.usage import Usage

        d = src_dict.copy()
        data = []
        _data = d.pop("data")
        for data_item_data in _data:
            data_item = []
            _data_item = data_item_data
            for data_item_item_data in _data_item:
                data_item_item = ClassifyObject.from_dict(data_item_item_data)

                data_item.append(data_item_item)

            data.append(data_item)

        model = d.pop("model")

        usage = Usage.from_dict(d.pop("usage"))

        _object_ = d.pop("object", UNSET)
        object_: Union[Unset, ClassifyResultObject]
        if isinstance(_object_, Unset):
            object_ = UNSET
        else:
            object_ = ClassifyResultObject(_object_)

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
