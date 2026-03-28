from typing import TYPE_CHECKING, Any, Dict, List, Literal, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.stats import Stats


T = TypeVar("T", bound="ModelInfo")


@_attrs_define
class ModelInfo:
    """
    Attributes:
        id (str):
        stats (Stats):
        object_ (Union[Literal['model'], Unset]):  Default: 'model'.
        owned_by (Union[Literal['infinity'], Unset]):  Default: 'infinity'.
        created (Union[Unset, int]):
        backend (Union[Unset, str]):  Default: ''.
        capabilities (Union[Unset, List[str]]):
    """

    id: str
    stats: "Stats"
    object_: Union[Literal["model"], Unset] = "model"
    owned_by: Union[Literal["infinity"], Unset] = "infinity"
    created: Union[Unset, int] = UNSET
    backend: Union[Unset, str] = ""
    capabilities: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id

        stats = self.stats.to_dict()

        object_ = self.object_

        owned_by = self.owned_by

        created = self.created

        backend = self.backend

        capabilities: Union[Unset, List[str]] = UNSET
        if not isinstance(self.capabilities, Unset):
            capabilities = self.capabilities

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "stats": stats,
            }
        )
        if object_ is not UNSET:
            field_dict["object"] = object_
        if owned_by is not UNSET:
            field_dict["owned_by"] = owned_by
        if created is not UNSET:
            field_dict["created"] = created
        if backend is not UNSET:
            field_dict["backend"] = backend
        if capabilities is not UNSET:
            field_dict["capabilities"] = capabilities

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.stats import Stats

        d = src_dict.copy()
        id = d.pop("id")

        stats = Stats.from_dict(d.pop("stats"))

        object_ = cast(Union[Literal["model"], Unset], d.pop("object", UNSET))
        if object_ != "model" and not isinstance(object_, Unset):
            raise ValueError(f"object must match const 'model', got '{object_}'")

        owned_by = cast(Union[Literal["infinity"], Unset], d.pop("owned_by", UNSET))
        if owned_by != "infinity" and not isinstance(owned_by, Unset):
            raise ValueError(f"owned_by must match const 'infinity', got '{owned_by}'")

        created = d.pop("created", UNSET)

        backend = d.pop("backend", UNSET)

        capabilities = cast(List[str], d.pop("capabilities", UNSET))

        model_info = cls(
            id=id,
            stats=stats,
            object_=object_,
            owned_by=owned_by,
            created=created,
            backend=backend,
            capabilities=capabilities,
        )

        model_info.additional_properties = d
        return model_info

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
