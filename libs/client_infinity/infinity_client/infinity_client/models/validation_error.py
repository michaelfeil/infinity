from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.context import Context


T = TypeVar("T", bound="ValidationError")


@_attrs_define
class ValidationError:
    """
    Attributes:
        loc (List[Union[int, str]]):
        msg (str):
        type (str):
        input_ (Union[Unset, Any]):
        ctx (Union[Unset, Context]):
    """

    loc: List[Union[int, str]]
    msg: str
    type: str
    input_: Union[Unset, Any] = UNSET
    ctx: Union[Unset, "Context"] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        loc = []
        for loc_item_data in self.loc:
            loc_item: Union[int, str]
            loc_item = loc_item_data
            loc.append(loc_item)

        msg = self.msg

        type = self.type

        input_ = self.input_

        ctx: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.ctx, Unset):
            ctx = self.ctx.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "loc": loc,
                "msg": msg,
                "type": type,
            }
        )
        if input_ is not UNSET:
            field_dict["input"] = input_
        if ctx is not UNSET:
            field_dict["ctx"] = ctx

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.context import Context

        d = src_dict.copy()
        loc = []
        _loc = d.pop("loc")
        for loc_item_data in _loc:

            def _parse_loc_item(data: object) -> Union[int, str]:
                return cast(Union[int, str], data)

            loc_item = _parse_loc_item(loc_item_data)

            loc.append(loc_item)

        msg = d.pop("msg")

        type = d.pop("type")

        input_ = d.pop("input", UNSET)

        _ctx = d.pop("ctx", UNSET)
        ctx: Union[Unset, Context]
        if isinstance(_ctx, Unset):
            ctx = UNSET
        else:
            ctx = Context.from_dict(_ctx)

        validation_error = cls(
            loc=loc,
            msg=msg,
            type=type,
            input_=input_,
            ctx=ctx,
        )

        validation_error.additional_properties = d
        return validation_error

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
