from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.embedding_encoding_format import EmbeddingEncodingFormat
from ..models.open_ai_embedding_input_text_modality import OpenAIEmbeddingInputTextModality
from ..types import UNSET, Unset

T = TypeVar("T", bound="OpenAIEmbeddingInputText")


@_attrs_define
class OpenAIEmbeddingInputText:
    """helper

    Attributes:
        input_ (Union[List[str], str]):
        model (Union[Unset, str]):  Default: 'default/not-specified'.
        encoding_format (Union[Unset, EmbeddingEncodingFormat]):
        user (Union[None, Unset, str]):
        modality (Union[Unset, OpenAIEmbeddingInputTextModality]):  Default: OpenAIEmbeddingInputTextModality.TEXT.
    """

    input_: Union[List[str], str]
    model: Union[Unset, str] = "default/not-specified"
    encoding_format: Union[Unset, EmbeddingEncodingFormat] = UNSET
    user: Union[None, Unset, str] = UNSET
    modality: Union[Unset, OpenAIEmbeddingInputTextModality] = OpenAIEmbeddingInputTextModality.TEXT
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        input_: Union[List[str], str]
        if isinstance(self.input_, list):
            input_ = self.input_

        else:
            input_ = self.input_

        model = self.model

        encoding_format: Union[Unset, str] = UNSET
        if not isinstance(self.encoding_format, Unset):
            encoding_format = self.encoding_format.value

        user: Union[None, Unset, str]
        if isinstance(self.user, Unset):
            user = UNSET
        else:
            user = self.user

        modality: Union[Unset, str] = UNSET
        if not isinstance(self.modality, Unset):
            modality = self.modality.value

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input": input_,
            }
        )
        if model is not UNSET:
            field_dict["model"] = model
        if encoding_format is not UNSET:
            field_dict["encoding_format"] = encoding_format
        if user is not UNSET:
            field_dict["user"] = user
        if modality is not UNSET:
            field_dict["modality"] = modality

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()

        def _parse_input_(data: object) -> Union[List[str], str]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                input_type_0 = cast(List[str], data)

                return input_type_0
            except:  # noqa: E722
                pass
            return cast(Union[List[str], str], data)

        input_ = _parse_input_(d.pop("input"))

        model = d.pop("model", UNSET)

        _encoding_format = d.pop("encoding_format", UNSET)
        encoding_format: Union[Unset, EmbeddingEncodingFormat]
        if isinstance(_encoding_format, Unset):
            encoding_format = UNSET
        else:
            encoding_format = EmbeddingEncodingFormat(_encoding_format)

        def _parse_user(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        user = _parse_user(d.pop("user", UNSET))

        _modality = d.pop("modality", UNSET)
        modality: Union[Unset, OpenAIEmbeddingInputTextModality]
        if isinstance(_modality, Unset):
            modality = UNSET
        else:
            modality = OpenAIEmbeddingInputTextModality(_modality)

        open_ai_embedding_input_text = cls(
            input_=input_,
            model=model,
            encoding_format=encoding_format,
            user=user,
            modality=modality,
        )

        open_ai_embedding_input_text.additional_properties = d
        return open_ai_embedding_input_text

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
