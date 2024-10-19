import mimetypes
import re
import sys
import textwrap
from base64 import b64decode as decode64
from base64 import b64encode as encode64
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional, TypeVar, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from urllib.parse import quote, unquote

T = TypeVar("T")

MIMETYPE_REGEX = r"[\w]+\/[\w\-\+\.]+"
MIMETYPE_REGEX_AUDIO_IMAGE = r"(audio|image)\/[\w\-\+\.]+"
_MIMETYPE_RE = re.compile("^{}$".format(MIMETYPE_REGEX_AUDIO_IMAGE))

CHARSET_REGEX = r"[\w\-\+\.]+"
_CHARSET_RE = re.compile("^{}$".format(CHARSET_REGEX))

DATA_URI_REGEX = (
    r"data:"
    + r"(?P<mimetype>{})?".format(MIMETYPE_REGEX)
    + r"(?:\;name\=(?P<name>[\w\.\-%!*'~\(\)]+))?"
    + r"(?:\;charset\=(?P<charset>{}))?".format(CHARSET_REGEX)
    + r"(?P<base64>\;base64)?"
    + r",(?P<data>.*)"
)
_DATA_URI_RE = re.compile(r"^{}$".format(DATA_URI_REGEX), re.DOTALL)


class InvalidMimeType(ValueError):
    pass


class InvalidCharset(ValueError):
    pass


class InvalidDataURI(ValueError):
    pass


@dataclass
class DataURIHolder:
    mimetype: Optional[str]
    charset: Optional[str]
    base64: bool
    data: Union[str, bytes]


class DataURI(str):
    @classmethod
    def make(
        cls,
        mimetype: Optional[str],
        charset: Optional[str],
        base64: Optional[bool],
        data: Union[str, bytes],
    ) -> Self:
        parts = ["data:"]
        if mimetype is not None:
            if not _MIMETYPE_RE.match(mimetype):
                raise InvalidMimeType("Invalid mimetype: %r" % mimetype)
            parts.append(mimetype)
        if charset is not None:
            if not _CHARSET_RE.match(charset):
                raise InvalidCharset("Invalid charset: %r" % charset)
            parts.extend([";charset=", charset])
        if base64:
            parts.append(";base64")
            _charset = charset or "utf-8"
            if isinstance(data, bytes):
                _data = data
            else:
                _data = bytes(data, _charset)
            encoded_data = encode64(_data).decode(_charset).strip()
        else:
            encoded_data = quote(data)
        parts.extend([",", encoded_data])
        return cls("".join(parts))

    @classmethod
    def from_file(
        cls,
        filename: str,
        charset: Optional[str] = None,
        base64: Optional[bool] = True,
        mimetype: Optional[str] = None,
    ) -> Self:
        if mimetype is None:
            mimetype, _ = mimetypes.guess_type(filename, strict=False)
        with open(filename, "rb") as fp:
            data = fp.read()

        return cls.make(mimetype, charset, base64, data)

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        uri = super(DataURI, cls).__new__(cls, *args, **kwargs)
        uri._parse  # Trigger any ValueErrors on instantiation.
        return uri

    def __repr__(self) -> str:
        truncated = str(self)
        if len(truncated) > 80:
            truncated = truncated[:79] + "…"
        return "DataURI(%s)" % (truncated,)

    def wrap(self, width: int = 76) -> str:
        return "\n".join(textwrap.wrap(self, width, break_on_hyphens=False))

    @property
    def mimetype(self) -> Optional[str]:
        return self._parse[0]

    @property
    def name(self) -> Optional[str]:
        name = self._parse[1]
        if name is not None:
            return unquote(name)
        return name

    @property
    def charset(self) -> Optional[str]:
        return self._parse[2]

    @property
    def is_base64(self) -> bool:
        return self._parse[3]

    @property
    def data(self) -> bytes:
        return self._parse[4]

    def convert_to_data_uri_holder(self) -> DataURIHolder:
        return DataURIHolder(
            mimetype=self.mimetype,
            charset=self.charset,
            base64=self.is_base64,
            data=self.data,
        )

    @property
    def text(self) -> str:
        if self.charset is None:
            raise InvalidCharset("DataURI has no encoding set.")

        return self.data.decode(self.charset)

    @property
    def is_valid(self) -> bool:
        match = _DATA_URI_RE.match(self)
        if not match:
            return False
        return True

    @property
    def _parse(
        self,
    ) -> tuple[Optional[str], Optional[str], Optional[str], bool, bytes]:
        match = _DATA_URI_RE.match(self)
        if match is None:
            raise InvalidDataURI("Not a valid data URI: %r" % self)
        mimetype = match.group("mimetype") or None
        name = match.group("name") or None
        charset = match.group("charset") or None
        _charset = charset or "utf-8"

        if match.group("base64"):
            _data = bytes(match.group("data"), _charset)
            data = decode64(_data)
        else:
            data = bytes(unquote(match.group("data")), _charset)

        return mimetype, name, charset, bool(match.group("base64")), data

    # Pydantic methods
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        # return core_schema.no_info_after_validator_function(cls, handler(str))
        return core_schema.no_info_after_validator_function(cls.validate, core_schema.str_schema())

    @classmethod
    def validate(
        cls,
        value: str,
        values: Optional[MutableMapping[str, Any]] = None,
        config: Any = None,
        field: Any = None,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(value, str):
            raise TypeError("string required")

        m = cls(value)
        if not m.is_valid:
            raise ValueError("invalid data-uri format")
        return m

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: MutableMapping[str, Any], handler: Any) -> Any:
        json_schema = handler(schema)
        json_schema.update(
            pattern=DATA_URI_REGEX,
            examples=[
                "data:text/plain;charset=utf-8;base64,"
                "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wZWQgb3ZlciB0aGUgbGF6eSBkb2cu"
            ],
        )
        return json_schema

    @classmethod
    def __modify_schema__(cls, field_schema: dict[str, Any]) -> None:
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            pattern=DATA_URI_REGEX,
            examples=[
                "data:text/plain;charset=utf-8;base64,VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wZWQgb3ZlciB0aGUgbGF6eSBkb2cu"
            ],
        )
