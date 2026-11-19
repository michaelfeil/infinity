import io
from urllib.parse import quote, quote_from_bytes

import pytest
from PIL import Image

from infinity_emb.fastapi_schemas.data_uri import DataURI
from infinity_emb.fastapi_schemas.pymodels import MultiModalOpenAIEmbedding
from infinity_emb.transformer.vision.utils import resolve_from_img_bytes


def test_data_uri_build():
    DataURI("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABEklEQVR42mNk")


def test_data_uri_convert():
    data_uri = DataURI(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABEklEQVR42mNk"
    )
    assert hasattr(data_uri.convert_to_data_uri_holder(), "data")


@pytest.mark.parametrize("base64", [True, False])
@pytest.mark.parametrize("charset", [None, "utf-8", "iso-8859-1"])
def test_data_uri_binary_roundtrip(base64, charset):
    payload = bytes(range(256))
    uri = DataURI.make("image/png", charset, base64, payload)
    assert uri.data == payload
    assert uri.is_base64 is base64
    assert uri.charset == charset
    assert uri.convert_to_data_uri_holder().data == payload


@pytest.mark.parametrize("base64", [True, False])
@pytest.mark.parametrize(
    "charset, text",
    [(None, "café"), ("utf-8", "café €"), ("iso-8859-1", "café"), ("ascii", "a + 50%")],
)
def test_data_uri_text_roundtrip(base64, charset, text):
    expected = text.encode(charset or "utf-8")
    uri = DataURI.make("image/svg+xml", charset, base64, text)
    assert uri.data == expected
    if charset:
        assert uri.text == text
    if not base64:
        assert uri.partition(",")[2] == quote_from_bytes(expected)


@pytest.mark.parametrize(
    "payload, charset, expected",
    [
        ("%00%80%89%FF", "", b"\x00\x80\x89\xff"),
        ("a+b%2520", "", b"a+b%20"),
        ("%E9", ";charset=iso-8859-1", b"\xe9"),
        ("%C3%A9", ";charset=utf-8", b"\xc3\xa9"),
        ("", "", b""),
    ],
)
def test_data_uri_percent_encoded_octets(payload, charset, expected):
    assert DataURI(f"data:image/svg+xml{charset},{payload}").data == expected


@pytest.mark.parametrize("payload", ["%00%80%FF", ";base64,AID/"])
def test_data_uri_name_decoding_is_unchanged(payload):
    suffix = payload if payload.startswith(";") else f",{payload}"
    name = quote("café image.png")
    uri = DataURI(f"data:image/png;name={name}{suffix}")
    assert uri.name == "café image.png"
    assert uri.data == b"\x00\x80\xff"


@pytest.mark.parametrize("base64", [True, False])
def test_data_uri_from_file(base64, tmp_path):
    payload = b"\x89PNG\r\n\x1a\n\x00\xff"
    path = tmp_path / "image.png"
    path.write_bytes(payload)
    uri = DataURI.from_file(str(path), base64=base64)
    assert uri.mimetype == "image/png"
    assert uri.data == payload


@pytest.mark.parametrize("as_list", [True, False])
def test_data_uri_image_request_preserves_pixels(as_list):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 128, 255)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    uri = "data:image/png," + quote_from_bytes(payload)
    request = MultiModalOpenAIEmbedding.model_validate(
        {"input": [uri] if as_list else uri, "modality": "image"}
    )
    parsed_uri = request.root.input[0] if as_list else request.root.input
    assert parsed_uri.data == payload
    image = resolve_from_img_bytes(parsed_uri.data).image
    assert image.size == (4, 4)
    assert image.getpixel((0, 0)) == (0, 128, 255)
