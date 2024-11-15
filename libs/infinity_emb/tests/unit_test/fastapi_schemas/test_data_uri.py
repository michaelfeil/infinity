from infinity_emb.fastapi_schemas.data_uri import DataURI


def test_data_uri_build():
    DataURI("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABEklEQVR42mNk")


def test_data_uri_convert():
    data_uri = DataURI(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABEklEQVR42mNk"
    )
    assert hasattr(data_uri.convert_to_data_uri_holder(), "data")
