"""`oci://` model references resolve to a local path.

The scheme is explicit on purpose: a bare `registry/name:tag` is the same shape
as a HuggingFace repo id, so sniffing would hijack existing deployments.
"""

import os
from unittest import mock

import pytest

from infinity_emb import llmman
from infinity_emb.oci import is_oci_ref, resolve, strip_scheme


def test_recognizes_the_oci_scheme():
    assert is_oci_ref("oci://ghcr.io/org/model:tag")
    assert is_oci_ref("OCI://ghcr.io/org/model:tag")


@pytest.mark.parametrize(
    "value",
    [
        "michaelfeil/bge-small-en-v1.5",
        "ghcr.io/org/model:tag",
        "/local/path/to/model",
        "s3://bucket/key",
        "",
        None,
    ],
)
def test_leaves_every_other_shape_alone(value):
    # A bare HF repo id must never be claimed.
    assert not is_oci_ref(value)


def test_strips_the_scheme_only_when_present():
    assert strip_scheme("oci://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"
    assert strip_scheme("OCI://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"
    assert strip_scheme("michaelfeil/bge") == "michaelfeil/bge"


@pytest.mark.parametrize("ref", ["oci://", "oci://   "])
def test_rejects_an_empty_reference(ref):
    with pytest.raises(ValueError):
        resolve(ref)


def test_hands_the_bare_reference_to_the_daemon():
    with mock.patch(
        "infinity_emb.oci.llmman.pull_and_resolve", return_value="/resolved"
    ) as acquire:
        assert resolve("oci://ghcr.io/org/model:tag") == "/resolved"
    assert acquire.call_args[0][0] == "ghcr.io/org/model:tag"
    assert acquire.call_args[1]["progress"] is not None


@pytest.mark.parametrize(
    "host,want",
    [
        ("", "http://127.0.0.1:17434"),
        ("1.2.3.4:9999", "http://1.2.3.4:9999"),
        ("1.2.3.4", "http://1.2.3.4:17434"),
        # A wildcard bind is meaningful to the server but not to a client.
        ("0.0.0.0:9999", "http://127.0.0.1:9999"),
        ("[::]:9999", "http://[::1]:9999"),
    ],
)
def test_endpoint_parsing(host, want):
    with mock.patch.dict(os.environ, {llmman.HOST_ENV: host}):
        assert llmman.endpoint() == want


class TestEngineArgsIntegration:
    """EngineArgs resolves the reference before anything else reads it."""

    def test_rewrites_model_name_or_path_and_keeps_the_served_name(self):
        from infinity_emb.args import EngineArgs

        with mock.patch("infinity_emb.oci.resolve", return_value="/resolved"):
            args = EngineArgs(model_name_or_path="oci://ghcr.io/org/model:tag")

        assert args.model_name_or_path == "/resolved"
        # The served name stays the reference the user typed, not the store path.
        assert args.served_model_name == "oci://ghcr.io/org/model:tag"

    def test_an_explicit_served_name_wins(self):
        from infinity_emb.args import EngineArgs

        with mock.patch("infinity_emb.oci.resolve", return_value="/resolved"):
            args = EngineArgs(
                model_name_or_path="oci://ghcr.io/org/model:tag",
                served_model_name="my-model",
            )

        assert args.served_model_name == "my-model"

    def test_a_hf_repo_id_is_untouched(self):
        from infinity_emb.args import EngineArgs

        with mock.patch("infinity_emb.oci.resolve") as resolver:
            args = EngineArgs(model_name_or_path="michaelfeil/bge-small-en-v1.5")

        resolver.assert_not_called()
        assert args.model_name_or_path == "michaelfeil/bge-small-en-v1.5"
