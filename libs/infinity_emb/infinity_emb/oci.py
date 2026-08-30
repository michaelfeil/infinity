"""Resolve ``oci://`` model references to a local path.

A model published as a CNCF ModelPack (https://github.com/modelpack/model-spec)
artifact lives in an ordinary container registry, so it reuses the registry,
credentials, mirroring and air-gap tooling a deployment already has for
container images.

Acquisition is delegated to a running ``llmman serve``
(https://github.com/llmmanorg/llmman), which already implements the ModelPack
media types, registry auth, resumable blob download and a content-addressed
store. The daemon does the pull (POST /api/pull, streamed so a multi-gigabyte
fetch is not silent) but deliberately exposes no local path, so
``llmman resolve --no-pull`` reports where the bytes landed.

An explicit ``oci://`` scheme is required rather than sniffing a bare
``registry/name:tag``: that shape is indistinguishable from a HuggingFace repo
id (``org/model``), so guessing would silently hijack existing deployments.
"""

import logging

from infinity_emb import llmman

logger = logging.getLogger("infinity_emb")

SCHEME = "oci://"


def is_oci_ref(model_name_or_path) -> bool:
    """Whether the reference carries the ``oci://`` scheme."""
    if not model_name_or_path:
        return False
    return str(model_name_or_path).lower().startswith(SCHEME)


def strip_scheme(model_name_or_path) -> str:
    """Drop the ``oci://`` prefix, leaving the bare registry reference."""
    text = str(model_name_or_path)
    if is_oci_ref(text):
        return text[len(SCHEME) :]
    return text


def resolve(model_name_or_path) -> str:
    """Pull an ``oci://`` reference through llmman and return the local path."""
    reference = strip_scheme(model_name_or_path).strip()
    if not reference:
        raise ValueError(f"empty OCI model reference: {model_name_or_path!r}")

    def _progress(status, completed, total):
        if total:
            logger.info("llmman: %s (%s/%s bytes)", status, completed, total)
        else:
            logger.info("llmman: %s", status)

    return llmman.pull_and_resolve(reference, progress=_progress)
