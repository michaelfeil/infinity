from pydantic import AnyUrl, HttpUrl, StringConstraints
from infinity_emb.env import MANAGER

__all__ = [
    "INPUT_STRING",
    "ITEMS_LIMIT",
    "ITEMS_LIMIT_SMALL",
    "AnyUrl",
    "HttpUrl",
]

# Note: adding artificial limit, this might reveal splitting
# issues on the client side
#      and is not a hard limit on the server side.
INPUT_STRING = StringConstraints(max_length=8192 * 15, strip_whitespace=True)
ITEMS_LIMIT = {
    "min_length": 1,
    "max_length": MANAGER.max_client_batch_size,
}
ITEMS_LIMIT_SMALL = {
    "min_length": 1,
    "max_length": min(32, MANAGER.max_client_batch_size),
}
