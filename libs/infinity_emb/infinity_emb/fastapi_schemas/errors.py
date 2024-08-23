# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from infinity_emb._optional_imports import CHECK_FASTAPI

if CHECK_FASTAPI.is_available:
    from fastapi.responses import ORJSONResponse

if TYPE_CHECKING:
    from fastapi import Request


class OpenAIException(Exception):
    """An exception in OpenAI Style"""

    def __init__(
        self,
        message: str,
        code: int,
        type: Optional[str] = None,
        param: Optional[str] = None,
    ):
        self.message = message
        self.type = type
        self.param = param
        self.code = code

    def json(self):
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }


def openai_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, OpenAIException):
        return ORJSONResponse(
            status_code=exc.code,
            content=exc.json(),
        )
    else:
        return ORJSONResponse(
            status_code=500,
            content=OpenAIException(
                message="Internal Server Error",
                code=500,
            ).json(),
        )
