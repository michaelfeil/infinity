from typing import Optional

try:
    from fastapi import Request
    from fastapi.responses import JSONResponse
except ImportError:
    Request = None


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


def openai_exception_handler(request: Request, exc: OpenAIException):
    return JSONResponse(
        status_code=exc.code,
        content=exc.json(),
    )
