# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import json
import logging
import re
from contextvars import ContextVar
from enum import Enum
from functools import wraps
from time import perf_counter, time
from traceback import format_exception
from uuid import uuid4

from fastapi import Request, HTTPException
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse


def sla_breached(duration):
    return duration > 3000


class StructuredLogging:
    request_time = ContextVar("request_time", default=perf_counter())
    account_id = ContextVar("account_id", default="null")
    request_id = ContextVar("request_id", default="null")
    trace_parent = ContextVar("trace_parent", default="null")
    controller = ContextVar("controller", default=["null"])  # needed for implementation

    _config_dict = {
        "ts": "%(timestamp_ms)d",
        "type": "app",
        "svc": "freddy-infinity",
        "lvl": "%(levelname)s",
        "act": "%(controller)s",
        "a_id": "%(account_id)s",
        "r_id": "%(request_id)s",
        "p": "freddy-fs",
        "tp": "%(trace_parent)s",
        "td": "%(thread)s",
        "trace_id": "%(otelTraceID)s",
        "span_id": "%(otelSpanID)s",
        "msg": "%(message)s",
    }

    _dumped_dict = json.dumps(_config_dict)

    _number_in_quotes, _group_without_quotes = r"(\")(%\([^\)]+\)[df])(\")", r"\g<2>"
    _config_string_static = re.sub(
        _number_in_quotes, _group_without_quotes, _dumped_dict
    )

    config_string = _config_string_static[:-1] + "%(extra_data)s}"

    @classmethod
    def modify_logging(cls):
        cls._change_factory()
        cls._format_extra_data()
        LoggingInstrumentor().instrument(logging_format=StructuredLogging.config_string)

    @classmethod
    def _change_factory(cls):
        def sanitize_string(data) -> str:
            return json.dumps(str(data), ensure_ascii=False)[1:-1]

        original_factory = logging.getLogRecordFactory()

        @wraps(original_factory)
        def record_factory(*args, **kwargs):
            record = original_factory(*args, **kwargs)
            record.timestamp_ms = int(time() * 1000)
            record.account_id = sanitize_string(cls.account_id.get())
            record.request_id = sanitize_string(cls.request_id.get())
            record.controller = sanitize_string(
                cls.controller.get()[0]
            )  # first element of list
            record.trace_parent = sanitize_string(cls.trace_parent.get())
            record.time_elapsed = perf_counter() - cls.request_time.get()
            record.msg = sanitize_string(record.msg)
            if record.args:
                record.args = tuple(
                    sanitize_string(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            return record

        logging.setLogRecordFactory(record_factory)

    @staticmethod
    def _format_extra_data():
        """changes the logging.Logger.makeRecord method"""

        def nice_format(data):
            if type(data) in (int, str, float):
                return data
            try:
                return json.dumps(data, ensure_ascii=False)
            except Exception:
                return str(data)

        original_make_record = logging.Logger.makeRecord

        @wraps(original_make_record)
        def new_make_record(*args, **kwargs):
            record = original_make_record(*args, **kwargs)
            # original_make_record.__code__.co_varnames[:original_make_record.__code__.co_argcount].index('extra')
            extra_index = 9  # obtained from function signature, see above comment.
            extra = args[extra_index]

            # correctly format the extra fields
            if not extra:
                extra = dict()
            elif not isinstance(extra, dict):
                extra = {"extra_data": nice_format(extra)}
            else:
                extra = {str(k): nice_format(v) for k, v in extra.items()}

            # add the ex field if needed
            if record.exc_info:
                extra["ex"] = (
                    "".join(format_exception(*record.exc_info))
                    if isinstance(record.exc_info, tuple)
                    else str(record.exc_info)
                )

            # add it to extra_data if not empty
            record.extra_data = (
                (", " + json.dumps(extra, ensure_ascii=False)[1:-1]) if extra else ""
            )
            return record

        logging.Logger.makeRecord = new_make_record


class ControllerFieldHelper:
    """Helper class to log act (controller) field"""

    # Note - ContextVar contains list with single element which is mutated
    # This helps persist the object beyond its 'official' context, allowing simpler implementation
    _request_log_fn = ContextVar(
        "_request_log_fn", default=[lambda: logging.critical("REQUEST LOG FAILURE")]
    )

    @classmethod
    def modify_fastapi_run_endpoint_function(cls):
        """to log act, we override fastapi's routing.run_endpoint_function"""
        from fastapi import routing

        old_run_endpoint_function = routing.run_endpoint_function

        async def new_run_endpoint_function(**kwargs):
            StructuredLogging.controller.get()[0] = kwargs[
                "dependant"  # codespell:ignore
            ].call.__name__
            cls._request_log_fn.get()[0]()
            cls._request_log_fn.get()[0] = lambda: None
            return await old_run_endpoint_function(**kwargs)

        routing.run_endpoint_function = new_run_endpoint_function

    @classmethod
    def log_when_controller_found(cls, fn):
        StructuredLogging.controller.set(["-"])
        cls._request_log_fn.set([fn])

    @classmethod
    def log_if_controller_not_found(cls):
        cls._request_log_fn.get()[0]()


# https://stackoverflow.com/questions/71525132/how-to-write-a-custom-fastapi-middleware-class Pure ASGI middleware
# https://www.starlette.io/middleware/#limitations is not used because we don't need the features that it provides
class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    @staticmethod
    async def list_to_async_iterator(body_list):
        for part in body_list:
            yield part

    @staticmethod
    def log_response_info(request, response, content, passed, duration):
        if sla_breached(duration):
            passed = 0
        response_info = {
            "st": response.status_code,
            "hdr": json.dumps(
                {k.decode(): v.decode() for k, v in response.raw_headers}
            ),
            "path": request.url.path,
            "lg": "http",
            "pass": passed,
            "body": (b"".join(content)).decode(),
            "d": duration,
            "m": request.method,
        }
        if response.status_code >= 400:
            logging.error("RESPONSE", extra=response_info)
        else:
            logging.info("RESPONSE", extra=response_info)

        # for every successful non-health request
        if not any(map(request.url.path.__contains__, ("health", "metrics"))):
            logging.info(
                None,
                extra={
                    "st": response.status_code,
                    "path": request.url.path,
                    "lg": "delight",
                    "pass": passed,
                    "d": duration,
                    "m": request.method,
                },
            )

    async def dispatch(self, request: Request, call_next):
        StructuredLogging.request_time.set(perf_counter())
        request_id = request.headers.get(
            "request-id", request.headers.get("x-request-id", str(uuid4()))
        )
        account_id = request.headers.get("account-id", "null")
        trace_parent = request.headers.get("traceparent", "-")

        StructuredLogging.request_id.set(request_id)
        StructuredLogging.account_id.set(account_id)
        StructuredLogging.trace_parent.set(trace_parent)

        processed_headers = json.dumps(dict(request.headers.items()))
        body = await request.body()

        ControllerFieldHelper.log_when_controller_found(
            lambda: logging.info(
                "REQUEST",
                extra={
                    "m": request.method,
                    "path": request.url.path,
                    "hdr": processed_headers,
                    "body": body,
                    "lg": "http",
                },
            ),
        )
        passed = 1
        content = []
        try:
            response = await call_next(request)
            ControllerFieldHelper.log_if_controller_not_found()
            # GETTING THE ASYNC RESPONSE BODY NEEDS WORKAROUNDS
            content = [gen async for gen in response.body_iterator]
            response.body_iterator = self.list_to_async_iterator(content)
        except HTTPException as http_exc:
            ControllerFieldHelper.log_if_controller_not_found()
            logging.error("HTTPException occurred", exc_info=http_exc)
            error_msg_422 = f"422 Unprocessable Entity: {http_exc.detail}".encode()
            response = StreamingResponse(iter((error_msg_422,)), status_code=422)
            passed = 0
        except RuntimeError as runtime_exc:
            ControllerFieldHelper.log_if_controller_not_found()
            logging.critical("RuntimeError occurred", exc_info=runtime_exc)
            error_msg_500 = f"500 Internal Server Error: {runtime_exc}".encode()
            response = StreamingResponse(iter((error_msg_500,)), status_code=500)
            passed = 0
        except Exception as e:
            ControllerFieldHelper.log_if_controller_not_found()
            logging.critical("Failure occurred in endpoint", exc_info=e)
            error_msg_5xx = b"500 internal server error due to: " + str(e).encode()
            response = StreamingResponse(iter((error_msg_5xx,)), status_code=500)
            passed = 0

        duration = int(1000 * (perf_counter() - StructuredLogging.request_time.get()))
        self.log_response_info(request, response, content, passed, duration)

        return response


ControllerFieldHelper.modify_fastapi_run_endpoint_function()
StructuredLogging.modify_logging()

LOG_LEVELS: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": 5,
}

logging.basicConfig(
    level="INFO",
    format=StructuredLogging.config_string,
)

logger = logging.getLogger()


class UVICORN_LOG_LEVELS(Enum):
    """Re-exports the uvicorn log levels for type hinting and usage."""

    critical = "critical"
    error = "error"
    warning = "warning"
    info = "info"
    debug = "debug"
    trace = "trace"

    def to_int(self) -> int:
        return LOG_LEVELS[self.name]
