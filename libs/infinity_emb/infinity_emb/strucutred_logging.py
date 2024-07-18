import json
import logging
import re
from contextvars import ContextVar
from functools import wraps
from time import perf_counter, time
from traceback import format_exception
from uuid import uuid4

from fastapi import Request, HTTPException
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse


class StructuredLogging:
    request_time = ContextVar("request_time", default=perf_counter())
    account_id = ContextVar("account_id", default="null")
    request_id = ContextVar("request_id", default="No request ID")
    trace_parent = ContextVar("trace_parent", default="-")

    _config_dict = {
        "ts": "%(timestamp_ms)d",
        "type": "app",
        "svc": "embedding-service",
        "lvl": "%(levelname)s",
        "act": "%(pathname)s:%(funcName)s:%(lineno)d",
        "a_id": "%(account_id)s",
        "r_id": "%(request_id)s",
        "p": "freddy-freshservice",
        "tp": "%(trace_parent)s",
        "d": "%(time_elapsed)f",
        "thread_id": "%(thread)s",
        "trace_id": "%(otelTraceID)s",
        "dur": "%(time_elapsed)f",
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


# https://stackoverflow.com/questions/71525132/how-to-write-a-custom-fastapi-middleware-class
# Pure ASGI middleware https://www.starlette.io/middleware/#limitations is not used because we don't need the features that it provides
class StructuredLoggingMiddleware(BaseHTTPMiddleware):

    def __init__(self, app):
        super().__init__(app)

    @staticmethod
    async def list_to_async_iterator(body_list):
        for part in body_list:
            yield part

    @staticmethod
    def log_response_info(path, response, content, passed):
        response_info = {
            "st": response.status_code,
            "hdr": json.dumps({k.decode(): v.decode() for k, v in response.raw_headers}),
            "path": path,
            "lg": "http",
            "pass": passed,
            "content": (b''.join(content)).decode(),
            "d": int(1000 * (perf_counter() - StructuredLogging.request_time.get()))
        }
        if response.status_code >= 400:
            logging.error("RESPONSE", extra=response_info)
        else:
            logging.info("RESPONSE", extra=response_info)

    async def dispatch(self, request: Request, call_next):
        StructuredLogging.request_time.set(perf_counter())
        request_id = request.headers.get('request-id', request.headers.get('x-request-id', str(uuid4())))
        account_id = request.headers.get('account-id', "null")
        trace_parent = request.headers.get("traceparent", "-")

        StructuredLogging.request_id.set(request_id)
        StructuredLogging.account_id.set(account_id)
        StructuredLogging.trace_parent.set(trace_parent)

        # for every successful non-health request
        if not any(map(request.url.path.__contains__, ("health", "metrics"))):
            logging.info("SKIP", extra={"lg": "delight", "path": request.url.path})

        processed_headers = json.dumps(dict(request.headers.items()))
        body = await request.body()
        logging.info("REQUEST",
                     extra={"m": request.method, "path": request.url.path, "hdr": processed_headers,
                            "lg": "http", "bdy": body.decode()})

        passed = 1
        try:
            response = await call_next(request)
            # GETTING THE ASYNC RESPONSE BODY NEEDS WORKAROUNDS
            content = [gen async for gen in response.body_iterator]
            response.body_iterator = self.list_to_async_iterator(content)
        except HTTPException as http_exc:
            logging.error("HTTPException occurred", exc_info=http_exc)
            error_msg_422 = f"422 Unprocessable Entity: {http_exc.detail}".encode()
            response = StreamingResponse(iter((error_msg_422,)), status_code=422)
            passed = 0
        except RuntimeError as runtime_exc:
            logging.critical("RuntimeError occurred", exc_info=runtime_exc)
            error_msg_500 = f"500 Internal Server Error: {runtime_exc}".encode()
            response = StreamingResponse(iter((error_msg_500,)), status_code=500)
            passed = 0
        except Exception as e:
            logging.critical("Failure occurred in endpoint", exc_info=e)
            error_msg_5xx = b"500 internal server error due to: " + str(e).encode()
            response = StreamingResponse(iter((error_msg_5xx,)), status_code=500)
            passed = 0

        self.log_response_info(request.url.path, response, content, passed)

        return response


StructuredLogging.modify_logging()