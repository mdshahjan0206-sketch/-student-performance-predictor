"""Structured, user-safe error handling.

Every failure that reaches the client is shaped as:

    {"success": false, "error": {"code": "...", "message": "...", "details": {...}}}

Raw exception text, tracebacks and filesystem paths are logged server-side
and never serialised into a response. This mirrors the original project's
rule that users only ever see clean, templated messages.
"""

import logging
import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("edupredict")


class AppError(Exception):
    """A failure with a stable machine code and a message safe to display."""

    def __init__(self, code: str, message: str, status_code: int = 400, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def error_payload(code: str, message: str, details=None) -> dict:
    body = {"success": False, "error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return body


# --- Common error factories -------------------------------------------

def dataset_not_found(dataset_id: str) -> AppError:
    return AppError(
        "DATASET_NOT_FOUND",
        "That dataset is no longer available. Upload the file again to continue.",
        status_code=404,
        details={"dataset_id": dataset_id},
    )


def model_not_found(model_id: str) -> AppError:
    return AppError(
        "MODEL_NOT_FOUND",
        "No trained model was found. Train a model before running predictions.",
        status_code=404,
        details={"model_id": model_id},
    )


def stale_model() -> AppError:
    return AppError(
        "STALE_MODEL",
        "This model was trained on a different dataset. Train a new model on the "
        "active dataset before predicting.",
        status_code=409,
    )


def register_exception_handlers(app):
    """Attach handlers that guarantee a clean response envelope."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        logger.info("AppError %s on %s: %s", exc.code, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Pydantic's raw error list leaks internal field paths and types, so a
        # single readable sentence is sent instead.
        fields = []
        for err in exc.errors():
            loc = [str(p) for p in err.get("loc", []) if p not in ("body", "query", "path")]
            if loc:
                fields.append(".".join(loc))
        detail = f" Check: {', '.join(sorted(set(fields)))}." if fields else ""
        return JSONResponse(
            status_code=422,
            content=error_payload(
                "INVALID_REQUEST",
                f"The request was missing or had invalid values.{detail}",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        messages = {
            404: "That resource does not exist.",
            405: "That action is not allowed on this address.",
            413: "The uploaded file is too large.",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(
                f"HTTP_{exc.status_code}",
                messages.get(exc.status_code, "The request could not be completed."),
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Full traceback goes to the server log with a reference id; the client
        # gets the id and nothing else.
        ref = uuid.uuid4().hex[:8]
        logger.exception("Unhandled error [ref=%s] on %s", ref, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "INTERNAL_ERROR",
                "Something went wrong while processing that request. "
                f"Try again, or report reference {ref} if it keeps happening.",
            ),
        )
