from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from config.settings import Settings
from app.core.database import Database
from app.core.pipeline import ForecastPipeline


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_pipeline(request: Request) -> ForecastPipeline:
    return request.app.state.pipeline


async def require_auth(request: Request) -> None:
    settings: Settings = request.app.state.settings
    if not settings.api_token:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, settings.api_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
