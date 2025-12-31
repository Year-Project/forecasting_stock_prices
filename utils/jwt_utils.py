import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from jwt import DecodeError, ExpiredSignatureError

from exceptions.exceptions import InvalidTokenException, TokenExpiredException
from schemas.user_role import UserRole


def create_access_token(user_id: UUID, role: UserRole) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")))

    payload = {
        "sub": str(user_id),
        "role": role.value,
        "type": "access",
        "exp": expire,
    }

    return jwt.encode(payload, os.getenv("JWT_SECRET_KEY"), os.getenv("JWT_ALGORITHM"))


def create_refresh_token(token_id: UUID, user_id: UUID, version: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS")))

    payload = {
        "jti": str(token_id),
        "sub": str(user_id),
        "version": version,
        "type": "refresh",
        "exp": expire,
    }

    return jwt.encode(payload, os.getenv("JWT_SECRET_KEY"), os.getenv("JWT_ALGORITHM"))


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=[os.getenv("JWT_ALGORITHM")])
        return payload
    except ExpiredSignatureError:
        return None
    except DecodeError:
        return None


def verify_and_decode_token(token: str, token_type: str = "access") -> dict:
    payload = decode_token(token)

    if payload is None:
        raise InvalidTokenException("Invalid refresh token's payload")

    if payload.get("type") != token_type:
        raise InvalidTokenException("Invalid refresh token's type")

    if "exp" not in payload:
        raise InvalidTokenException("Invalid expiration timestamp")

    token_expiration = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)

    if token_expiration < datetime.now(timezone.utc):
        raise TokenExpiredException()

    return payload


def get_token_expiration(token: str, token_type: str = "access") -> datetime | None:
    payload = verify_and_decode_token(token, token_type)

    return datetime.fromtimestamp(payload["exp"], tz=timezone.utc)

