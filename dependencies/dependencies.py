from uuid import UUID

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from db.registry import DatabaseRegistry
from exceptions.exceptions import AdminAccessRequiredException
from schemas.current_user import CurrentUser
from schemas.user_role import UserRole
from utils.jwt_utils import verify_and_decode_token

db_registry = DatabaseRegistry()
security = HTTPBearer()


def get_session_maker(name: str):
    return db_registry.get(name).sessionmaker


async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> CurrentUser:
    token = creds.credentials
    payload = verify_and_decode_token(token)
    return CurrentUser(user_id=UUID(payload["sub"]), role=UserRole(payload["role"]))


def get_admin_user(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != UserRole.ADMIN:
        raise AdminAccessRequiredException()
    return user
