from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from guard.src.dependencies import get_auth_session_maker, get_admin_secret_producer
from guard.src.kafka.admin_secret_producer import AdminSecretProducer
from guard.src.schemas.request.admin_init_request import AdminInitRequest
from guard.src.schemas.response.admin_init_response import AdminInitResponse
from guard.src.services.admin_init_service import AdminInitService

router = APIRouter(prefix="/internal/v1", tags=["admin"], include_in_schema=False)


@router.post("/init_admin", response_model=AdminInitResponse)
async def auth_handler(request: AdminInitRequest, admin_init_service: Annotated[AdminInitService, Depends()],
                       session_builder: async_sessionmaker[AsyncSession] = Depends(get_auth_session_maker),
                       broker_producer: AdminSecretProducer = Depends(get_admin_secret_producer)):
    return await admin_init_service.init_admin(session_builder, request, broker_producer)

