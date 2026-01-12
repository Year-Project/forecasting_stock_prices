from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseBrokerProducer(ABC, Generic[T]):
    @abstractmethod
    async def send(self, payload: T) -> None:
        pass