from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseBrokerConsumer(ABC, Generic[T]):
    @abstractmethod
    async def consume(self, payload: T) -> None:
        pass