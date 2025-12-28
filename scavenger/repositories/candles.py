from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.candle import CandleModel

class CandlesRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_range(
        self,
        ticker: str,
        interval: str,
        start,
        end,
    ):
        stmt = (
            select(CandleModel)
            .where(
                CandleModel.ticker == ticker,
                CandleModel.interval == interval,
                CandleModel.begin >= start,
                CandleModel.end <= end,
            )
            .order_by(CandleModel.begin)
        )
        res = await self.session.execute(stmt)
        return res.scalars().all()

    async def bulk_insert_if_not_exists(self, candles: list[dict]):
        objs = [CandleModel(**c) for c in candles]
        self.session.add_all(objs)
        await self.session.commit()