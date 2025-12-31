from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Float,
    UniqueConstraint,
)
from src.models.base import Base


class CandleModel(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True)

    ticker = Column(String, nullable=False)
    isin = Column(String, nullable=True)
    interval = Column(int, nullable=False)

    begin = Column(DateTime, nullable=False)
    end = Column(DateTime, nullable=False)

    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "interval", "begin",
            name="uq_candles_ticker_interval_begin"
        ),
    )
