from pydantic import BaseModel


class ParsedTimeframe(BaseModel):
    time_frame_interval: int
    time_frame_unit: str
