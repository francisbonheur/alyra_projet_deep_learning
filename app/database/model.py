from sqlmodel import SQLModel, Field, Column, LargeBinary


class PredictionRequest (SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    image: bytes = Field(sa_column=Column(LargeBinary))
    result: str | None = None
    status: str | None = None

    def __init__(self,
                 image: bytes) -> None:
        self.image = image
        self.status = "pending"
