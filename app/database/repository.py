import os
from abc import ABC, abstractmethod
from app.database.model import PredictionRequest
from fastapi import Depends
from sqlmodel import Session, create_engine, SQLModel, select

SQLITE3_DATABASE_URL = os.getenv(
    "SQLITE3_DATABASE_URL",
    "sqlite:///predictions.db")


def get_session() -> Session:
    engine = create_engine(SQLITE3_DATABASE_URL)
    SQLModel.metadata.create_all(engine)
    return Session(engine)


class PredictionRepository(ABC):
    """
    Classe de manipulation (CRUD) des objets de type PredictionRequest
    """
    @abstractmethod
    def get(self, id: int) -> PredictionRequest | None:
        pass

    @abstractmethod
    def add(
        self,
        prediction_request: PredictionRequest
    ) -> PredictionRequest | None:
        pass

    @abstractmethod
    def update(
        self,
        id: int,
        result: str,
        status: str
    ) -> PredictionRequest | None:
        pass

    @abstractmethod
    def get_all(self) -> list[PredictionRequest]:
        pass


class PredictionRepositoryImpl(PredictionRepository):
    """
    Implémentation sqlite
    """

    def __init__(self, session: Session = Depends(get_session)):
        self.session = session

    def add(
        self,
        prediction_request: PredictionRequest
    ) -> PredictionRequest | None:
        self.session.add(prediction_request)
        self.session.commit()
        self.session.refresh(prediction_request)
        return prediction_request

    def update(
        self,
        id: int,
        result: str,
        status: str
    ) -> PredictionRequest | None:
        prediction_request = self.get(id)
        if prediction_request:
            prediction_request.result = result
            prediction_request.status = status
            self.add(prediction_request)
        return prediction_request

    def get(self, id: int) -> PredictionRequest | None:
        statement = select(PredictionRequest).where(PredictionRequest.id == id)
        return self.session.exec(statement).first()

    def get_all(self) -> list[PredictionRequest]:
        statement = select(PredictionRequest)
        return self.session.exec(statement).all()
