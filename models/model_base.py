import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrainData:
    train_x: list
    train_y: list

    test_x: list
    test_y: list


class ModelBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, train_data: TrainData):
        pass

    @abstractmethod
    def predict(self, steps: int):
        pass
