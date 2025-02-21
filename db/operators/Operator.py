from abc import ABC, abstractmethod
from typing import Iterator

from db.structure import SQLTable


class Operator(ABC, Iterator):
    """
    Abstract class for Operators.
    Implements the Iterator interface -> __next__
    Implements the Volcano Model [Graefe, G. (1989). An Extensible and Parallel Query Evaluation System.]
     -> open, __next__, close
    """
    def __init__(self, table: SQLTable, num_tuples: int) -> None:
        self.table: SQLTable = table
        self.num_tuples: int = num_tuples # Amount for tuples for vectorization

    @abstractmethod
    def __next__(self) -> dict: # Returns the next Record (Volcano)
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def open(self) -> 'Operator': # Init the operator (Volcano)
        raise NotImplementedError()

    @abstractmethod
    def next_vectorized(self) -> list[dict]: # Returns the next <num_tuples> Record
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None: # Closes the operator (Volcano)
        raise NotImplementedError()

    def get_description(self) -> str:
        return str(self)

    def get_structure(self) -> tuple[str, list] | str: # Structure for visualization
        return f"{id(self)}:{self.get_description()}"

    def fetch_all(self): # Interface for user to retrieve all records
        return list(self)

    def fetch_one(self): # Interface for user to retrieve next record
        return next(self)