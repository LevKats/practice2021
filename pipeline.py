from collections.abc import Callable
from collections.abc import Sequence
from typing import Any


class Pipeline:
    def __init__(self, *args: Callable[[Sequence[Any], Sequence[Any]]]):
        self.__functions = args

    def __call__(self, array: Sequence[Any]) -> Sequence[Any]:
        for func in self.__functions:
            array = func(array)
        return array
