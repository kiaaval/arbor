from typing import TypedDict


class feature(TypedDict):
    name: str
    array: list[str]


class branch(TypedDict):
    value: str
    numerator: int
    yes_no_arr: list[str]
    indices: list[int]


class features_branches(TypedDict):
    name: str
    total: int
    branches: list[branch]


class feature_IG(TypedDict):
    name: str
    IG: float
