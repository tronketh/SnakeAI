from collections import namedtuple
from enum import Enum


class Direction(Enum):
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'


Point = namedtuple('Point', ['x', 'y'])
