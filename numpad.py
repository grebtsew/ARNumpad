from enum import Enum

"""
Represent Landmarks to numpad coordinates for left and right hand.
"""

class NumpadValues(Enum):
    Key7 = "7",
    Key8 = "8",
    Key9 = "9",
    Key6 = "6",
    Key5 = "5",
    Key4 = "4",
    Key3 = "3",
    Key2 = "2",
    Key1 = "1",
    Key0 = "0",
    Del = "del",
    Enter = "enter"

class RightNumpadPoints(Enum):
    Key7 = "8",
    Key8 = "7",
    Key9 = "6",
    Key6 = "10",
    Key5 = "11",
    Key4 = "12",
    Key3 = "16",
    Key2 = "15",
    Key1 = "14",
    Key0 = "18",
    Del = "19",
    Enter = "20",
    Thumb = "4"

class LeftNumpadPoints(Enum):
    Key7 = 6,
    Key8 = 7,
    Key9 = 8,
    Key6 = 12,
    Key5 = 11,
    Key4 = 10,
    Key3 = 14,
    Key2 = 15,
    Key1 = 16,
    Key0 = 20,
    Del = 19,
    Enter = 18,
    Thumb = 4