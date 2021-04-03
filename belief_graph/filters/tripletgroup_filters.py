"""
Filters applied to triplet groups
"""
from ..data_classes import TripletGroup

class CountFilter:
    def __init__(self, count=5):
        self.count = 5

    def filter(self, tg: TripletGroup):
        if self.count > tg.count:
            return True
        return False

    def __call__(self, tg: TripletGroup):
        return self.filter(tg)
