#!/usr/bin/env python
# -*- coding: utf-8 -*-

_author__ = "Tim Krebs"
__email__ = "timkrebs9@gmail.com"
__version__ = "V1.0.0"



class Statistics:
    def __init__(self, debug=False) -> None:
        self.debug = debug

    def mean(x):
        return sum(x)/float(len(x))

    def min(x):
        return min(x)

    def max(x):
        return max(x)

    def median(x):
        idx = len(x) // 2
        if len(x) % 2:
            return sorted(x)[len(x)//2]
        return sum(sorted(x)[(len(x)//2) - 1:(len(x)//2) + 1]) / 2
