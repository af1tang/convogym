#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:23:34 2021

@author: af1tang
"""

flatten = lambda l: [item for sublist in l for item in sublist]

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def split_by_index(seq, sep):
    result = []
    for el in seq:
        result.append(el)
        if el == sep:
            yield result
            result = []