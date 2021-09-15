#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:23:34 2021

@author: af1tang
"""

flatten = lambda l: [item for sublist in l for item in sublist]

def chunker(seq, size):
    """
    originally from
    https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def split_by_index(seq, sep):
    """
    originally from 
    https://stackoverflow.com/questions/15357830/splitting-a-list-based-on-a-delimiter-word
    """
    result = []
    for el in seq:
        result.append(el)
        if el == sep:
            yield result
            result = []