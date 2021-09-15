#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 19:06:05 2021

@author: af1tang
"""

import setuptools

setuptools.setup( 
    name = "convogym",
    version = "0.1.0",
    url = "https://github.com/af1tang/convogym",
    author = "Fengyi (Andy) Tang",
    author_email = "af1tang2@gmail.com",
    description = """
    A gym environment to train open-domain conversational agents through 
    active learning and self-play.
    """,
    long_description = open('README.md').read(),
    package = setuptools.find_packages(),
    install_require = [
        "pytorch==1.4.0",
        "numpy>=1.20.2",
        "scipy>=1.6.2",
        "pandas>=1.2.4",
        "transformers==4.10.0",
        "dotenv",
        "tqdm",
    ],
    classifiers = [
        "Programming Language :: Python :: 3", 
        "Programming Language :: Python :: 3.7",
    ],
  
)