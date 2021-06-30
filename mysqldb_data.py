#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 02:31:53 2021

@author: af1tang
"""
import numpy as np
import csv

with open('/home/af1tang/Desktop/tmp/db1.csv', 'r') as f: 
    reader = csv.reader(f, delimiter='|')
    lines, dat = [],[]
    for line in reader:
        if len(line) == 3:
            dat.append( int(line[-2].strip()) )
        elif len(line) == 7:
            dat.append( [int(k.strip()) for k in line[-6:-1]] )
        if len(dat) > 1:
            lines.append(dat)
            dat = []
            
with open('/home/af1tang/Desktop/tmp/db2.csv', 'r') as f: 
    reader = csv.reader(f, delimiter='|')
    lines2=[]
    for line in reader:
        if len(line) == 8:
            lines2.append( [line[-4].split(), line[-3].split()] )
            
            
data1 = {0:[], 1:[]}
for line in lines:
    if line[0] == 0:
        data1[0].append(line[1])
    else:
        data1[1].append(line[1])
        
p = np.array(data1[0]); d = np.array(data1[1])


prec1s, prec5s, rec5s = [], [],[]
for line in lines2:
    y = line[0]; yhat = line[1]
    if len(yhat) >0:
        prec1 = float(yhat[0] in y)
        hits = set(yhat).intersection(set(y))
        prec5 = len(hits) / len(yhat) * 1.0
        rec5 = len(hits) / len(y) * 1.0
        prec1s.append(prec1)
        prec5s.append(prec5)
        rec5s.append(rec5)
    