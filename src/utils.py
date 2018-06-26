# -*- coding: utf-8 -*-
from time import time
import logging,inspect
import cPickle as pickle
from itertools import islice
import os.path

dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def returnPathStruc2vec():
    return dir_f.replace(" ","\ ")

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in xrange(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

    return





