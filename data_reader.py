#!/usr/bin/env python
# -*- encoding: utf-8 -*-


class Reader:
    def __init__(self, path, model="r"):
        super().__init__()
        self.path = path
        self.model = model

    def __iter__(self):
        for line in open(self.path, self.model):
            yield line.split()


data = Reader("20191201_20191220.csv")
i = 0
for line in data:
    print(line)
    break
