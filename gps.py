#!/usr/bin/env python
# -*- encoding: utf-8 -*-

roads = {
    '276183': [(114.011574, 22.590562),
               (114.029255, 22.593177),
               (114.029599, 22.589334),
               (114.012647, 22.584024)],
    '276184': [(114.011574, 22.590562),
               (114.029255, 22.593177),
               (114.029599, 22.589334),
               (114.012647, 22.584024)],
    '275911': [],

}


def get_block_rec(block):
    # 上下左右
    block_rec = [max([i[1] for i in block]),
                 min([i[1] for i in block]),
                 min([i[0] for i in block]),
                 max([i[0] for i in block])]
    return block_rec


def in_block(sample, block, block_rec):
    # 初筛，上下左右
    x, y = sample
    if y > block_rec[0] or \
            y < block_rec[1] or \
            x < block_rec[2] or \
            x > block_rec[3]:
        return False
    # 细筛，计算四边形
    A, B, C, D = block
    # 顺时针的格子顶点，若逆时针方向则false
    a = (B[0]-A[0])*(y-A[1])-(B[1]-A[1])*(x-A[0])
    b = (C[0]-B[0])*(y-B[1])-(C[1]-B[1])*(x-B[0])
    c = (D[0]-C[0])*(y-C[1])-(D[1]-C[1])*(x-C[0])
    d = (A[0]-D[0])*(y-D[1])-(A[1]-D[1])*(x-D[0])
    if (a > 0 and b > 0 and c > 0 and d > 0) or \
            (a < 0 and b < 0 and c < 0 and d < 0):
        return True
    return False
