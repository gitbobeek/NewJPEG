# В это файле содержатся общие элементы для кодирования и декодирования изображений.
# А именно: матрицы квантования (и scale), функция зиг-заг обхода и DCT-II матрица.
# Также (в основном по приколу) все импорты находятся тут.

import struct

import numpy as np
from PIL import Image

from codes import *

Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def create_dct_matrix():
    """Создает матрицы для оптимизированного DCT"""
    M = np.zeros((8, 8))
    for u in range(8):
        for x in range(8):
            cu = 1/np.sqrt(2) if u == 0 else 1
            M[u,x] = 0.5 * cu * np.cos((2*x+1)*u*np.pi/16)

    return M


DCT_MATRIX = create_dct_matrix()


def scale_quantization_matrix(Q, quality):
    quality = np.clip(quality, 1, 100)

    if quality == 50:
        return Q.copy()

    scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
    scaled_Q = np.round((Q * scale + 50) / 100).clip(1, 255)

    return scaled_Q.astype(np.uint8)


def zigzag_indices(N=8):
    """Генерирует индексы для зигзаг-сканирования матрицы N x N."""
    indices = []
    row, col = 0, 0
    going_up = True

    for _ in range(N * N):
        indices.append((row, col))
        if going_up:
            if col == N - 1:
                row += 1
                going_up = False
            elif row == 0:
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:  # going_down
            if row == N - 1:
                col += 1
                going_up = True
            elif col == 0:
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1
    return indices


ZIGZAG_INDICES_8x8 = zigzag_indices(8)