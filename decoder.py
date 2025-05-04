import numpy as np
from encoder import zigzag_indices

def idct_2d(coeffs: np.ndarray, dct_matrix: np.ndarray) -> np.ndarray:
    temp = np.dot(dct_matrix.T, coeffs)
    return np.dot(temp, dct_matrix)


def dequantize_block(quant_block: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    return (quant_block * quant_matrix).astype(np.float32)



def inverse_zigzag(vector, N=8):
    indices = zigzag_indices(N)
    matrix = np.zeros((N, N), dtype=vector.dtype)
    for idx, (i, j) in enumerate(indices):
        matrix[i, j] = vector[idx]
    return matrix


def decode_dc_differences(encoded_data):
    if not encoded_data:
        return []

    diffs = []
    for category, value in encoded_data:
        if category == 0:
            diffs.append(0)
        else:
            diffs.append(value)

    values = [diffs[0]]
    for i in range(1, len(diffs)):
        values.append(values[-1] + diffs[i])
    return values


def decode_ac_coefficients(encoded_ac, block_size=8):
    ac_coeffs = np.zeros(block_size * block_size - 1, dtype=np.int32)
    idx = 0

    for item in encoded_ac:
        if len(item) == 2 and item == (0, 0):  # EOB
            break
        elif len(item) == 2 and item[1] == 0:  # ZRL
            idx += 16
        else:
            zero_run, category, value = item
            idx += zero_run
            if idx >= len(ac_coeffs):
                break
            ac_coeffs[idx] = value
            idx += 1

    return ac_coeffs