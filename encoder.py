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


def rgb_to_ycbcr(img: Image.Image):
    """Преобразует RGB изображение (PIL) в компоненты Y, Cb, Cr (numpy float32)."""
    arr = np.asarray(img).astype(np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B

    return Y, Cb, Cr


def downsample_channel(channel: np.ndarray, factor: int = 2):
    """Уменьшает разрешение канала в factor раз по каждой оси усреднением."""
    h, w = channel.shape
    h_trim = h - (h % factor)
    w_trim = w - (w % factor)
    channel = channel[:h_trim, :w_trim]

    return channel.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def split_into_blocks(channel: np.ndarray, block_size: int = 8):
    """Разбивает канал на блоки с добавлением паддинга нулями."""
    h, w = channel.shape

    h_blocks = (h + block_size - 1) // block_size
    w_blocks = (w + block_size - 1) // block_size

    padded_h = h_blocks * block_size
    padded_w = w_blocks * block_size
    padded = np.zeros((padded_h, padded_w), dtype=channel.dtype)
    padded[:h, :w] = channel

    blocks = np.zeros((h_blocks, w_blocks, block_size, block_size), dtype=channel.dtype)
    for i in range(h_blocks):
        for j in range(w_blocks):
            blocks[i, j] = padded[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]

    return blocks, (padded_h, padded_w)


def create_dct_matrix():
    """Создает матрицы для оптимизированного DCT"""
    M = np.zeros((8, 8))
    for u in range(8):
        for x in range(8):
            cu = 1/np.sqrt(2) if u == 0 else 1
            M[u,x] = 0.5 * cu * np.cos((2*x+1)*u*np.pi/16)

    return M


DCT_MATRIX = create_dct_matrix()


def fdct(block):
    """Оптимизированная версия FDCT"""
    return DCT_MATRIX @ block @ DCT_MATRIX.T


def idct(coefficients):
    """Оптимизированная версия IDCT"""
    return DCT_MATRIX.T @ coefficients @ DCT_MATRIX


def scale_quantization_matrix(Q, quality):
    quality = np.clip(quality, 1, 100)

    if quality == 100:
        return Q.copy()

    scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
    scaled_Q = np.round((Q * scale + 50) / 100).clip(1, 255)

    return scaled_Q.astype(np.uint8)


def quantize_block(dct_block: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    """Квантует блок DCT коэффициентов."""
    return np.round(dct_block / quant_matrix).astype(np.int32)


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


def zigzag_scan(matrix: np.ndarray, indices: list = ZIGZAG_INDICES_8x8):
    """Преобразует 2D матрицу в 1D вектор по зигзагу."""
    return np.array([matrix[i] for i in indices])


def get_category(value):
    """Определяет категорию (число бит) для значения."""
    if value == 0:
        return 0
    abs_val = abs(int(value))

    return abs_val.bit_length()


def encode_ac_coefficients(ac_zigzag_coeffs: np.ndarray):
    """
    :param ac_zigzag_coeffs: коэффициенты AC после зигзагирования
    :return: (RUNSIZE, LENGTH), VAL
    """
    encoded_symbols = []
    encoded_values = []
    zero_run = 0

    for coeff in ac_zigzag_coeffs:
        if coeff == 0:
            zero_run += 1
            if zero_run == 16:
                encoded_symbols.append((15, 0)) # ZRL
                zero_run = 0
        else:
            while zero_run >= 16:
                encoded_symbols.append((15, 0))
                zero_run -= 16

            category = get_category(coeff)
            encoded_symbols.append((zero_run, category))
            encoded_values.append(coeff)
            zero_run = 0

    if zero_run > 0:
        encoded_symbols.append((0, 0))  # EOB

    return encoded_symbols, encoded_values


class BitWriter:
    """Класс для побитовой записи данных."""

    def __init__(self):
        self.buffer = bytearray()
        self.accumulator = 0
        self.bit_count = 0

    def write_bits(self, bits_string: str):
        """Записывает биты из строки '0' и '1'."""
        for bit_char in bits_string:
            bit = int(bit_char)
            self.accumulator = (self.accumulator << 1) | bit
            self.bit_count += 1
            if self.bit_count == 8:
                self.buffer.append(self.accumulator)
                self.accumulator = 0
                self.bit_count = 0

    def write_value(self, value: int, category: int):
        """Записывает значение 'value' используя 'category' бит."""
        if category == 0:
            return

        if value > 0:
            bits_string = format(value, f'0{category}b')
        else:
            val_encoded = value + (1 << category) - 1
            bits_string = format(val_encoded, f'0{category}b')

        self.write_bits(bits_string)

    def flush(self):
        """Записывает оставшиеся биты в последний байт (дополняя нулями справа)."""
        if self.bit_count > 0:
            self.accumulator <<= (8 - self.bit_count)
            self.buffer.append(self.accumulator)
        final_bytes = bytes(self.buffer)
        self.buffer = bytearray()
        self.accumulator = 0
        self.bit_count = 0
        return final_bytes


def encode_dc_stream(dc_values: list, huffman_table: dict, writer: BitWriter):
    """Кодирует поток DC-разностей с использованием таблицы Хаффмана."""
    if not dc_values:
        return

    diffs = [dc_values[0]]
    for i in range(1, len(dc_values)):
        diff = dc_values[i] - dc_values[i - 1]
        diffs.append(diff)

    for diff in diffs:
        category = get_category(diff)
        try:
            huffman_code = huffman_table[category]
            writer.write_bits(huffman_code)
        except KeyError:
            print(f"ОШИБКА: Нет кода Хаффмана для DC категории {category} (diff={diff})")
            continue

        writer.write_value(diff, category)


def encode_ac_stream(ac_rle_symbols: list, ac_rle_values: list, huffman_table: dict, writer: BitWriter):
    """Кодирует поток AC RLE символов и значений с использованием таблицы Хаффмана."""
    value_idx = 0
    for symbol in ac_rle_symbols:
        try:
            huffman_code = huffman_table[symbol]
            writer.write_bits(huffman_code)
        except KeyError:
            print(f"ОШИБКА: Нет кода Хаффмана для AC символа {symbol}")
            if symbol != (0, 0) and symbol != (15, 0):
                if value_idx < len(ac_rle_values): value_idx += 1
            continue

        if symbol != (0, 0) and symbol != (15, 0):  # Не EOB и не ZRL
            run, size = symbol
            if size > 0:
                if value_idx < len(ac_rle_values):
                    value = ac_rle_values[value_idx]
                    writer.write_value(value, size)
                    value_idx += 1
                else:
                    print(f"ОШИБКА: Не хватает значений для AC символа {symbol}")


def process_channel(channel_blocks: np.ndarray, quant_matrix: np.ndarray):
    """Обрабатывает все блоки одного канала: DCT, Q, Zigzag, RLE."""
    num_h_blocks, num_w_blocks, _, _ = channel_blocks.shape
    all_dc_values = []
    all_ac_rle_symbols = []
    all_ac_rle_values = []

    for i in range(num_h_blocks):
        for j in range(num_w_blocks):
            block = channel_blocks[i, j] - 128.0

            dct_block = fdct(block)
            quant_block = quantize_block(dct_block, quant_matrix)
            zigzag_coeffs = zigzag_scan(quant_block)

            dc_value = zigzag_coeffs[0]
            ac_coeffs = zigzag_coeffs[1:]

            rle_symbols, rle_values = encode_ac_coefficients(ac_coeffs)

            all_dc_values.append(dc_value)
            all_ac_rle_symbols.extend(rle_symbols)
            all_ac_rle_values.extend(rle_values)

    return all_dc_values, all_ac_rle_symbols, all_ac_rle_values


def encode_image(img: Image.Image, quality: int = 75):
    """
    Выполняет все шаги кодирования нашего легендарного JPEG-подобного формата.
    :return: b"BYTESTRING"
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    original_width, original_height = img.size

    Y, Cb, Cr = rgb_to_ycbcr(img)

    Cb_downsampled = downsample_channel(Cb)
    Cr_downsampled = downsample_channel(Cr)

    Y_blocks, Y_padded_shape = split_into_blocks(Y)
    Cb_blocks, Cb_padded_shape = split_into_blocks(Cb_downsampled)
    Cr_blocks, Cr_padded_shape = split_into_blocks(Cr_downsampled)

    QY_scaled = scale_quantization_matrix(Q_Y, quality)
    QC_scaled = scale_quantization_matrix(Q_C, quality)

    y_dc, y_ac_sym, y_ac_val = process_channel(Y_blocks, QY_scaled)
    cb_dc, cb_ac_sym, cb_ac_val = process_channel(Cb_blocks, QC_scaled)
    cr_dc, cr_ac_sym, cr_ac_val = process_channel(Cr_blocks, QC_scaled)

    writer = BitWriter()

    encode_dc_stream(y_dc, HUFFMAN_DC_LUMINANCE, writer)
    encode_ac_stream(y_ac_sym, y_ac_val, HUFFMAN_AC_LUMINANCE, writer)

    encode_dc_stream(cb_dc, HUFFMAN_DC_CHROMINANCE, writer)
    encode_ac_stream(cb_ac_sym, cb_ac_val, HUFFMAN_AC_CHROMINANCE, writer)

    encode_dc_stream(cr_dc, HUFFMAN_DC_CHROMINANCE, writer)
    encode_ac_stream(cr_ac_sym, cr_ac_val, HUFFMAN_AC_CHROMINANCE, writer)

    encoded_data = writer.flush()

    header = struct.pack(">2sHHB", b"JP", original_width, original_height, quality)

    print("Изображение успешно закодировано.\n")

    return header + encoded_data
