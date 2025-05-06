import struct

import numpy as np
from PIL import Image

from codes import *
from encoder import ZIGZAG_INDICES_8x8, DCT_MATRIX, scale_quantization_matrix, Q_Y, Q_C


def inverse_zigzag_scan(vector: np.ndarray, indices: list = ZIGZAG_INDICES_8x8):
    """Преобразует зигзаг-вектор обратно в 2D матрицу 8x8"""
    if len(vector) != len(indices):
        raise ValueError(f"vector ({len(vector)}) не совпадает с indices ({len(indices)})")
    matrix = np.zeros((8, 8), dtype=vector.dtype)
    for i, (r, c) in enumerate(indices):
        matrix[r, c] = vector[i]
    return matrix


def dequantize_block(quantized_block: np.ndarray, scaled_quant_matrix: np.ndarray) -> np.ndarray:
    """Деквантует блок DCT коэффициентов"""
    return quantized_block.astype(np.float64) * scaled_quant_matrix


def idct(coefficients):
    """Оптимизированная версия IDCT"""
    res = DCT_MATRIX.T @ coefficients @ DCT_MATRIX
    return res + 128


def reassemble_blocks(blocks: np.ndarray, padded_height: int, padded_width: int) -> np.ndarray:
    """Собирает 2D канал из массива блоков."""
    n_blocks_h, n_blocks_w, block_size, _ = blocks.shape
    channel = np.zeros((padded_height, padded_width), dtype=blocks.dtype)

    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            row_start = i * block_size
            row_end = row_start + block_size
            col_start = j * block_size
            col_end = col_start + block_size
            channel[row_start:row_end, col_start:col_end] = blocks[i, j]

    return channel


def upsample_channel(channel: np.ndarray, factor: int = 2) -> np.ndarray:
    """Увеличивает разрешение канала в factor раз (простое повторение пикселей)."""
    upsampled = np.repeat(channel, factor, axis=1)
    upsampled = np.repeat(upsampled, factor, axis=0)
    return upsampled


def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> Image.Image:
    """Преобразует каналы Y, Cb, Cr (numpy) обратно в RGB изображение (PIL)."""
    Y = Y.astype(np.float64)
    Cb = Cb.astype(np.float64)
    Cr = Cr.astype(np.float64)

    # Сдвигаем Cb и Cr обратно
    Cb_shifted = Cb - 128.0
    Cr_shifted = Cr - 128.0

    R = Y + 1.402 * Cr_shifted
    G = Y - 0.344136 * Cb_shifted - 0.714136 * Cr_shifted
    B = Y + 1.772 * Cb_shifted

    rgb_array = np.stack([R, G, B], axis=-1)
    rgb_array_clipped = np.round(rgb_array).clip(0, 255)

    return Image.fromarray(rgb_array_clipped.astype(np.uint8), 'RGB')


def invert_huffman_table(huffman_table):
    return {v: k for k, v in huffman_table.items()}


INV_HUFFMAN_DC_LUMINANCE = invert_huffman_table(HUFFMAN_DC_LUMINANCE)
INV_HUFFMAN_AC_LUMINANCE = invert_huffman_table(HUFFMAN_AC_LUMINANCE)
INV_HUFFMAN_DC_CHROMINANCE = invert_huffman_table(HUFFMAN_DC_CHROMINANCE)
INV_HUFFMAN_AC_CHROMINANCE = invert_huffman_table(HUFFMAN_AC_CHROMINANCE)


class BitReader:
    def __init__(self, byte_data: bytes):
        self.byte_data = byte_data
        self.byte_index = 0
        self.bit_offset = 0
        self.current_byte = 0
        self._load_next_byte()

    def _load_next_byte(self):
        if self.byte_index < len(self.byte_data):
            self.current_byte = self.byte_data[self.byte_index]
            self.byte_index += 1
            self.bit_offset = 0
            return True
        return False

    def read_bit(self) -> int:
        if self.bit_offset == 8:
            if not self._load_next_byte():
                raise EOFError("Неожиданный конец потока при чтении бита.")

        bit = (self.current_byte >> (7 - self.bit_offset)) & 1
        self.bit_offset += 1
        return bit

    def read_bits(self, num_bits: int) -> int:
        """Читает num_bits и возвращает их как целое число."""
        if num_bits == 0:
            return 0
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value

    def decode_huffman_symbol(self, inv_huffman_table: dict):
        """Декодирует следующий символ Хаффмана из потока."""
        current_code = ""
        while True:
            bit = self.read_bit()
            current_code += str(bit)
            if current_code in inv_huffman_table:
                return inv_huffman_table[current_code]
            if len(current_code) > 32:
                raise ValueError(f"Не найден код Хаффмана для последовательности: {current_code[:10]}...")

    def decode_value(self, category: int) -> int:
        """Декодирует значение на основе его категории (количества бит)."""
        if category == 0:
            return 0

        value_bits = self.read_bits(category)

        if value_bits < (1 << (category - 1)):
            return value_bits - ((1 << category) - 1)
        else:
            # Положительное значение
            return value_bits

def decode_dc_coefficients(num_dc_coeffs: int, inv_huffman_table: dict, reader: BitReader) -> list:
    """Декодирует DC-коэффициенты из потока."""
    dc_values = []
    prev_dc = 0
    for _ in range(num_dc_coeffs):
        category = reader.decode_huffman_symbol(inv_huffman_table)
        diff = reader.decode_value(category)
        current_dc = prev_dc + diff
        dc_values.append(current_dc)
        prev_dc = current_dc
    return dc_values

def decode_ac_coefficients(reader: BitReader, inv_huffman_table: dict) -> list:
    """
    Декодирует AC-коэффициенты для ОДНОГО блока 8x8.
    Возвращает список из 63 AC-коэффициентов.
    """
    ac_coeffs = [0] * 63
    idx = 0
    while idx < 63:
        symbol = reader.decode_huffman_symbol(inv_huffman_table)

        if symbol == (0, 0):  # EOB
            break
        elif symbol == (15, 0):  # ZRL
            idx += 16
        else:
            run, size = symbol
            idx += run
            if idx < 63:
                value = reader.decode_value(size)
                ac_coeffs[idx] = value
                idx += 1
            else:
                break
    return ac_coeffs

def decode_channel_data(num_h_blocks: int, num_w_blocks: int,
                        reader: BitReader,
                        inv_dc_huff_table: dict, inv_ac_huff_table: dict,
                        quant_matrix: np.ndarray,
                        padded_shape: tuple, original_shape: tuple):
    """Декодирует все блоки одного канала, выполняет IDCT, деквантование."""

    num_total_blocks = num_h_blocks * num_w_blocks
    reconstructed_blocks_4d = np.zeros((num_h_blocks, num_w_blocks, 8, 8), dtype=np.float64)

    all_dc_values = decode_dc_coefficients(num_total_blocks, inv_dc_huff_table, reader)

    block_idx = 0
    for i in range(num_h_blocks):
        for j in range(num_w_blocks):
            dc_val = all_dc_values[block_idx]
            ac_vals_list = decode_ac_coefficients(reader, inv_ac_huff_table)

            zigzag_coeffs = np.array([dc_val] + ac_vals_list, dtype=np.float64)
            quant_block_2d = inverse_zigzag_scan(zigzag_coeffs)
            dequant_block = dequantize_block(quant_block_2d, quant_matrix)
            pixel_block = idct(dequant_block)

            reconstructed_blocks_4d[i, j] = pixel_block
            block_idx += 1

    padded_channel = reassemble_blocks(reconstructed_blocks_4d, padded_shape[0], padded_shape[1])
    original_channel = padded_channel[:original_shape[0], :original_shape[1]]

    return original_channel


def decode_image_data(encoded_bytes: bytes) -> Image.Image:
    """
    Декодирует данные, созданные вашим encode_image.
    """
    header_format = ">2sHHB"
    header_size = struct.calcsize(header_format)

    magic, original_width, original_height, quality = struct.unpack(header_format, encoded_bytes[:header_size])

    if magic != b"JP":
        raise ValueError("Неверный magic number в заголовке.")

    print(f"Декодирование изображения: {original_width}x{original_height}, Quality: {quality}")

    huffman_data = encoded_bytes[header_size:]
    reader = BitReader(huffman_data)

    QY_scaled = scale_quantization_matrix(Q_Y, quality)
    QC_scaled = scale_quantization_matrix(Q_C, quality)

    h_y, w_y = original_height, original_width

    factor = 2
    h_cb = (h_y // factor)
    w_cb = (w_y // factor)

    h_cb_trimmed_orig = h_y - (h_y % factor)
    w_cb_trimmed_orig = w_y - (w_y % factor)
    h_cb_orig = h_cb_trimmed_orig // factor
    w_cb_orig = w_cb_trimmed_orig // factor

    block_size = 8

    num_h_blocks_y = (h_y + block_size - 1) // block_size
    num_w_blocks_y = (w_y + block_size - 1) // block_size
    padded_h_y = num_h_blocks_y * block_size
    padded_w_y = num_w_blocks_y * block_size

    num_h_blocks_c = (h_cb_orig + block_size - 1) // block_size
    num_w_blocks_c = (w_cb_orig + block_size - 1) // block_size
    padded_h_c = num_h_blocks_c * block_size
    padded_w_c = num_w_blocks_c * block_size

    # print("Выпускайте кракена Y")
    Y_rec = decode_channel_data(num_h_blocks_y, num_w_blocks_y, reader,
                                INV_HUFFMAN_DC_LUMINANCE, INV_HUFFMAN_AC_LUMINANCE,
                                QY_scaled,
                                (padded_h_y, padded_w_y), (h_y, w_y))
    # print(f"Кракен Y выпущен. Размер: {Y_rec.shape}")

    # print("Выпускайте кракена Cb")
    Cb_rec_downsampled = decode_channel_data(num_h_blocks_c, num_w_blocks_c, reader,
                                           INV_HUFFMAN_DC_CHROMINANCE, INV_HUFFMAN_AC_CHROMINANCE,
                                           QC_scaled,
                                           (padded_h_c, padded_w_c), (h_cb_orig, w_cb_orig))
    # print(f"Кракен Cb выпущен (дундундунахахахах). Размер: {Cb_rec_downsampled.shape}")

    # print("Выпускайте Cr_акена")
    Cr_rec_downsampled = decode_channel_data(num_h_blocks_c, num_w_blocks_c, reader,
                                           INV_HUFFMAN_DC_CHROMINANCE, INV_HUFFMAN_AC_CHROMINANCE,
                                           QC_scaled,
                                           (padded_h_c, padded_w_c), (h_cb_orig, w_cb_orig))
    # print(f"Кракен Cr выпущен (дунай). Размер: {Cr_rec_downsampled.shape}")

    Cb_rec_upsampled = upsample_channel(Cb_rec_downsampled, factor=2)
    Cr_rec_upsampled = upsample_channel(Cr_rec_downsampled, factor=2)

    Cb_rec_upsampled = Cb_rec_upsampled[:h_y, :w_y]
    Cr_rec_upsampled = Cr_rec_upsampled[:h_y, :w_y]
    #print(f"Размеры Cb/Cr: {Cb_rec_upsampled.shape}")

    reconstructed_image = ycbcr_to_rgb(Y_rec, Cb_rec_upsampled, Cr_rec_upsampled)
    # print("Изображение преобразовано в RGB.")

    return reconstructed_image