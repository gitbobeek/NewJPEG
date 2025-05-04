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

def scale_quantization_matrix(Q, quality):
    """Масштабирует матрицу квантования Q по уровню качества (1-100)."""
    if quality <= 0: quality = 1
    if quality > 100: quality = 100

    if quality == 100:
        scaled_Q = Q.copy()  # Просто копируем
    elif quality < 50:
        scale = 5000 / quality
        scaled_Q = np.floor((Q * scale + 50) / 100)
    else:
        scale = 200 - 2 * quality
        scaled_Q = np.floor((Q * scale + 50) / 100)

    # Все значения должны быть в диапазоне [1, 255]
    scaled_Q = np.clip(scaled_Q, 1, 255)
    return scaled_Q.astype(np.uint8)

def create_dct_matrix(N: int = 8):
    """Создает матрицу N x N для ортонормированного DCT-II."""
    M = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            M[k, n] = alpha * np.cos(((2 * n + 1) * k * np.pi) / (2 * N))
    return M

# Precompute DCT matrix (and its transpose for IDCT)
DCT_MATRIX_8x8 = create_dct_matrix(8)
IDCT_MATRIX_8x8 = DCT_MATRIX_8x8.T # Inverse DCT is the transpose for orthonormal matrices

def inverse_dct_2d(dct_block: np.ndarray, idct_matrix: np.ndarray = IDCT_MATRIX_8x8) -> np.ndarray:
    """Выполняет 2D Inverse DCT блока."""
    temp = np.dot(idct_matrix, dct_block)
    return np.dot(temp, idct_matrix.T) # Note: IDCT is M.T @ block @ M.T, which is (M @ block.T @ M).T = M @ block @ M.T if M is orthogonal/orthonormal. Here M=DCT_MATRIX_8x8, so IDCT is M.T @ block @ M.

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

def inverse_zigzag_scan(coeffs_1d: np.ndarray, indices: list = ZIGZAG_INDICES_8x8, N: int = 8) -> np.ndarray:
    """Преобразует 1D вектор (по зигзагу) обратно в 2D матрицу N x N."""
    matrix = np.zeros((N, N), dtype=coeffs_1d.dtype)
    for i, (row, col) in enumerate(indices):
        matrix[row, col] = coeffs_1d[i]
    return matrix

def decode_value(bits_string: str, category: int) -> int:
    """
    Декодирует значение из его битового представления и категории.
    Это обратная операция к BitWriter.write_value.
    """
    if category == 0:
        return 0 # Should not happen with category 0, but for safety

    value_encoded = int(bits_string, 2)

    # Если MSB (первый бит в bits_string) равен 1, значение положительное.
    # Если MSB равен 0, значение отрицательное (добавлено смещение).
    if bits_string[0] == '1':
        return value_encoded
    else:
        # Обратное смещение для отрицательных чисел
        # value_encoded = value + (1 << category) - 1
        # value = value_encoded - (1 << category) + 1
        return value_encoded - (1 << category) + 1


# === Декомпрессор ===

class BitReader:
    """Класс для побитового чтения данных."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_index = 0
        self.bit_index = 0 # 0 is MSB, 7 is LSB

    def read_bit(self) -> int:
        """Считывает один бит."""
        if self.byte_index >= len(self.data):
            raise IndexError("Attempt to read past end of data")

        byte = self.data[self.byte_index]
        # Извлекаем нужный бит
        bit = (byte >> (7 - self.bit_index)) & 1

        self.bit_index += 1
        if self.bit_index == 8:
            self.bit_index = 0
            self.byte_index += 1

        return bit

    def read_bits(self, num_bits: int) -> str:
        """Считывает указанное количество бит и возвращает их как строку '0' и '1'."""
        if num_bits < 0:
             raise ValueError("Number of bits must be non-negative")
        if num_bits == 0:
            return ""

        bits_string = ""
        for _ in range(num_bits):
            try:
                bits_string += str(self.read_bit())
            except IndexError:
                 # Ran out of bits - should not happen if data is complete
                 raise ValueError("Unexpected end of data while reading bits")
        return bits_string

    def read_huffman_code(self, huffman_table: dict) -> int | tuple: # Уточняем возможные возвращаемые типы
        """
        Читает биты из потока до тех пор, пока не будет найден код Хаффмана
        в данной таблице. Возвращает декодированный символ (ключ из таблицы).
        """
        current_code = ""
        inverse_huffman_table = huffman_table # Ожидаем инвертированную таблицу

        # Добавим диагностику
        print(f"DEBUG: Starting Huffman read. Table size: {len(inverse_huffman_table)}")
        print(f"DEBUG: Table keys sample: {list(inverse_huffman_table.keys())[:10]}...") # Первые 10 ключей для примера

        while True:
            try:
                bit = self.read_bit()
                current_code += str(bit)
            except IndexError:
                # print(f"DEBUG: Ran out of bits while reading Huffman code. Current code: {current_code}")
                raise ValueError("Unexpected end of data while reading Huffman code")

            print(f"DEBUG: Checking code: {current_code}")
            # Перед вызовом decode_image

            if current_code in inverse_huffman_table:
                symbol = inverse_huffman_table[current_code]
                print(f"DEBUG: Found match! Code: {current_code}, Symbol: {symbol}")
                return symbol

            # Проверка на слишком длинный код
            if len(current_code) > 16: # Max possible length for standard JPEG is 16
                 print(f"DEBUG: Code became too long: {current_code}")
                 raise ValueError(f"Huffman code too long: {current_code}. Possible data corruption.")

def build_inverse_huffman_table(huffman_table: dict) -> dict:
    """Строит инвертированную таблицу Хаффмана (код: символ)."""
    return {code: symbol for symbol, code in huffman_table.items()}

# Build inverse tables once
INV_HUFFMAN_DC_L = build_inverse_huffman_table(HUFFMAN_DC_LUMINANCE)
INV_HUFFMAN_DC_C = build_inverse_huffman_table(HUFFMAN_DC_CHROMINANCE)
INV_HUFFMAN_AC_L = build_inverse_huffman_table(HUFFMAN_AC_LUMINANCE)
INV_HUFFMAN_AC_C = build_inverse_huffman_table(HUFFMAN_AC_CHROMINANCE)


def decode_ac_coefficients(reader: BitReader, huffman_table: dict) -> np.ndarray:
    """
    Декодирует AC коэффициенты для одного блока из битового потока
    с использованием таблицы Хаффмана.
    """
    ac_coeffs = np.zeros(63, dtype=np.int32)
    ac_index = 0 # Index into the 63 AC coefficients array

    while ac_index < 63:
        symbol = reader.read_huffman_code(huffman_table)

        if symbol == (0, 0): # EOB (End of Block)
            # Remaining coefficients are zero, which is already handled by np.zeros
            break
        elif symbol == (15, 0): # ZRL (Zero Run Length 16)
            ac_index += 16 # Skip 16 coefficients (they are zero)
            if ac_index > 63:
                # Should not happen in a valid stream, but safety check
                ac_index = 63
        else:
            # (Run, Size) symbol
            run, size = symbol
            if run > 0:
                ac_index += run # Skip 'run' zero coefficients
                if ac_index >= 63:
                     # Safety check: Ran out of AC slots before placing the value
                     # This might indicate a stream issue or unexpected RLE
                     print(f"Warning: Run length {run} exceeded remaining AC slots ({63 - (ac_index - run)}). Data potentially corrupted.")
                     ac_index = 63 # Cap at 63 and break
                     break # Stop processing this block's AC

            if size > 0:
                # Read the value bits
                value_bits = reader.read_bits(size)
                value = decode_value(value_bits, size)

                if ac_index < 63:
                    ac_coeffs[ac_index] = value
                    ac_index += 1
                else:
                    # This case should ideally be caught by the run check above,
                    # but added for extra safety if a run=0, size>0 symbol occurs at the very end.
                    print(f"Warning: Attempted to write AC value at index {ac_index} >= 63. Data potentially corrupted.")
                    break # Stop processing this block's AC

    # Ensure the array is exactly 63 elements (padding with zeros if EOB was premature or ZRL/Run ended past 63)
    return ac_coeffs[:63]


def decode_image(compressed_bytes: bytes) -> Image.Image:
    """
    Декодирует изображение из байтовой последовательности,
    сгенерированной encode_image.
    """
    # 1. Чтение заголовка
    # Header format: >2sHHB (JP, width, height, quality)
    header_size = struct.calcsize(">2sHHB")
    if len(compressed_bytes) < header_size:
        raise ValueError("Compressed data too short for header")

    signature, original_width, original_height, quality = struct.unpack_from(">2sHHB", compressed_bytes, 0)

    if signature != b"JP":
        raise ValueError(f"Invalid signature: Expected b'JP', got {signature}")

    # 2. Восстановление матриц квантования
    QY_scaled = scale_quantization_matrix(Q_Y, quality)
    QC_scaled = scale_quantization_matrix(Q_C, quality)

    # 3. Инициализация BitReader с данными после заголовка
    reader = BitReader(compressed_bytes[header_size:])

    # 4. Расчет размеров дополненных каналов
    # Assuming block size is 8 and downsampling factor for Cb/Cr is 2
    block_size = 8
    h_blocks_y = (original_height + block_size - 1) // block_size
    w_blocks_y = (original_width + block_size - 1) // block_size
    padded_h_y = h_blocks_y * block_size
    padded_w_y = w_blocks_y * block_size

    # Cb/Cr dimensions after downsampling and padding
    # The compressor downsamples THEN splits into blocks.
    # So padded Cb/Cr dims correspond to the padded Y dims // 2
    padded_h_c = (padded_h_y + 1) // 2 # Equivalent to (original_height // 2 + 7) // 8 * 8 ? No, it's just padded_h_y / 2 rounded up if needed
    padded_w_c = (padded_w_y + 1) // 2

    h_blocks_c = (original_height // 2 + block_size - 1) // block_size
    w_blocks_c = (original_width // 2 + block_size - 1) // block_size
    # No, the compressor uses the *original* dimensions for downsampling, then pads the result.
    # Cb/Cr downsampled size: (original_height + 1) // 2, (original_width + 1) // 2
    # Then split and padded.
    # Cb/Cr downsampled height: (original_height + 1) // 2
    # Cb/Cr downsampled width: (original_width + 1) // 2
    h_blocks_c = ((original_height + 1) // 2 + block_size - 1) // block_size
    w_blocks_c = ((original_width + 1) // 2 + block_size - 1) // block_size
    padded_h_c = h_blocks_c * block_size
    padded_w_c = w_blocks_c * block_size


    # Initialize arrays to hold reconstructed blocks
    y_blocks_recon = np.zeros((h_blocks_y, w_blocks_y, block_size, block_size), dtype=np.float32)
    cb_blocks_recon = np.zeros((h_blocks_c, w_blocks_c, block_size, block_size), dtype=np.float32)
    cr_blocks_recon = np.zeros((h_blocks_c, w_blocks_c, block_size, block_size), dtype=np.float32)

    # 5. Декодирование DC и AC коэффициентов для каждого канала
    last_dc_y = 0
    last_dc_cb = 0
    last_dc_cr = 0

    # --- Декодирование Y канала ---
    for i in range(h_blocks_y):
        for j in range(w_blocks_y):
            # DC
            dc_category = reader.read_huffman_code(INV_HUFFMAN_DC_L)
            if dc_category > 0:
                dc_value_bits = reader.read_bits(dc_category)
                dc_diff = decode_value(dc_value_bits, dc_category)
            else: # category 0, diff is 0
                 dc_diff = 0
                 # Note: The compressor's get_category returns 0 for 0.
                 # write_value does nothing for category 0.
                 # decode_value expects bits_string and category.
                 # If DC diff is 0, category is 0. Huffman table has 0 -> '00'.
                 # Need to handle category 0 correctly. dc_diff will be 0.

            current_dc_y = last_dc_y + dc_diff
            last_dc_y = current_dc_y

            # AC
            ac_coeffs_1d = decode_ac_coefficients(reader, INV_HUFFMAN_AC_L)

            # Combine DC and AC
            zigzag_coeffs = np.concatenate(([current_dc_y], ac_coeffs_1d))

            # Inverse Zigzag, Inverse Quantize, IDCT, Add 128
            quant_block_recon = inverse_zigzag_scan(zigzag_coeffs)
            dct_block_recon = quant_block_recon * QY_scaled # Inverse Quantization
            block_recon = inverse_dct_2d(dct_block_recon)

            # Add 128 back (since it was subtracted before DCT)
            y_blocks_recon[i, j] = block_recon + 128.0

    # --- Декодирование Cb канала ---
    for i in range(h_blocks_c):
        for j in range(w_blocks_c):
            # DC (using Chrominance tables)
            dc_category = reader.read_huffman_code(INV_HUFFMAN_DC_C)
            if dc_category > 0:
                dc_value_bits = reader.read_bits(dc_category)
                dc_diff = decode_value(dc_value_bits, dc_category)
            else:
                 dc_diff = 0

            current_dc_cb = last_dc_cb + dc_diff
            last_dc_cb = current_dc_cb

            # AC (using Chrominance tables)
            ac_coeffs_1d = decode_ac_coefficients(reader, INV_HUFFMAN_AC_C)

            # Combine DC and AC
            zigzag_coeffs = np.concatenate(([current_dc_cb], ac_coeffs_1d))

            # Inverse Zigzag, Inverse Quantize, IDCT, Add 128
            # Note: Q_C is used for both Cb and Cr
            quant_block_recon = inverse_zigzag_scan(zigzag_coeffs)
            dct_block_recon = quant_block_recon * QC_scaled # Inverse Quantization
            block_recon = inverse_dct_2d(dct_block_recon)

            # Add 128 back (since it was subtracted before DCT)
            cb_blocks_recon[i, j] = block_recon + 128.0

    # --- Декодирование Cr канала ---
    for i in range(h_blocks_c):
        for j in range(w_blocks_c):
            # DC (using Chrominance tables)
            dc_category = reader.read_huffman_code(INV_HUFFMAN_DC_C)
            if dc_category > 0:
                dc_value_bits = reader.read_bits(dc_category)
                dc_diff = decode_value(dc_value_bits, dc_category)
            else:
                dc_diff = 0

            current_dc_cr = last_dc_cr + dc_diff
            last_dc_cr = current_dc_cr

            # AC (using Chrominance tables)
            ac_coeffs_1d = decode_ac_coefficients(reader, INV_HUFFMAN_AC_C)

            # Combine DC and AC
            zigzag_coeffs = np.concatenate(([current_dc_cr], ac_coeffs_1d))

            # Inverse Zigzag, Inverse Quantize, IDCT, Add 128
            quant_block_recon = inverse_zigzag_scan(zigzag_coeffs)
            dct_block_recon = quant_block_recon * QC_scaled # Inverse Quantization
            block_recon = inverse_dct_2d(dct_block_recon)

            # Add 128 back (since it was subtracted before DCT)
            cr_blocks_recon[i, j] = block_recon + 128.0

    # 6. Сборка блоков в полные каналы (с учетом паддинга)
    Y_recon_padded = np.zeros((padded_h_y, padded_w_y), dtype=np.float32)
    Cb_recon_padded = np.zeros((padded_h_c, padded_w_c), dtype=np.float32)
    Cr_recon_padded = np.zeros((padded_h_c, padded_w_c), dtype=np.float32)

    for i in range(h_blocks_y):
        for j in range(w_blocks_y):
            Y_recon_padded[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = y_blocks_recon[i, j]

    for i in range(h_blocks_c):
        for j in range(w_blocks_c):
            Cb_recon_padded[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = cb_blocks_recon[i, j]
            Cr_recon_padded[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = cr_blocks_recon[i, j]


    # 7. Апсемплинг Cb и Cr
    # This needs to reverse the downsampling. The compressor used averaging.
    # We need to repeat the downsampled values.
    # Assuming downsample factor was 2
    Cb_upsampled = np.zeros((padded_h_y, padded_w_y), dtype=np.float32)
    Cr_upsampled = np.zeros((padded_h_y, padded_w_y), dtype=np.float32)

    # Repeat each Cb/Cr pixel 2x2 times
    for i in range(padded_h_c):
        for j in range(padded_w_c):
            Cb_upsampled[i*2 : i*2 + 2, j*2 : j*2 + 2] = Cb_recon_padded[i, j]
            Cr_upsampled[i*2 : i*2 + 2, j*2 : j*2 + 2] = Cr_recon_padded[i, j]

    # Ensure upsampled dimensions match padded Y dimensions
    Cb_upsampled = Cb_upsampled[:padded_h_y, :padded_w_y]
    Cr_upsampled = Cr_upsampled[:padded_h_y, :padded_w_y]


    # 8. YCbCr -> RGB конвертация
    # Use the standard JPEG conversion formulas.
    # Note: The compressor added 128 to Cb/Cr and potentially to Y before DCT.
    # The +128 after IDCT brings values back to their ranges.
    # Y values are roughly 0-255. Cb/Cr are roughly 0-255 (centered around 128).
    # Formulas require Y, Cb-128, Cr-128.

    Y_final = Y_recon_padded
    Cb_final = Cb_upsampled - 128.0
    Cr_final = Cr_upsampled - 128.0

    R = Y_final + 1.402 * Cr_final
    G = Y_final - 0.344136 * Cb_final - 0.714136 * Cr_final
    B = Y_final + 1.772 * Cb_final

    # 9. Клиппирование значений и преобразование в uint8
    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    # Combine channels back into an image array
    rgb_array_padded = np.stack([R, G, B], axis=-1)

    # 10. Обрезка до оригинальных размеров
    rgb_array_original = rgb_array_padded[:original_height, :original_width]

    # 11. Создание PIL Image
    img_recon = Image.fromarray(rgb_array_original, 'RGB')

    return img_recon


# Example Usage (assuming you have the encoder code and codes.py):
