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


# === ЭТАП 1: Переход в цветовое пространство YCbCr ===
def rgb_to_ycbcr(img: Image.Image):
    """Преобразует RGB изображение (PIL) в компоненты Y, Cb, Cr (numpy float32)."""
    arr = np.asarray(img).astype(np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = - 0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

    # Важно: В JPEG обычно YCbCr центрируется вокруг 0 перед DCT,
    # вычитая 128 из Y и оставляя Cb, Cr как есть (т.к. они уже смещены).
    # Здесь Y не центрирован. Это может потребовать корректировки позже.
    # Для стандартного JPEG: Y = Y - 128
    # Y -= 128 # <- Раскомментировать для стандартного смещения уровня

    return Y, Cb, Cr


# === ЭТАП 2: Даунсэмплинг каналов Cb и Сr ===
def downsample_channel(channel: np.ndarray, factor: int = 2):
    """Уменьшает разрешение канала в factor раз по каждой оси усреднением."""
    h, w = channel.shape
    h_trim = h - (h % factor)
    w_trim = w - (w % factor)
    channel = channel[:h_trim, :w_trim]

    return channel.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


# === ЭТАП 3: Разбиение на блоки ===
def split_into_blocks(channel: np.ndarray, block_size: int = 8):
    """Разбивает канал на блоки с добавлением паддинга нулями."""
    h, w = channel.shape

    # Расчет количества блоков с учетом некратных размеров
    h_blocks = (h + block_size - 1) // block_size
    w_blocks = (w + block_size - 1) // block_size

    # Создание дополненного массива (паддинг нулями)
    padded_h = h_blocks * block_size
    padded_w = w_blocks * block_size
    padded = np.zeros((padded_h, padded_w), dtype=channel.dtype)
    padded[:h, :w] = channel  # Копирование исходных данных

    # Извлечение блоков
    blocks = np.zeros((h_blocks, w_blocks, block_size, block_size), dtype=channel.dtype)
    for i in range(h_blocks):
        for j in range(w_blocks):
            blocks[i, j] = padded[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]

    return blocks, (padded_h, padded_w)  # Возвращаем блоки и размер дополненного канала


# === ЭТАП 4: Дискретное косинусное преобразование (DCT) ===
def create_dct_matrix(N: int = 8):
    """Создает матрицу N x N для ортонормированного DCT-II."""
    M = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            M[k, n] = alpha * np.cos(((2 * n + 1) * k * np.pi) / (2 * N))
    return M


# Предварительно вычислим матрицу DCT, т.к. она не меняется
DCT_MATRIX_8x8 = create_dct_matrix(8)


def dct_2d(block: np.ndarray, dct_matrix: np.ndarray = DCT_MATRIX_8x8) -> np.ndarray:
    """Выполняет 2D DCT блока с использованием предвычисленной матрицы."""
    # Важно: Перед DCT значения пикселей (особенно Y) должны быть смещены
    # к нулю (обычно вычитанием 128). Это не сделано здесь или в rgb_to_ycbcr.
    # block = block - 128 # <- Раскомментировать, если Y не был смещен ранее
    temp = np.dot(dct_matrix, block)
    return np.dot(temp, dct_matrix.T)


# === ЭТАП 5: Квантование ===
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


def quantize_block(dct_block: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    """Квантует блок DCT коэффициентов."""
    return np.round(dct_block / quant_matrix).astype(np.int32)


# === ЭТАП 6 и 8: Упорядочивание коэффициентов (Zigzag) и RLE для AC ===

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
    abs_val = abs(int(value))  # Убедимся, что работаем с целым числом
    # Используем встроенный метод для битовой длины - эффективнее log2
    return abs_val.bit_length()


def encode_ac_coefficients(ac_zigzag_coeffs: np.ndarray):
    """
    Выполняет RLE-кодирование AC-коэффициентов (ожидает вектор ПОСЛЕ зигзага, БЕЗ DC).
    Возвращает список символов: (run, size) или (15, 0) для ZRL, (0, 0) для EOB.
    Также возвращает список соответствующих значений для ненулевых коэффициентов.
    """
    encoded_symbols = []
    encoded_values = []
    zero_run = 0

    for coeff in ac_zigzag_coeffs:  # Ожидается массив только из AC (63 элемента)
        if coeff == 0:
            zero_run += 1
            if zero_run == 16:  # Достигли ZRL
                encoded_symbols.append((15, 0))  # ZRL символ
                zero_run = 0
        else:
            # Ненулевой коэффициент найден
            while zero_run >= 16:  # Записываем предыдущие ZRL, если были
                encoded_symbols.append((15, 0))
                zero_run -= 16

            category = get_category(coeff)  # Категория/размер
            encoded_symbols.append((zero_run, category))  # Символ (run, size)
            encoded_values.append(coeff)  # Сохраняем значение отдельно
            zero_run = 0  # Сбрасываем счетчик нулей

    # Если остались нули в конце блока
    if zero_run > 0:
        encoded_symbols.append((0, 0))  # EOB символ

    return encoded_symbols, encoded_values


# === ЭТАП 7 и 9: Разностное кодирование DC и Энтропийное кодирование Хаффмана ===

class BitWriter:
    """Класс для побитовой записи данных."""

    def __init__(self):
        self.buffer = bytearray()
        self.accumulator = 0
        self.bit_count = 0  # Количество бит в аккумуляторе

    def write_bits(self, bits_string: str):
        """Записывает биты из строки '0' и '1'."""
        for bit_char in bits_string:
            bit = int(bit_char)
            # Помещаем бит на правильную позицию слева направо (MSB first)
            self.accumulator = (self.accumulator << 1) | bit
            self.bit_count += 1
            if self.bit_count == 8:
                self.buffer.append(self.accumulator)
                self.accumulator = 0
                self.bit_count = 0

    def write_value(self, value: int, category: int):
        """Записывает значение 'value' используя 'category' бит."""
        if category == 0:
            return  # Ничего не записываем для категории 0

        if value > 0:
            bits_string = format(value, f'0{category}b')
        else:
            # Отрицательные кодируются как (value + (1 << category) - 1)
            # Пример: cat=3, val=-3 -> -3 + (1<<3) - 1 = -3 + 8 - 1 = 4 -> '100'
            # Пример: cat=3, val=-1 -> -1 + (1<<3) - 1 = -1 + 8 - 1 = 6 -> '110'
            val_encoded = value + (1 << category) - 1
            bits_string = format(val_encoded, f'0{category}b')

        print(bits_string)
        self.write_bits(bits_string)

    def flush(self):
        """Записывает оставшиеся биты в последний байт (дополняя нулями справа)."""
        if self.bit_count > 0:
            # Сдвигаем оставшиеся биты влево, чтобы выровнять по MSB
            self.accumulator <<= (8 - self.bit_count)
            self.buffer.append(self.accumulator)
        # Сброс состояния для возможного повторного использования
        final_bytes = bytes(self.buffer)
        self.buffer = bytearray()
        self.accumulator = 0
        self.bit_count = 0
        return final_bytes


def encode_dc_stream(dc_values: list, huffman_table: dict, writer: BitWriter):
    """Кодирует поток DC-разностей с использованием таблицы Хаффмана."""
    if not dc_values:
        return

    # 1. Вычисляем разности
    diffs = [dc_values[0]]  # Первая разность - это сам первый DC
    for i in range(1, len(dc_values)):
        diff = dc_values[i] - dc_values[i - 1]
        diffs.append(diff)  # Не клиппуем здесь

    # 2. Кодируем каждую разность
    for diff in diffs:
        category = get_category(diff)
        # Записываем код Хаффмана для категории
        try:
            huffman_code = huffman_table[category]
            writer.write_bits(huffman_code)
        except KeyError:
            print(f"ОШИБКА: Нет кода Хаффмана для DC категории {category} (diff={diff})")
            # Можно добавить обработку ошибки, например, пропуск
            continue

        # Записываем биты самого значения разности
        writer.write_value(diff, category)


def encode_ac_stream(ac_rle_symbols: list, ac_rle_values: list, huffman_table: dict, writer: BitWriter):
    """Кодирует поток AC RLE символов и значений с использованием таблицы Хаффмана."""
    print(len(HUFFMAN_AC_LUMINANCE))
    value_idx = 0
    for symbol in ac_rle_symbols:
        # Записываем код Хаффмана для символа (run, size), ZRL или EOB
        try:
            huffman_code = huffman_table[symbol]
            writer.write_bits(huffman_code)
        except KeyError:
            print(f"ОШИБКА: Нет кода Хаффмана для AC символа {symbol}")
            # Если символ был (run, size), нам нужно пропустить соответствующее значение
            if symbol != (0, 0) and symbol != (15, 0):  # Пропускаем значение, если это не EOB/ZRL
                if value_idx < len(ac_rle_values): value_idx += 1
            continue  # Пропускаем запись значения для этого символа

        # Если символ был (run, size), где size > 0, записываем значение
        if symbol != (0, 0) and symbol != (15, 0):  # Не EOB и не ZRL
            run, size = symbol
            if size > 0:
                if value_idx < len(ac_rle_values):
                    value = ac_rle_values[value_idx]
                    writer.write_value(value, size)
                    value_idx += 1
                else:
                    # Этого не должно произойти, если encode_ac_coefficients работает правильно
                    print(f"ОШИБКА: Не хватает значений для AC символа {symbol}")


# === ЭТАП 10: Сборка файла и Запись метаданных ===

# Вспомогательная функция для основной обработки одного канала
def process_channel(channel_blocks: np.ndarray, quant_matrix: np.ndarray):
    """Обрабатывает все блоки одного канала: DCT, Q, Zigzag, RLE."""
    num_h_blocks, num_w_blocks, _, _ = channel_blocks.shape
    all_dc_values = []
    all_ac_rle_symbols = []
    all_ac_rle_values = []

    # Предвычисляем DCT матрицу один раз
    dct_matrix = DCT_MATRIX_8x8

    for i in range(num_h_blocks):
        for j in range(num_w_blocks):
            # Получаем блок (и смещаем его к нулю!)
            block = channel_blocks[i, j] - 128.0

            # DCT -> Квантование -> Зигзаг
            dct_block = dct_2d(block, dct_matrix)
            quant_block = quantize_block(dct_block, quant_matrix)
            zigzag_coeffs = zigzag_scan(quant_block)

            # Разделяем DC и AC
            dc_value = zigzag_coeffs[0]
            ac_coeffs = zigzag_coeffs[1:]

            # RLE кодирование AC
            rle_symbols, rle_values = encode_ac_coefficients(ac_coeffs)

            # Сохраняем результаты
            all_dc_values.append(dc_value)
            all_ac_rle_symbols.extend(rle_symbols)  # Просто добавляем символы в общий список
            all_ac_rle_values.extend(rle_values)  # И значения

    return all_dc_values, all_ac_rle_symbols, all_ac_rle_values


# Основная функция кодирования (заменяет pack_to_jpeg)
def encode_image(img: Image.Image, quality: int = 75):
    """
    Выполняет все шаги кодирования JPEG-подобного формата.
    Возвращает байтовую строку с результатом.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    original_width, original_height = img.size

    # 1. RGB -> YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(img)

    # 2. Даунсэмплинг Cb, Cr
    Cb_downsampled = downsample_channel(Cb)
    Cr_downsampled = downsample_channel(Cr)

    # 3. Разбиение на блоки
    Y_blocks, Y_padded_shape = split_into_blocks(Y)
    Cb_blocks, Cb_padded_shape = split_into_blocks(Cb_downsampled)
    Cr_blocks, Cr_padded_shape = split_into_blocks(Cr_downsampled)

    # 4. Масштабирование матриц квантования
    QY_scaled = scale_quantization_matrix(Q_Y, quality)
    QC_scaled = scale_quantization_matrix(Q_C, quality)

    # 5. Обработка каналов (DCT, Q, Zigzag, RLE)
    # (Важно: внутри process_channel происходит смещение на -128)
    y_dc, y_ac_sym, y_ac_val = process_channel(Y_blocks, QY_scaled)
    cb_dc, cb_ac_sym, cb_ac_val = process_channel(Cb_blocks, QC_scaled)
    cr_dc, cr_ac_sym, cr_ac_val = process_channel(Cr_blocks, QC_scaled)

    # 6. Энтропийное кодирование (Хаффман)
    writer = BitWriter()

    # --- Сюда нужно будет добавить запись метаданных ---
    # Например: размеры, таблицы квантования, таблицы Хаффмана
    # Пока пропустим для простоты структуры кода

    # Кодируем Y
    encode_dc_stream(y_dc, HUFFMAN_DC_LUMINANCE, writer)
    encode_ac_stream(y_ac_sym, y_ac_val, HUFFMAN_AC_LUMINANCE, writer)

    # Кодируем Cb (Используем таблицы цветности!)
    encode_dc_stream(cb_dc, HUFFMAN_DC_CHROMINANCE, writer)
    encode_ac_stream(cb_ac_sym, cb_ac_val, HUFFMAN_AC_CHROMINANCE, writer)

    # Кодируем Cr (Используем таблицы цветности!)
    encode_dc_stream(cr_dc, HUFFMAN_DC_CHROMINANCE, writer)
    encode_ac_stream(cr_ac_sym, cr_ac_val, HUFFMAN_AC_CHROMINANCE, writer)

    # Получаем финальный битовый поток
    encoded_data = writer.flush()

    header = struct.pack(">2sHHB", b"JP", original_width, original_height, quality)
    return header + encoded_data