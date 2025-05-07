import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from encoder import encode_image
from decoder import decode_image


# --- Конфигурация ---
TEST_IMAGES_DIR = "Test Images"
OUTPUT_DIR_TASK2 = "Tests" # Папка для сохранения изображений из пункта 2
PLOTS_DIR = "Plots"      # Папка для сохранения графиков из пункта 1

# Список тестовых изображений
IMAGE_FILENAMES = [
    "Beach.jpg", "Beach_BW.jpg", "Beach_D.png", "Beach_GS.jpg",
    "Lenna.png", "Lenna_BW.png", "Lenna_D.png", "Lenna_GS.png"
]

# Качества для пункта 1 (графики)
QUALITY_LEVELS_TASK1 = list(range(0, 101, 5)) # Шаг 5, от 0 до 100 включительно

# Качества для пункта 2 (сжатие/декомпрессия)
QUALITY_LEVELS_TASK2 = [0, 20, 40, 60, 80, 100]

# --- Вспомогательные функции ---

def get_image_path(filename):
    return os.path.join(TEST_IMAGES_DIR, filename)

def get_output_path_task2(original_filename, quality):
    base, ext = os.path.splitext(original_filename)
    # Добавляем суффикс с качеством, сохраняем в PNG для универсальности и чтобы избежать потерь от JPEG
    new_filename = f"{base}_quality{quality}.png"
    return os.path.join(OUTPUT_DIR_TASK2, new_filename)

def get_plot_path(original_filename_base):
    plot_filename = f"{original_filename_base}_compression_vs_quality.png"
    return os.path.join(PLOTS_DIR, plot_filename)

# --- Основная логика ---

def main():
    # Создаем папки для вывода, если их нет
    if not os.path.exists(OUTPUT_DIR_TASK2):
        os.makedirs(OUTPUT_DIR_TASK2)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # --- ЗАДАНИЕ 1: Графики зависимости размера от качества ---
    print("--- Задание 1: Построение графиков зависимости размера от качества ---")
    for image_filename in IMAGE_FILENAMES:
        image_path = get_image_path(image_filename)
        print(f"\nОбработка изображения для графика: {image_filename}")

        if not os.path.exists(image_path):
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Файл {image_path} не найден. Пропускаю.")
            continue

        try:
            img_pil = Image.open(image_path)
            # Убедимся, что изображение в RGB, если ваши функции этого требуют
            if img_pil.mode != 'RGB':
                img_pil_rgb = img_pil.convert('RGB')
            else:
                img_pil_rgb = img_pil
        except Exception as e:
            print(f"  ОШИБКА при открытии {image_path}: {e}. Пропускаю.")
            continue

        compressed_sizes = []
        valid_qualities_for_plot = []

        for quality in QUALITY_LEVELS_TASK1:
            try:
                print(f"  Сжатие с качеством: {quality}...")
                encoded_byte_stream = encode_image(img_pil_rgb, quality=quality)
                compressed_sizes.append(len(encoded_byte_stream))
                valid_qualities_for_plot.append(quality)
            except Exception as e:
                print(f"    ОШИБКА при кодировании {image_filename} с качеством {quality}: {e}")
                # Можно добавить None или пропустить точку, если кодирование не удалось
                # compressed_sizes.append(None) # Если хотим разрыв на графике
                continue  # Пропускаем эту точку качества

            if not compressed_sizes:
                print(f"  Нет данных для построения графика для {image_filename}.")
                continue

                # Построение графика
            plt.figure(figsize=(10, 6))
            plt.plot(valid_qualities_for_plot, compressed_sizes, marker='o', linestyle='-')
            plt.title(f"Зависимость размера сжатого файла от качества\n{image_filename}")
            plt.xlabel("Коэффициент качества сжатия")
            plt.ylabel("Размер сжатого файла (байты)")
            plt.xticks(np.arange(min(valid_qualities_for_plot), max(valid_qualities_for_plot) + 1,
                                 10))  # Метки на оси X каждые 10
            plt.grid(True)

            # Сохранение графика
            base_filename, _ = os.path.splitext(image_filename)
            plot_save_path = get_plot_path(base_filename)
            try:
                plt.savefig(plot_save_path)
                print(f"  График сохранен в: {plot_save_path}")
            except Exception as e:
                print(f"  ОШИБКА при сохранении графика {plot_save_path}: {e}")
            plt.close()  # Закрываем фигуру, чтобы не накапливались в памяти

            # --- ЗАДАНИЕ 2: Сжатие, декомпрессия и сохранение изображений ---
    print("\n\n--- Задание 2: Сжатие, декомпрессия и сохранение изображений ---")
    for image_filename in IMAGE_FILENAMES:
        image_path = get_image_path(image_filename)
        print(f"\nОбработка изображения для сжатия/декомпрессии: {image_filename}")

        if not os.path.exists(image_path):
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Файл {image_path} не найден. Пропускаю.")
            continue

        try:
            img_pil_original = Image.open(image_path)
            if img_pil_original.mode != 'RGB':
                img_pil_rgb_original = img_pil_original.convert('RGB')
            else:
                img_pil_rgb_original = img_pil_original
        except Exception as e:
            print(f"  ОШИБКА при открытии {image_path}: {e}. Пропускаю.")
            continue

        for quality in QUALITY_LEVELS_TASK2:
            print(f"  Качество: {quality}")
            try:
                # Сжатие
                print(f"    Кодирование...")
                encoded_byte_stream = encode_image(img_pil_rgb_original, quality=quality)

                # Декомпрессия
                print(f"    Декодирование...")
                decoded_img_pil = decode_image(encoded_byte_stream)

                # Сохранение результата
                output_path = get_output_path_task2(image_filename, quality)
                decoded_img_pil.save(output_path)
                print(f"    Результат сохранен в: {output_path} (Размер сжатого: {len(encoded_byte_stream)} байт)")

            except Exception as e:
                print(f"    ОШИБКА при обработке {image_filename} с качеством {quality}: {e}")

    print("\n--- Все задания выполнены ---")

if __name__ == "__main__":
    main()

