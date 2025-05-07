from PIL import Image
import os

from PIL.Image import Dither


def convert_image_variations(image_path, output_path_prefix, threshold=128):
    try:
        img = Image.open(image_path)

        # 1. Изображение в оттенках серого (Grayscale)
        img_grayscale = img.convert('L')
        grayscale_output_path = f"{output_path_prefix}_GS.png"
        img_grayscale.save(grayscale_output_path)

        # 2. Черно-белое (двоичное) изображение
        img_thresholded = img_grayscale.point(lambda x: 255 if x > threshold else 0)
        img_bw_final = img_thresholded.convert('1')
        bw_output_path = f"{output_path_prefix}_BW.png"
        img_bw_final.save(bw_output_path)

        # 3. Изображение с дизерингом (черно-белое)
        img_dithered = img_grayscale.convert('1', dither=Dither.ORDERED)
        dithered_output_path = f"{output_path_prefix}_D.png"
        img_dithered.save(dithered_output_path)

        print(f"Преобразования для {image_path} завершены. Результаты в папке '{os.path.dirname(output_path_prefix)}'.")

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {image_path}")
    except Exception as e:
        print(f"Произошла ошибка при обработке {image_path}: {e}")


if __name__ == "__main__":
    test_images_data = [
        {"path": "Test Images/Lenna.png", "prefix": "output/Lenna", "threshold": 128},
        {"path": "Test Images/Beach.jpg", "prefix": "output/Beach", "threshold": 100}
    ]

    for item in test_images_data:
        output_dir = os.path.dirname(item["prefix"])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\nОбработка: {item['path']}")
        convert_image_variations(item["path"], item["prefix"], threshold=item["threshold"])
