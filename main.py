from shitDecoder import *
from codes import *
from PIL import Image
from encoder import encode_image

if __name__ == '__main__':
    # Rebuild inverse tables after importing
    INV_HUFFMAN_DC_L = build_inverse_huffman_table(HUFFMAN_DC_LUMINANCE)
    INV_HUFFMAN_DC_C = build_inverse_huffman_table(HUFFMAN_DC_CHROMINANCE)
    INV_HUFFMAN_AC_L = build_inverse_huffman_table(HUFFMAN_AC_LUMINANCE)
    INV_HUFFMAN_AC_C = build_inverse_huffman_table(HUFFMAN_AC_CHROMINANCE)
    print("Inverse Huffman tables rebuilt.")

    img = Image.open("Lenna.png") # Or load a real image

    print(f"Original image size: {img.size}")

    # Encode the image
    print("Encoding image...")
    compressed_data = encode_image(img, quality=75)
    print(f"Encoded data size: {len(compressed_data)} bytes")

    print("--- DC Luminance Table ---")
    print(HUFFMAN_DC_LUMINANCE)
    print("--- Inverse DC Luminance Table ---")
    print(INV_HUFFMAN_DC_L)
    print("--- First few bytes of compressed data after header ---")
    header_size = struct.calcsize(">2sHHB")
    print(compressed_data[
          header_size:header_size + 20].hex())  # Выводим первые 20 байт данных в шестнадцатеричном формате

    # Save the compressed data (optional, for testing file read)
    # with open("compressed.bin", "wb") as f:
    #     f.write(compressed_data)

    # Decode the image
    print("Decoding image...")
    img_decoded = decode_image(compressed_data)
    print(f"Decoded image size: {img_decoded.size}")


    # Save the decoded image
    img_decoded.save("decoded_image.png")
    print("Decoded image saved as decoded_image.png")

    # Optional: Compare original and decoded image (e.g., using PIL's ImageChops if available)
    from PIL import ImageChops
    diff = ImageChops.difference(img, img_decoded)
    diff.save("difference.png")
    print("Difference image saved as difference.png")
    print(f"Max pixel difference: {np.max(np.asarray(diff))}")
