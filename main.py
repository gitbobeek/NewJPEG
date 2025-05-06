from encoder import *
from decoder import *

img_pil = Image.open("Lenna.png")
encoded_byte_stream = encode_image(img_pil, quality=20)

with open("compressed.bin", "wb") as f:
    f.write(encoded_byte_stream)

decoded_img_pil = decode_image_data(encoded_byte_stream)

decoded_img_pil.show()