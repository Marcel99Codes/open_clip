from PIL import Image
import numpy as np


image = Image.open("/home/marcel/Bilder/Hintergründe/test.JPEG")

image_rgb = image.convert("RGB")
image_ycbcr = image.convert("YCbCr")
image_lab = image.convert("LAB")


img_rgb = np.array(image_rgb)
img_ycbcr = np.array(image_ycbcr)
img_lab = np.array(image_lab)

print (img_rgb)
print (img_ycbcr)
print (img_lab)
