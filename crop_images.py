import os
from PIL import Image
import math


def crop_all_images_in_directory(origin_dir, scale, crop_size, save_location):
    """
    :param origin_dir
    :param scale
    :param crop_size
    :param save_location
    :return:
    """
    for filename in os.listdir(origin_dir):
        img_path = os.path.join(origin_dir, filename)
        crop_img_high_resolution(img_path, scale, crop_size, save_location)


def crop_img_high_resolution(img_path, scale, crop_size, save_location):
    """
    :param img_path
    :param scale
    :param crop_size
    :param save_location
    :return:
    """
    img = Image.open(img_path)
    origin_width, origin_height = img.size

    new_width = math.ceil(origin_width * scale)
    new_height = math.ceil(origin_height * scale)

    resized_img = img.resize(size=(new_width, new_height), resample=Image.LANCZOS)

    width, height = resized_img.size
    num_slices_width = math.ceil(width / crop_size)
    num_slices_height = math.ceil(height / crop_size)

    for i in range(num_slices_width):
        for j in range(num_slices_height):
            left = i * crop_size
            top = j * crop_size
            right = min(left + crop_size, width)
            bottom = min(top + crop_size, height)

            crop_img = resized_img.crop(box=(left, top, right, bottom))
            crop_img.save(f"{save_location}/{os.path.basename(img_path).replace('.', '_crop_')}_{i}_{j}.bmp", dpi=(300*scale, 300*scale))


if __name__ == '__main__':
    crop_all_images_in_directory(
        origin_dir="./data/origin",
        scale=2,
        crop_size=224,
        save_location="./data/after"
    )
