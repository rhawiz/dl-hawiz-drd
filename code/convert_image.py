import os
import click
import cv2
from scipy import misc
import numpy as np


def find_border_threshold(img, axis):
    """
    Credits to ktugan @ https://github.com/ktugan/diabetic_retina
    Sums up the axis to one dimension only. The resulting 1 dimensional array has the property that
    a higher number represents an overall brighter picture on the axis. This allows to find the border which is very dark
    (black). The border is found by taking the maximum value of this summed up array and the border starts where the
    image is a 1/10th of this value.
    This relative approach is necessary because some of the images are very bright, others are very dark. So the border
    and the pixel where they start differ a lot.
    """
    img_array = np.sum(np.sum(img, axis=axis), axis=1)
    threshold = img_array.max() / 10
    indices = np.where(img_array > threshold)
    return indices[0][0], indices[0][-1]


def crop_border_threshold(img):
    """
    Credits to ktugan @ https://github.com/ktugan/diabetic_retina
    Finds the indices for the black border and then crops them from there.
    After the operation it may be necessary to center the picture by adding a black border.
    Yes the irony is not lost on me...
    """
    # finding min max indices
    min_x, max_x = find_border_threshold(img, 0)
    min_y, max_y = find_border_threshold(img, 1)

    # crop
    img = img[min_y:max_y, min_x:max_x]

    # calculate the indices and amount of padding so the image is a square
    s1, s2, _ = img.shape

    m = max(s1, s2)
    b = np.zeros((m, m, 3), dtype=img.dtype)
    x, y = m - s1, m - s2
    x1, x2 = x / 2, m - (x - x / 2)
    y1, y2 = y / 2, m - (y - y / 2)

    # centering the image
    b[x1:x2, y1:y2, :] = img
    return b


def scale_radius(img, scale):
    x = img[img.shape[0] / 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def convert_image(path, size=256, colour_norm=True, crop_border=False):
    img = cv2.imread(path)

    if crop_border:
        img = crop_border_threshold(img)
    # img = misc.imresize(img, (size, size))
    img = scale_radius(img, size)

    if colour_norm:
        # Subtract local mean colour
        img = cv2.addWeighted(
            img,
            4,
            cv2.GaussianBlur(img, (0, 0), size / 30),
            -4,
            128

        )
    if crop_border:
        # remove outer 10%
        b = np.zeros(img.shape)
        cv2.circle(b, (img.shape[1] / 2, img.shape[0] / 2), int(size * 0.9), (1, 1, 1), -1, 8, 0)
        img = img * b + 128 * (1 - b)
    img = misc.imresize(img, (size, size))
    return img


def convert(in_dir, out_dir, size=256, colour_norm=True,
            debug=False, overwrite=False, crop_border=False):
    # Get absolute paths
    in_dir = os.path.relpath(in_dir)
    out_dir = os.path.relpath(out_dir)
    # Create output directory if it doesn't exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Iterate through files in folder
    counter = 0
    print 'Conversion starting.'
    for subdir, dirs, files in os.walk(in_dir):

        for file in files:
            counter += 1
            new_path = os.path.join(out_dir, file)
            if not overwrite:
                if os.path.isfile(new_path):
                    if debug:
                        print '{}/{}\t{} already exists, skipping conversion.\n\tUse -ow or --overwrite flag to overwrite files.'.format(
                            counter, len(files), file)
                    continue
            if debug or (counter % 100 == 0):
                print '{}/{}\t{}'.format(counter, len(files), file)
            path = os.path.join(subdir, file)
            converted_img = convert_image(path=path, size=size, colour_norm=colour_norm, crop_border=crop_border)
            cv2.imwrite(new_path, converted_img)
    print 'Conversion starting.'


@click.command()
@click.option('--in_dir', '-i', show_default=True, help="Input directory with original images.")
@click.option('--out_dir', '-o', show_default=True, help="Output directory to saved converted images.")
@click.option('--debug', '-d', is_flag=True, default=False, show_default=True,
              help="Show debug information on console.")
@click.option('--size', '-s', default=256, show_default=True, help="Size of newly converted images.")
@click.option('--colour_norm', '-cn', is_flag=True, default=False, show_default=True, help="Subtract local mean colour")
@click.option('--overwrite', '-ow', is_flag=True, default=False, show_default=True, help="Overwrite files")
@click.option('--crop_border', '-cb', is_flag=True, default=False, show_default=True, help="Crop border")
def main(in_dir, out_dir, size=256, debug=False, colour_norm=True, overwrite=False, crop_border=False):
    convert(in_dir=in_dir, out_dir=out_dir, size=size, colour_norm=colour_norm, debug=debug, overwrite=overwrite,
            crop_border=crop_border)


if __name__ == '__main__':
    main()
