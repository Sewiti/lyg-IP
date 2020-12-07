from multiprocessing import Pool, cpu_count
from os import listdir, makedirs
from re import search
from time import time
import cv2

ALPHA_BIAS = 0.5
BLUR_STRENGTH = 69
PROCESSES = [12, 9, 6, 4, 3, 2, 1]
cv2.setNumThreads(1)  # For OpenCV


def main():
    data_sets = ['data/{:s}'.format(set) for set in listdir('data')]
    data_sets.sort()

    for set in data_sets:
        execute_set(set)


def execute_set(dir):
    """
    Manipulates a set of images in the given directory.

    Completes images manipulation multiple times with given PROCESSES counts, defined by a constant.

    Measures time that it took each time.

    Does a warmup before hand.
    """
    print('Executing \'{:s}\'...'.format(dir))

    imgs = ['{:s}/{:s}'.format(dir, f) for f in listdir(dir)]
    imgs.sort()

    pairs = [(imgs[i-1], imgs[i]) for i in range(len(imgs))]

    print('Warming up...')
    with Pool(cpu_count()) as p:
        p.map(manipulate, pairs)

    for i in PROCESSES:
        start = time()

        with Pool(i) as p:
            p.map(manipulate, pairs)

        print('Time taken ({:d}):{:s}{:>7.3f}s'.format(
            i, '' if i > 9 else ' ', time() - start))

    print()


def manipulate(pair) -> str:
    """
    Reads & manipulates a pair of given images.

    Resizes the first one to fill the second, crops it to match & applies GaussianBlur filter.
    The strength of the blur is defined by the BLUR_STRENGTH constant.

    Grayscales the second image and merges them together relative to the ALPHA_BIAS constant.
    Saves the result to output directory, keeping second image's name. 

    Returns the path of the saved image.
    """
    imgA = cv2.imread(pair[0])
    imgB = cv2.imread(pair[1])
    y, x = imgB.shape[:2]

    imgA = fill(imgA, x, y)
    imgA = crop(imgA, x, y)
    imgA = cv2.GaussianBlur(imgA, (BLUR_STRENGTH, BLUR_STRENGTH), 0)

    # Applies Grayscale
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # Need to convert back to match type when merging (Grayscale is kept)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)

    # Merges the images, according to ALPHA_BIAS
    img = cv2.addWeighted(imgA, 1.0 - ALPHA_BIAS, imgB, ALPHA_BIAS, 0)

    output = pair[1].replace('data/', 'output/')
    save(img, output)

    return output


def save(img, filepath):
    dir = search(r'(.+)/.+', filepath).group(1)
    makedirs(dir, exist_ok=True)

    cv2.imwrite(filepath, img)


def fill(img, x, y):
    _y, _x = img.shape[:2]

    _y = _y/_x*x
    _x = x

    if _y < y:
        _x = _x/_y*y
        _y = y

    _x, _y = round(_x), round(_y)
    img = cv2.resize(img, (_x, _y), interpolation=cv2.INTER_CUBIC)

    return img


def crop(img, x, y):
    _y, _x = img.shape[:2]

    dx = round((_x - x)/2)
    dy = round((_y - y)/2)

    img = img[dy:_y-dy, dx:_x-dx]
    return cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)


if __name__ == "__main__":
    main()
