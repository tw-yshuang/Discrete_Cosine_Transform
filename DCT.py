import math
from PIL import Image
import numpy as np
from numba import jit


@jit(nopython=True)
def dct_2d(img: np.ndarray, dct_img: np.ndarray):
    """
    ## Original 2D-DCT algorithm
    input
    `img`: a 2D array content time domain values.
    `dct_img`: an empty array that shape is same as img, require for numba @jit accelerate.

    output
    `dct_img`: a 2D array content frequency domain values.
    """

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[0]):
                for l in range(img.shape[1]):
                    if i == 0 and j == 0:
                        # print('here:(0, 0)')
                        dct_img[i, j] += 1 / ((img.shape[0] + img.shape[1]) / 2) * img[k][l]
                    else:
                        dct_img[i, j] += (
                            ((1 / img.shape[0]) + (1 / img.shape[1]))
                            * img[k, l]
                            * ((1 / math.sqrt(2)) if i == 0 else (math.cos((2 * k + 1) * i * math.pi / (2 * img.shape[0]))))
                            * ((1 / math.sqrt(2)) if j == 0 else (math.cos((2 * l + 1) * j * math.pi / (2 * img.shape[1]))))
                        )

    return dct_img


@jit(nopython=True)
def idct_2d(dct_img, idct_img):
    """
    ## Original 2D-IDCT algorithm
    input
    `dct_img`: a 2D array content frequency domain values.
    `idct_img`: an empty array that shape is same as dct_img, require for numba @jit accelerate.

    output
    `idct_img`: a 2D array content time domain values.
    """

    for i in range(dct_img.shape[0]):
        for j in range(dct_img.shape[1]):
            for k in range(dct_img.shape[0]):
                for l in range(dct_img.shape[1]):
                    idct_img[i, j] += (
                        ((1 / dct_img.shape[0]) + (1 / dct_img.shape[1]))
                        * dct_img[k, l]
                        * ((1 / math.sqrt(2)) if k == 0 else math.cos((2 * i + 1) * k * math.pi / (2 * dct_img.shape[0])))
                        * ((1 / math.sqrt(2)) if l == 0 else math.cos((2 * j + 1) * l * math.pi / (2 * dct_img.shape[1])))
                    )

    return idct_img


@jit(nopython=True)
def dct_1d(seq: np.ndarray):
    """
    ## 1D-IDCT algorithm
    input
    `seq`: a 1D array content time domain values.

    output
    `dct_seq`: a 1D array content frequency domain values.
    """
    dct_seq = np.zeros(seq.shape, dtype=np.float64)  # create a 1D array for saving frequency domain values
    for u_idx in range(len(seq)):  # find index of dct_seq
        value = 0.0
        for m_idx, m in enumerate(seq):
            value += m * math.cos((2 * m_idx + 1) * u_idx * math.pi / (2 * len(seq)))

        dct_seq[u_idx] = value * math.sqrt(2 / len(seq)) if u_idx > 0 else value / math.sqrt(len(seq))
    return dct_seq


@jit(nopython=True)
def idct_1d(seq: np.ndarray):
    """
    ## 1D-IDCT algorithm
    input
    `seq`: a 1D array content frequency domain values.

    output
    `idct_seq`: a 1D array content time domain values.
    """
    idct_seq = np.zeros(seq.shape, dtype=np.float64)
    for m_idx in range(len(seq)):
        for u_idx, u in enumerate(seq):
            value = math.sqrt(2 / len(seq)) if u_idx > 0 else 1 / math.sqrt(len(seq))
            value *= u * math.cos((2 * m_idx + 1) * u_idx * math.pi / (2 * len(seq)))

            idct_seq[m_idx] += value
    return idct_seq


def dct_fast_2d(img: np.ndarray):
    """
    ## Fast 2D-DCT algorithm
    input
    `img`: a 2D array content time domain values, usually is an image.

    output
    `dct_img`: a 2D array content frequency domain values.
    """
    dct_img = img.astype(np.float64, copy=False)
    dims = 1 if len(img.shape) == 2 else img.shape[2]  # find the dims of image

    for dim in range(dims):
        dct_dim = dct_img if dims == 1 else dct_img[:, :, dim]  # iterate by dim's sequnce
        for w in range(img.shape[0]):  # do 1D-DCT form row side first
            dct_dim[w, :] = dct_1d(dct_dim[w, :])
        for h in range(img.shape[1]):  # do 1D-DCT form column side
            dct_dim[:, h] = dct_1d(dct_dim[:, h])

    return dct_img


def idct_fast_2d(dct_img: np.ndarray):
    """
    ## Fast 2D-IDCT algorithm
    input
    `dct_img`: a 2D array content frequency domain values.

    output
    `idct_img`: a 2D array content time domain values.
    """
    idct_img = dct_img.astype(np.float64, copy=False)
    dims = 1 if len(dct_img.shape) == 2 else dct_img.shape[2]  # find the dims of image

    for dim in range(dims):
        idct_dim = idct_img if dims == 1 else idct_img[:, :, dim]  # iterate by dim's sequnce
        for w in range(dct_img.shape[0]):  # do 1D-IDCT form row side first
            idct_dim[w, :] = idct_1d(idct_dim[w, :])
        for h in range(dct_img.shape[1]):  # do 1D-IDCT form column side
            idct_dim[:, h] = idct_1d(idct_dim[:, h])

    return idct_img


def dct_block(img: np.ndarray, block: tuple = (8, 8)):
    """
    ## Fast 2D-DCT by block
    input
    `img`: a 2D array content time domain values, usually is an image.
    `block`: (row, col)

    output
    `dct_img`: a 2D array content frequency domain values.
    """
    w_stripe = block[0]  # the stripe of row side
    h_stripe = block[1]  # the stripe of column side

    num_w = img.shape[0] // w_stripe  # number of block in row side
    num_h = img.shape[1] // w_stripe  # number of block in column side

    dct_img = img.astype(np.float64, copy=False)
    dims = 1 if len(img.shape) == 2 else img.shape[2]  # find the dims of image
    for dim in range(dims):
        dct_dim = dct_img if dims == 1 else dct_img[:, :, dim]  # iterate by dim's sequnce
        for w in range(num_w):
            for h in range(num_h):
                # process Fast 2D-DCT algorithm by block, form left to right, top to bottom.
                dct_dim[w * w_stripe : (w + 1) * w_stripe, h * h_stripe : (h + 1) * h_stripe] = dct_fast_2d(
                    dct_dim[w * w_stripe : (w + 1) * w_stripe, h * h_stripe : (h + 1) * h_stripe]
                )

    return dct_img


def idct_block(dct_img: np.ndarray, block: tuple = (8, 8)):
    """
    ## Fast 2D-IDCT by block
    input
    `dct_img`: a 2D array content frequency domain values
    `block`: (row, col)

    output
    `idct_img`: a 2D array content time domain values
    """
    w_stripe = block[0]  # the stripe of row side
    h_stripe = block[1]  # the stripe of column side

    num_w = dct_img.shape[0] // w_stripe  # number of block in row side
    num_h = dct_img.shape[1] // w_stripe  # number of block in column side

    idct_img = dct_img.astype(np.float64, copy=False)
    dims = 1 if len(dct_img.shape) == 2 else dct_img.shape[2]  # find the dims of image
    for dim in range(dims):
        idct_dim = idct_img if dims == 1 else idct_img[:, :, dim]  # iterate by dim's sequnce
        for w in range(num_w):
            for h in range(num_h):
                # process Fast 2D-IDCT algorithm by block, form left to right, top to bottom.
                idct_dim[w * w_stripe : (w + 1) * w_stripe, h * h_stripe : (h + 1) * h_stripe] = idct_fast_2d(
                    idct_dim[w * w_stripe : (w + 1) * w_stripe, h * h_stripe : (h + 1) * h_stripe]
                )

    return idct_img


if __name__ == '__main__':

    paths = ['./images/lena.tif', './images/lena_std.tif', './images/puppy.jpg']
    for i, path in enumerate(paths):
        filename = path.split('/')[2].split('.')[0]  # find the filename, e.g. lena, child, etc.
        transformed_img = np.array(Image.open(path))
        # =====================================================#
        # process by basic DCT_2D full image
        dct_img = np.array(transformed_img, dtype=np.float64)
        dims = 1 if len(dct_img.shape) == 2 else dct_img.shape[2]
        for dim in range(dims):
            dct_dim = dct_img if dims == 1 else dct_img[:, :, dim]  # iterate by dim's sequnce
            dct_dim[:, :] = dct_2d(dct_dim, np.zeros_like(dct_dim))

        # print(dct_img)
        Image.fromarray(dct_img.copy().astype(np.uint8)).save(f'./out/original_dct_{filename}.png')

        idct_img = dct_img.copy()
        for dim in range(dims):
            idct_dim = idct_img if dims == 1 else idct_img[:, :, dim]  # iterate by dim's sequnce
            idct_dim[:, :] = idct_2d(idct_dim, np.zeros_like(idct_dim))

        # print(idct_img)
        Image.fromarray(idct_img.astype(np.uint8)).save(f'./out/original_idct_{filename}.png')

        # =====================================================#
        # process by fast DCT_2D full image
        dct_img = dct_fast_2d(transformed_img)
        # print(dct_img)
        Image.fromarray(dct_img.copy().astype(np.uint8)).save(f'./out/dct_{filename}.png')

        idct_img = idct_fast_2d(dct_img)
        # print(idct_img)
        Image.fromarray(idct_img.astype(np.uint8)).save(f'./out/idct_{filename}.png')

        # ======================================================#
        # process by block of image
        block = (8, 8)
        dct_img = dct_block(transformed_img, block=block)
        # print(dct_img)

        Image.fromarray(dct_img.copy().astype(np.uint8)).save(f'./out/dct-{block[0]}x{block[1]}_{filename}.png')

        idct_img = idct_block(dct_img, block=block)
        # print(idct_img)
        Image.fromarray(idct_img.copy().astype(np.uint8)).save(f'./out/idct-{block[0]}x{block[1]}_{filename}.png')
