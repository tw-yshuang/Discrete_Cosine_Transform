# Discrete_Cosine_Transform

> Data dir: `./images/` \
> Result dir: `./out/`

## Function Introduce

| Function name  |                   Parameter                    | Introdution   |
| -------------- | :--------------------------------------------: | ------------- |
| dct_1d()       |                 seq: np.array                  | 1D-DCT        |
| idct_1d()      |                 seq: np.array                  | 1D-IDCT       |
| dct_fast_2d()  |                 img: np.array                  | Fast 2D-IDCT  |
| idct_fast_2d() |                 img: np.array                  | Fast 2D-IDCT  |
| dct_block()    | img: np.array, block: (w, h) default is (8, 8) | 2D-DCT_block  |
| idct_block()   | img: np.array, block: (w, h) default is (8, 8) | 2D-IDCT_block |
