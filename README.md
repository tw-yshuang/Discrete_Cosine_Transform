# Discrete_Cosine_Transform

> Data dir: `./images/` \
> Result dir: `./out/`

## Requirement

You can install requirement by follow this command:

```sh
pip3 install PIL numpy numba
```

If you have `pipenv` to manage your python project environment, then you can follow this command:

```sh
pipenv install # install venv by following Pipfile.lock
pipenv shell # active your project venv.
```

## Function Introduce

| Function name  |                   Parameter                    | Introdution   |
| -------------- | :--------------------------------------------: | ------------- |
| dct_1d()       |                 seq: np.array                  | 1D-DCT        |
| idct_1d()      |                 seq: np.array                  | 1D-IDCT       |
| dct_fast_2d()  |                 img: np.array                  | Fast 2D-IDCT  |
| idct_fast_2d() |                 img: np.array                  | Fast 2D-IDCT  |
| dct_block()    | img: np.array, block: (w, h) default is (8, 8) | 2D-DCT_block  |
| idct_block()   | img: np.array, block: (w, h) default is (8, 8) | 2D-IDCT_block |
