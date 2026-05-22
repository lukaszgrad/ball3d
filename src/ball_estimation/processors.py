import numpy as np

def rle_to_bitmask(rle: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Takes our binary RLE encoding as np.array, insertes data into
    a zeros vector and finally reshapes it to the target shape.
    """
    rle = rle.cumsum()
    output_array = np.zeros(height * width, dtype=bool)
    for l_f, l_t in zip(rle[::2], rle[1::2]):
        output_array[l_f:l_t] = True
    return output_array.reshape(height, width)
