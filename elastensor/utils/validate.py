import numpy as np


def get_valid_array(array, attribute, dtype=float, shape=None):
    """Convert input to numpy array with validation.

    Parameters
    ----------
    array : array_like
        Input array to validate and convert.
    attribute : str
        Name of the array attribute for error messages.
    dtype : data-type, optional
        The desired data-type for the array. Default is float.
    shape : tuple of int, optional
        Required shape of the array. If None, any shape is allowed.

    Returns
    -------
    ndarray
        The validated numpy array with specified dtype and shape.
    """

    try:
        the_array = np.array(array, dtype=dtype)
    except (IndexError, ValueError, TypeError) as err:
        typename = dtype.__name__
        raise type(err)(
            f"'{attribute}' must be an arraylike object of '{typename}' elements."
        )

    if shape is not None:
        try:
            required_shape = tuple(int(i) for i in shape)
        except (ValueError, TypeError) as err:
            raise type(err)(f"Argument 'shape' must be a tuple of integers.")
        else:
            if the_array.shape != required_shape:
                raise ValueError(f"'{attribute}' must be of shape '{required_shape}'")

    return the_array
