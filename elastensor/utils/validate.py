import numpy as np

def get_valid_array(array, attribute, dtype=float, shape=None):

    try:
        the_array = np.array(array, dtype=dtype)
    except (IndexError, ValueError, TypeError) as err:
        typename = dtype.__name__
        raise type(err)(f"'{attribute}' must be an arraylike object of '{typename}' elements.")

    if shape is not None:
        try:
            required_shape = tuple(int(i) for i in shape)
        except (ValueError, TypeError) as err:
            raise type(err)(f"Argument 'shape' must be a tuple of integers.")
        else:
            if the_array.shape != required_shape:
                raise ValueError(f"'{attribute}' must be of shape '{required_shape}'")

    return the_array
