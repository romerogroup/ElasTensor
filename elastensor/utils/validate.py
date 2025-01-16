import numpy as np

available_modes = ['strain-stress', 'strain-energy']

def get_valid_mode(mode):
    """Validate mode for generating deformations and calculating elastic constants.

    Parameters
    ----------
    mode : str_like
        The method to use for calculating elastic constants

    Returns
    -------
    valid_mode
        Validated mode
    """

    valid_mode = str(mode)
    if mode not in available_modes:
        raise ValueError(
            f"Unrecognized mode '{mode}'. Valid options are: {', '.join(available_modes)}."
        )

    return valid_mode

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

def get_valid_indices(values):
    """Validates the indices of the elastic constants

    Parameters
    ----------
    values : sequence of tuples
        Sequence of tuples containing indices for elastic constants.

    Returns
    -------
    list of tuples
        The list of validated elsatic indices

    Raises
    ------
    TypeError
        If values cannot be converted to tuples of integers.
    ValueError
        If any index is negative or if the length of any tuple is smaller than 2.
    NotImplementedError
        If indices correspond to fourth or higher order constants.
    """

    try:
        indices = [tuple(int(n) for n in idx) for idx in values]
    except (TypeError, ValueError) as err:
        raise type(err)("elastic index must be a list of tuples of integers.")
    if any(n < 0 for idx in indices for n in idx):
        raise ValueError("elastic index must be nonnegative.")

    order_list = [len(idx) for idx in indices]
    if any(order < 2 for order in order_list):
        raise ValueError("elastic index must be at least of second-order.")
    if any(order > 3 for order in order_list):
        raise NotImplementedError(
            "fourth-order elastic constants and beyond are not available."
        )

    return indices
