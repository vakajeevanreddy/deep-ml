import numpy as np

def online_simple_regression(xs, ys):
    # return (N, 2) array of (w0, w1) after each point
    xs_array = np.asarray(xs, dtype = np.float64)
    ys_array = np.asarray(ys, dtype = np.float64)

    if xs_array.ndim != 1 or ys_array.ndim != 1 or xs_array.shape[0] != ys_array.shape[0]:
        raise ValueError("xs and ys must be 1D sequences of equal length.")
    n_total = xs_array.shape[0]
    params = np.empty((n_total , 2), dtype = np.float64)

    count = 0
    mean_x = 0.0
    mean_y = 0.0
    m2_x = 0.0
    m_xy = 0.0

    for x,y in zip(xs_array, ys_array):
        count += 1
        dx = x - mean_x
        mean_x += dx / count
        dx_new = x - mean_x

        dy = y - mean_y
        mean_y += dy/count

        m2_x += dx * dx_new
        m_xy += dx * (y - mean_y)

        if m2_x == 0.0:
            w1 = 0.0
            w0 = mean_y
        else:
            w1 = m_xy / m2_x
            w0 = mean_y - w1 * mean_x
        params[count - 1 , 0] = w0
        params[count - 1 , 1] = w1
    return params