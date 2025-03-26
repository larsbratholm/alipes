"""
Script to do linear fit with l1-norm of an input npz file.
"""

import numpy as np
from numpy.typing import NDArray


def l1_objective(
    w: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """
    Compute l1 loss, and the gradient of the loss wrt. the weights.

    Args:
        w: (m, ) the weights
        x: (n, m) the features
        y: (n, ) the target

    Returns:
        l1-loss and gradient
    """
    y_estimate = x.dot(w)
    error = y_estimate - y
    gradient = (np.sign(error)[:, None] * x).mean(0)

    return np.abs(error).mean(), gradient


def fit(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_iterations: int = 1000,
    learning_rate: float = 0.1,
    delta: float = 1e-9,
) -> NDArray[np.float64]:
    """
    Fit X @ w = y with an l1-norm.

    Since there is analytical solution with l1-norm, and numpy does not support
    numerical optimization, I solve this numerically with gradient descent.
    Since the learning rate is fixed, and the convergence criteria is simple,
    the last few decimal are approximate.

    Args:
        x: (n, m) the features
        y: (n, ) the target
        max_iterations: the maximum number of gradient descent iterations.
        learning_rate: the learning rate of the gradient descent algorithm
        delta: the convergence criteria

    Returns:
        (m, ) weights
    """
    w = np.ones(x.shape[1])

    learning_rate = 0.1
    for iteration in range(max_iterations):
        loss, gradient = l1_objective(w=w, x=x, y=y)
        delta_w = learning_rate * gradient
        w -= delta_w
        if np.abs(delta_w).max() < delta:
            break
    return w


def main(filename: str) -> None:
    """
    Do linear fit with l1-norm on data and print fitted parameters.

    Args:
        filename: the data filename
    """
    d = np.load(filename)
    x, y = d.values()
    w = fit(x=x, y=y)

    np.set_printoptions(precision=4, suppress=True)
    print(f"The optimal weights are {w}")


if __name__ == "__main__":
    # I would use argparse wrapped with pydantic
    # in production, instead of hardcoding
    filename = "./data/xy.npz"
    main(filename=filename)
