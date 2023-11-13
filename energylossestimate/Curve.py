import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma
import matplotlib.pyplot as plt


class Curve:
    """The energy distribution of a particle shower. """

    def __init__(self, x, y, E):
        self.t = x
        self.dEdt = y
        self.energy = E

    @classmethod
    def fit(cls, t, dEdt, energy, bin_size):
        """If fit is possible, returns Curve object. Otherwise, returns None."""

        # Attempt to fit the gamma distribution to the data
        t_max = max(t)
        b_est = 0.5
        a_est = t_max * b_est + 1
        
        try:
            poptGamma, pcov = curve_fit(cls.gammaFit, t, dEdt, p0=[a_est, b_est])

            # Generate curve data
            x_line = np.arange(min(t), max(t), bin_size)
            y_line_gamma = cls.gammaFit(x_line, *poptGamma)
            return cls(x_line, y_line_gamma, energy)

        except RuntimeError as e:
            # Handle any fitting errors (e.g., optimal parameters not found)
            print(f"An error occurred during curve fitting: {e}")

        return None

    @staticmethod
    def gammaFit(t, a, b):
        """Gamma PDF function for curve fitting.
        
        Parameters:
        - t: The independent variable (time or similar).
        - a: The shape parameter of the gamma distribution.
        - b: The rate parameter of the gamma distribution.

        Returns:
        - The gamma PDF evaluated at `t`.
        """
        return gamma.pdf(t, a, scale=1/b)