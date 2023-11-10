import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma
import matplotlib.pyplot as plt


class Curve:
    """The energy distribution of a particle shower. """

    def __init__(self, x, y, E, r2):
        self.t = x
        self.dEdt = y
        self.energy = E
        self.r_squared = r2

    @classmethod
    def fit(cls, t, dEdt, energy, bin_size, ignore=False, r2_threshold=0.5):
        """If fit is possible, returns Curve object. Otherwise, returns None."""
        
        # if energy > 1000000:
        #     # View high energy data!
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(t, dEdt)
        #     plt.show()

        if len(dEdt) < 20:  # minimum fit data required
            # print("not enough points")
            return None

        # TODO: better guess for shape and rate (expect: right skew, more squish / less squish depending on peak dEdt)

        # Attempt to fit the gamma distribution to the data
        try:
            poptGamma, pcov = curve_fit(cls.gammaFit, t, dEdt, p0=[1, 1])

            # Calculate residuals and R^2 value
            residuals = dEdt - cls.gammaFit(np.array(t), *poptGamma)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((dEdt - np.mean(dEdt)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Check if the fit is good enough
            if r_squared > r2_threshold or ignore:
                # Generate curve data
                x_line = np.arange(min(t), max(t), bin_size)
                y_line_gamma = cls.gammaFit(x_line, *poptGamma)
                return cls(x_line, y_line_gamma, energy, r_squared)
            else:
                print('Low r-squared.')

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