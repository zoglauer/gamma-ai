import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class Curve:
    """The energy distribution of a particle shower. """

    def __init__(self, x, y, E, r2):
        self.t = x
        self.dEdt = y
        self.energy = E
        self.r_squared = r2

    @classmethod
    def fit(cls, t, dEdt, energy, bin_size, ignore=False):
        """If fit is possible, returns Curve object. Otherwise, returns None."""

        if len(dEdt) >= 6:  # minimum fit data required

            # fitting a polynomial curve
            poptPoly, pcov = curve_fit(Curve.poly4Fit, t, dEdt)
            a, b, c, d, e = poptPoly

            # R^2 value
            residuals = dEdt - Curve.poly4Fit(np.array(t), a, b, c, d, e)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((dEdt - np.mean(dEdt)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # display a "good" curve
            """
            if r_squared > 0.9:

                print(r_squared)

                # curve data
                x_line = np.arange(min(t), max(t), bin_size)
                y_line_poly = Curve.poly4Fit(x_line, a, b, c, d, e)

                plt.plot(x_line, y_line_poly)
                plt.xlabel("penetration (radiation lengths)")
                plt.ylabel("energy deposited")
                plt.show()
            """

            # TODO: adjust this parameter for curve quality
            if r_squared > 0.6 or ignore:

                # curve data
                x_line = np.arange(min(t), max(t), bin_size)
                y_line_poly = Curve.poly4Fit(x_line, a, b, c, d, e)

                return cls(x_line, y_line_poly, energy, r_squared)

        return None

    def compare(self, curve, bin_size):
        """Return a factor rating the similarity of the given curve to self.
        A smaller value indicates a closer similarity. """

        # zero pad both signals to put them in the same plane
        y1, y2 = Curve.zeroPad(self.t, curve.t, self.dEdt, curve.dEdt, bin_size)

        corrs = []
        for s in range(len(y1)):
            signal_shifted = Curve.shift(y1, s)
            corrs.append(Curve.squareDifferences(signal_shifted, y2))

        return min(corrs)

    @staticmethod
    def squareDifferences(s1, s2):
        diff = s1 - s2
        return np.dot(diff, diff)

    @staticmethod
    def zeroPad(x1, x2, y1, y2, bin_size):
        x_min = min(x1[0], x2[0])
        x_max = max(x1[-1], x2[-1])
        ln = int((x_max - x_min) / bin_size) + 2

        y1_padded = np.zeros(ln)
        y2_padded = np.zeros(ln)

        for i in range(len(y1)):
            index_of_y1 = int((x1[i] - x_min) / bin_size)
            y1_padded[index_of_y1] = y1[i]

        for i in range(len(y2)):
            index_of_y2 = int((x2[i] - x_min) / bin_size)
            y2_padded[index_of_y2] = y2[i]

        return y1_padded, y2_padded

    @staticmethod
    def shift(lst, k):
        return np.concatenate([lst[-k:], lst[:-k]])

    @staticmethod
    def poly2Fit(x, a, b, c):
        """Order 2 polynomial"""
        return a * x + b * x ** 2 + c

    @staticmethod
    def poly4Fit(x, a, b, c, d, e):
        """order 4 polynomial"""
        return a * x + b * x ** 2 + c * x ** 3  + d * x ** 4 + e

    @staticmethod
    def gammaFit(t, b, a):
        return b * (((b * t) ** (a - 1)) * math.e ** (- b * t)) / math.gamma(a)