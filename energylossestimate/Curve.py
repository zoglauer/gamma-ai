import math

import numpy as np
from scipy.optimize import curve_fit


class Curve:
    """The energy distribution of a particle shower. """

    def __init__(self, x, y, E):
        self.t = x
        self.dEdt = y
        self.energy = E

    @classmethod
    def fit(cls, t, dEdt, energy, bin_size, ignore=False):
        """If fit is possible, returns Curve object. Otherwise, returns None."""

        if len(dEdt) >= 6:  # minimum fit data required

            # define weights based on y values
            # weights = 1 / np.array(dEdt)
            # poptPoly, _ = curve_fit(Curve.poly4Fit, t, dEdt, sigma=weights)

            # fitting a polynomial curve
            poptPoly, pcov = curve_fit(Curve.poly4Fit, t, dEdt)
            a, b, c, d, e = poptPoly

            # R^2 value
            residuals = dEdt - Curve.poly4Fit(np.array(t), a, b, c, d, e)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((dEdt - np.mean(dEdt)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # TODO: adjust this parameter for curve quality
            if r_squared > 0.2 or ignore:

                # curve data
                x_line = np.arange(min(t), max(t), bin_size)
                y_line_poly = Curve.poly4Fit(x_line, a, b, c, d, e)

                return cls(x_line, y_line_poly, energy)

        return None

    def compare(self, curve, bin_size):
        """Return a factor rating the similarity of the given curve to self.
        A smaller value indicates a closer similarity. """

        # zero pad both signals to put them in the same plane
        y1 = Curve.zeroPad(curve.t, curve.dEdt, bin_size)
        y2 = Curve.zeroPad(self.t, self.dEdt, bin_size)

        corrs = []
        for s in range(len(y1)):
            signal_shifted = Curve.shift(y1, s)
            corrs.append(Curve.squareDifferences(signal_shifted, y2))

        return min(corrs)

    @staticmethod
    def squareDifferences(s1, s2):
        return sum(map(lambda p: (p[0] - p[1]) ** 2, zip(s1, s2)))

    @staticmethod
    def zeroPad(x, y, bin_size):
        """Assumes x is ordered ascending. """
        binned_x = 0
        signal = []
        arbitrarilyHighNumber = 20

        # zeros before main data
        while binned_x < arbitrarilyHighNumber:
            if binned_x in x:
                signal.append(y[list(x).index(binned_x)])
            else:
                signal.append(0)
            binned_x += bin_size

        return signal

    @staticmethod
    def shift(lst, k):
        return lst[-k:] + lst[:-k]

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
