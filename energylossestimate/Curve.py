import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

class Curve:
    """The energy distribution of a particle shower. """

    def __init__(self, t, dEdtoverE0, x, y, E, a, b):
        self.t = t
        self.dEdtoverE0 = dEdtoverE0
        self.x = x 
        self.y = y
        self.energy = E
        self.a = a
        self.b = b

    @classmethod
    def fit(cls, t, dEdtoverE0, energy, bin_size):
        """If fit is possible, returns Curve object. Otherwise, returns None."""

        # Attempt to fit the gamma distribution to the data
        t_max = max(t)
        b_est = 0.5
        a_est = t_max * b_est + 1
        
        # Remove Vacuum Points
        t = [t[i] for i in range(len(t)) if dEdtoverE0[i] > 1e-9]
        dEdtoverE0 = [pt for pt in dEdtoverE0 if pt > 1e-9]
        assert len(t) == len(dEdtoverE0), "Length Check Failed."
        
        if not dEdtoverE0:
            return None
        
        try:
            poptGamma, pcov = curve_fit(cls.gammaFit, t, dEdtoverE0, p0=[a_est, b_est])

            # Generate curve data
            x_line = np.arange(min(t), max(t), bin_size)
            y_line_gamma = cls.gammaFit(x_line, *poptGamma)
            return cls(t, dEdtoverE0, x_line, y_line_gamma, energy, poptGamma[0], poptGamma[1])

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


    @classmethod
    def fit_bayesian(cls, t, dEdtoverE0, energy, bin_size):
        t_max = max(t)
        b_est = 0.5
        a_est = t_max * b_est + 1

        t = np.array([t[i] for i in range(len(t)) if dEdtoverE0[i] > 1e-9])
        dEdtoverE0 = np.array([pt for pt in dEdtoverE0 if pt > 1e-9])

        if not len(dEdtoverE0):
            return None

        with pm.Model() as model:
            # Priors for unknown model parameters
            a = pm.Gamma('a', alpha=a_est, beta=1.0)
            b = pm.Gamma('b', alpha=b_est, beta=1.0)
            
            # Y_obs = pm.Gamma('Y_obs', alpha=a, beta=b, observed=dEdtoverE0)

            trace = pm.sample(1000, return_inferencedata=False)

        a_post = np.mean(trace['a'])
        b_post = np.mean(trace['b'])

        x_line = np.arange(min(t), max(t), bin_size)
        y_line_gamma = cls.gammaFit(x_line, a_post, b_post)
        
        return cls(t, dEdtoverE0, x_line, y_line_gamma, energy, a_post, b_post)
