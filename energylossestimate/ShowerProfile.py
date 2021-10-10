'''
Some basic outline & pseudocode:
1. we still want to load the data. (thankfully we have this from the other model to pull from)
2. fitting dE(t)/dt = P(t) = E * (B*t)**(a-1)*B*exp(-B*t)/Gamma(a)
    - we have from our data the given values for E and dE(t)/dt = P(t)
    - E = energy of the event in the beginning
    - P = energy measured by the calorimeter (hit energy)
    - Gamma is a statistical function provided by various python libraries!
        - see here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html
    
What we are essentially attempting here is to produce all the possible 
alphas (a) and Betas(b), rewriting the equation with a and b as our unknowns.

3. fitting the curve: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
it is pretty simple with scipy.optimize.curve_fit, where we add a function and the data and it will give us the unknown paramters!
function should take in the unknown params (a and b in our case) and E and P for x and y.
- our E and P data should be split up by magnitude (based on what Rhea did) so we should attempt this first
- ideally our final model will be able to recognize the magnitude and feed it into an array of its relevant fitted equation (i believe.)
 ** material differences --> maybe why Rhea took a different approach to the fitting than using a library **
4. return function & attempt plots :)
-  this is a more flexible part, we will probably have to play around with a small dataset to test out what is possible & efficient.
    
5. Further steps:

Create alpha and beta distributions for specific energies

Account for variation from individual events.
'''