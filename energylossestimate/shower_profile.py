###read in data, argparse, etc
import pickle
import argparse
import os
import sys
from math import exp
import scipy
from scipy import optimize, special, spatial
import numpy as np
from event_data import EventData
import time

start_time = time.time()

parser = argparse.ArgumentParser(description=
        'Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz',
        help='File name used for training/testing')
parser.add_argument('-s', '--savefileto', default='shower_output/shower_events.pkl',
                    help='save file name for event data with shower profile estimates.')

args = parser.parse_args()



#define geometry of system with following:
# - x,y,z bounds within which it's silicon
# - overall x,y,z dimensions of detector
# - these maybe should be made into a class in event_data that can be imported

#go through events and create 'master list' of hits
# -> should be a list where each element is a coordinate/energy pair

#define x0 for silicon and x0 for other material

### func to find 't' for radiation depth - NOTE: NEEDS TO BE UNIT CHECKED

#create bins as follows: - NOTE: MAY WISH TO MORE FINELY SLICE X/Y
# z dimension: 0 to overall z dimension of detector, width of 0.5mm
# x dimension: 0 to overall x dimension of detector, 'widths' being:
# --> < silicon start, > silicon start and < silicon end, > silicon end
# y dimension: 0 to overall y dimension of detector, 'widths' being:
# --> < silicon start, > silicon start and < silicon end, > silicon end
# each bin should have:
# - (x, y, and z range)
# - energy_inside init. to 0
# - 't' init. to 0
# data structure - hashmap, where the key includes material/location
# function as value?
#dictionary: stores different trained parameters
# sort into xyz bins by numbering bins and adding that number to the
# hit list and then just do dictionary[bin number] += hit_energy

#go through passed in list of hits and:
# 1. find bin in which the hit goes based on (x,y,z)
# 2. add energy for that hit to the energy_inside for the bin

#for each bin, define t as follows:
# 1. multiply energy_inside by xy area of slice/bin
# 2. divide by appropriate x0

### fit for alpha, beta

#define function for actual shower profile equation
# - takes in measured energy, list of hits from 1 event; alpha, beta
# - gamma function imported
# - calculates t using func to find t operating on list of hits
# - plugs into eqn (dE/dt) / ((beta*t)^(alpha-1) * beta*e^(-beta*t))/(gamma(alpha))
# - returns eqn result/total energy

#use optimize.curve_fit on shower profile func
# - pass in total measured energy, t from master list of hits
# - get out alpha, beta

#store alpha beta in eventdata as class var
# ^ separately in EventData there should be a dictionary to match appropriate
# alpha/beta with bin (? for later) 

### add predicted shower energy to event data instances

# - iterate through event data instances
# - call shower profile func, pass in:
# - - individual event measured energy, individual event t
# - set event.shower_energy to result

### save data/end program


def t_calculate(hits, geometry):
    #sorts the hits into bins that hold energy in each bin based on geometry
    # -
    # for each bin:
        # t = find the summed hit energy / area of bin / zloc?(or zbin height) / x0

    # each t is now for each x y z bin
    # so we know from each hit, their xyz coords,
    # so we take the xzy bin t and apply it to each event hit
    # when we calculate shower profile stuff
    # hits into bins
    # energy in each bin weighted [area of the bin] * zbin height / x0
    return t

events = #list of events from event_extractor

def shower_profile(inputs), alpha, beta): # inputs is a tuple of all data taken in from event data. (measured energy, t (calculated from xyz coords))
    
    # inverse of the eqn rhea had, 
    return gamma_energy #'true' energy

#create list of measured/t/true based on events
#gamma[event 1] *  sp(a[event 1], b[event 1])= measured[hit 1]
#gamma[event 1] * sp(a[event 1], b[event1]) = measured[hit 2]
# a might be just a[Si] or a[Csl] similar for b.

# OR

#its:
#gamma[event1] = measured[hit 1] * some weight + measured[hit2] * some weight ...
# where some weight related to alpha beta and t somehow?
# some weight = alpha-beta part of the shower profile function which includes t.
# measured[for each hit] / [alpha/beta/[t for event]] = gamma energy[event 1]
# sum across hits(measured[hit n] / [alpha/beta/[t for hit]]) = gamma energy[event]
#^^^^ go with this

# store t for each hit in EventData, store shower_energy
# store alpha/beta as class variable

def shower_profile_fit(list_of_measured_e, list_of_t, list_of_true_e):
    # scipy fit
    return alpha, beta


def find_prediction(take the estimated params and gamma):

def predicted_shower(event):
    event.shower_energy = shower_profile(inputs, alpha, beta)

def error():
    event.gamma - event.shower_energy
    





