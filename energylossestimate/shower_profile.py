###read in data, argparse, etc

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

### add predicted shower energy to event data instances

# - iterate through event data instances
# - call shower profile func, pass in:
# - - individual event measured energy, individual event t
# - set event.shower_energy to result

### save data/end program
