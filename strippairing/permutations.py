import ROOT
import array
import sys 
 

import itertools

# A) Create the multiples
def CreateMultiples(X, Y):
  MultiplesFirst = []
  for I in range(Y):
    MultiplesFirst.append(I)
  NSpares = X - Y
  SpareCombinations = itertools.combinations_with_replacement(MultiplesFirst, NSpares)
  AllMultiples = []
  for A in SpareCombinations:
    B = list(A)
    for C in MultiplesFirst:
      B.append(C)
    B.sort()
    AllMultiples.append(B)
  
  #print(AllMultiples)
  
  return AllMultiples


# B) Create the permutations
def CreatePermutations(X, Y):
  # Create multiples
  Multiples = CreateMultiples(X, Y)  
  
  Permutations = []
  for M in Multiples:
    A = []
    Perm = itertools.permutations(M)
    for P in Perm:
      A.append(list(P))
    # Remove duplicates:
    A.sort()
    B = list(A for A,_ in itertools.groupby(A))
    # Add them to the final list
    for b in B:
      Permutations.append(b)
  
  
  return Permutations



# C) Create the strip combinations for X > Y
def CreateSortedStripCombinations(X, Y):

  Permutations = CreatePermutations(X, Y)
  
  # Now create the real strip combinations
  FullCombies = []
  Xes = []
  for E in range(X):
    Xes.append(E)  
  
  for C in Permutations:
    NewList = []
    for E in range(X):
      Tuple = []
      Tuple.append(E)
      Tuple.append(C[E])
      NewList.append(Tuple)
    FullCombies.append(NewList)

  #print(FullCombies)
  
  return FullCombies



# C) Create the strip combinations:
def CreateStripCombinations(X, Y):
  if X > Y:
    Combies = CreateSortedStripCombinations(X, Y)
  else:
    Combies = CreateSortedStripCombinations(Y, X)
    # Invert
    for C in Combies:
      for P in C:
        Temp = P[1]
        P[1] = P[0]
        P[0] = Temp

  return Combies

  