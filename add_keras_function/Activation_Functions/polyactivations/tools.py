from __future__ import absolute_import
from . import functions as f
import inspect
import keras.utils.generic_utils
import numpy as np
try:
  import matplotlib.pyplot as plt
  noPlot = False
except:
  noPlot = True
from . import polyGen as pg

__sig = {}
__relu = {}
__tanh = {}

debug = True


def _relu( x ):
  """
  internal helper.

  applies the relu funtcion to an array
  """
  return np.maximum( 0, x )

def _sigmoid( x ):
  """
  internal helper.

  applies the relu funtcion to an array
  """
  return 1.0  / ( 1.0  + np.exp( -x ) )

def _tanh( x ):
  """
  internal helper.

  applies the tanh funtcion to an array
  """
  return np.tanh( x )

def __initialize():
  """
  Sort the functions in the correct dicts for convience
  """
  global __sig, __relu, __tanh 
  for key, value in f.functions.iteritems():
    if key.lower().rfind( 'sig' ) != -1:
      __sig[ key ] = value
    elif key.lower().rfind( 'tanh' ) != -1:
      __tanh[ key ] = value
    elif key.lower().rfind( 'relu' ) != -1:
      __relu[ key ] = value

def __debug( msg ):
  if debug:
    print( msg )


def findBestExistingApproximation( mini, maxi, function, step = 0.01 ):
  """
  Finds a polynomial approximation over the given intervall for the given function
    
  Parameters
  ----------
  mini : number
      lower bound of the interval
  maxi : number
      upper bound of the interval
  function : str
     the function that will be approximated.
     Must be one of the following:
          relu
          tanh
          sig
  
  Returns
  -------
  tuple ( string, <function> )
      the name of the python function and the python function for the polynomial
  """
  global __sig, __relu, __tanh 
  penalty = 1
  if mini > maxi:
    raise Excpetion()
  
  if function == 'sig':
    dic = __sig
    y = _sigmoid
  elif function == 'relu':
    dic = __relu
    y = _relu
  elif function == 'tanh':
    dic = __tanh
    y = _tanh
  else:
    raise Excpetion()
  error = float( 'InF' )
  for name, f in dic.iteritems():
    e = _error( [ mini, maxi ], y, f )
    if e < error:  
      error = e
      best = ( name, f )
      __debug( '{} : error ({}) '.format( name, e ) )
  return best


def _error( interval, f1, f2, step = 0.01 ):
  x = np.arange( interval[ 0 ], interval[ 1 ], step )
  e = np.sum( ( f1( x ) - f2( x ) )**2 )
  return e


def plot( function, interval = [ -1.0, 1.0], step = 0.1 ):
  if noPLot:
    print( 'matplot lib not found. can not plot' )
    return
  x = np.arange( interval[ 0 ], interval[ 1 ], step )
  if inspect.isfunction( function ):
    title = function.__name__
  else:
    title = function
    function = f.functions[ function ]
  y = function( x )
  plt.plot( x, y )
  plt.title( title )
  plt.show()


def createApproximation( mini, maxi, function, maxDegree = 5, messure_exp = [ 2, 4, 8 ], uniqueNames = True ):
  """
  Finds a polynomial approximation over the given intervall for the given function
  
  createApproximation( mini, maxi, function [, maxDegree = 5, messure_exp = [ 2, 4, 8 ], uniqueNames = True ] )
  
  Parameters
  ----------
  mini : number
      lower bound of the interval
  maxi : number
      upper bound of the interval
  function : str
     the function that will be approximated.
     Must be one of the following:
          relu
          tanh
          sig
  maxDegree : int
      maximum degree of the approximation polynomial
    messure_exp : array-like of integers
      exponets that will be tried in the messure 
  uniqueNames : bool
      if True a timestamp will be added to the function name to make it unique
  Returns
  -------
  tuple ( string, <function> )
      the name of the python function and the python function for the polynomial
  """
  functions = pg.polyGen( [ max( abs( mini ), abs( maxi ) ) ], function, maxDegree = maxDegree, messure_exp = messure_exp, uniqueNames = uniqueNames )
  if function == 'sig':
    y = _sigmoid
  elif function == 'relu':
    y = _relu
  elif function == 'tanh':
    y = _tanh
  else:
    raise Excpetion()
  error = float( 'InF' )
  for name, f in functions.iteritems():
    func = createPythonFunction( name, f )
    e = _error( [ mini, maxi ], y, func )
    if e < error:  
      error = e
      best = ( name, func )
      __debug( '{}/{} : error ({}) '.format( name, f, e ) )
  return best

def createPythonFunction( name, function, klobals = {} ):
  """
  Creates a python function out of the given arguements

  createPythonFunction( name, function, [ klobals = {} ] )
  
  The function takes one argument x and will return whatever is specified in the function argument.

  The function is built using the following pattern:

  def name( x ):
    return function

  Example:

  createPythonFunction( 'test' 'x+1')

  will return function that is like:

  def test( x ):
    return x+1

  Parameters
  ----------
  name  : str
      name of the ptyhon function
  function  : str
      the computation of to be returned. x is the only variablie that can be used here. 
  klobals : dict
      dictornay of globals that should be used in the returned function. By default an empty dict is passed
      allowing only for globals to be used
  
  Returns
  -------
  FunctionType
      a python FunctionType object with the given name taking exactly one argument
  """
  template = """
def {}( x ):
  return {}
  """
  exec( template.format( name, function ), klobals )
  return klobals[ name ]


def registerWithKeras( name, function ):
  """
  Registers function with keras to be used as an activation function
  """
  keras.utils.generic_utils.get_custom_objects()[ name ] = function

# initialization code
__debug( 'running init code' )

__initialize()

if __name__ == '__main__':
  pass