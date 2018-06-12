import os
from sympy import *
from sympy.utilities.lambdify import implemented_function
# import the module
from pyProximation import Measure, OrthSystem, Graphics
import matplotlib.pyplot as plt
import time

def polyGen( meanList, function, maxDegree = 5, outFile = None, messure_exp = [ 2 ], plotPath = None, uniqueNames = False ):
    """
    TODO:
        - file output is bad
        - plotting for Too approximation

    Finds number of polynomial approximations
    
    polyGen( meanList, function [, maxDegree = 5, outFile = None, messure_exp = [ 2 ], plotFile = None, uniqueNames = False ] )

    Parameters
    ----------
    meanList : array-like of integers
        values to be used to for search intervals
    function  : str
        the function that will be approximated.
        Must be one of the following:
            relu
            tanh
            sig
    maxDegree : int
        maximum degree of the approximation polynomial
    oufile : string
        file to where outputs will be written. No output
        is written if None
    messure_exp : array-like of integers
        exponets that will be tried in the messure 
    plotPath : string
        path that plots will be saved to. 
        No plotting if None
    uniqueNames : bool
        if True a timestamp will be added to the function name to make it unique
    Returns
    -------
    dict
        containing all found plynomial approximations as string and a name for the function
        key : name of the approximation with the following pattern: function_ApproximationMethode_Interval_Degree[_MessureExp[_timestamp]]
            messureExp is only present for Too approximation
            timestamp is only present if uniqueNames is True
            e.g.    sig_Too_100_3_4
                    relu_Legendre_50_2_1508519466.510187
        value : string represention of the function
            e.g. '0.2312314*x**3+0.1231243587903457*x**2'

    """
    x = Symbol('x')

    polyFile = None

    if outFile is not None: 
        polyFile = open( outFile, "w" )
    # define  functions symbolically and numerically
    ReLU = implemented_function(Function('ReLU'), lambda a: max(0, a))
    Sigmoid = implemented_function(Function('sigmoid'), lambda a: 1/(1+exp(-a)))
    if function == 'relu':
        f = ReLU(x)
    elif function == 'sig':
        f = Sigmoid(x)
    elif function == 'tanh':
        f = tanh( x )
    else:
        raise Exception()

    counter = 0
    
    results = {}

        
    for l in meanList:
        D = [ ( (-1) * l, l ) ]
        w = lambda x: 1. / sqrt( 1. - ( x / l )**2 )
        
        # Half the length of a symmetric interval about 0
        #l = 5000 # Half the length of a symmetric interval about 0
        if polyFile is not None:
            polyFile.write("#Interval = ["+str(-l)+","+str(l)+"]\n")
    
        for precision in range(1):
            ring=RealField(precision)
            if polyFile is not None:
                polyFile.write("#presicion = "+str(precision)+"\n")

            for degree in range( 2, maxDegree + 1 ):
                str1 = ""
                M = Measure(D, w)
                S = OrthSystem([x], D, 'sympy')
                T = OrthSystem([x], D, 'sympy')
                
                # link the measure to S
                S.SetMeasure(M)
                
                # set B = {1, x, x^2, ..., x^n}
                B = S.PolyBasis(degree)
                # link B to S
                S.Basis(B)
                T.Basis(B)
                
                # generate the orthonormal basis
                S.FormBasis()
                T.FormBasis()
                
                # number of elements in the basis
                m = len(S.OrthBase)
                # extract the coefficients
                Coeffs1 = S.Series(f)
                Coeffs2 = T.Series(f)
                
                # form the approximation
                f_app1 = sum([S.OrthBase[i] * Coeffs1[i] for i in range(m)])
                f_app2 = sum([T.OrthBase[i] * Coeffs2[i] for i in range(m)])

                #construct name
                #function_ApproximationMethode_Interval_Degree[_MessureExp[_timestamp]]
                name = '' + function + '_{}_' + str( l ) + '_' + str( degree )
                if uniqueNames:
                    name += ( '_' + str( time.time() ) )
                results[ name.format( 'Chebyshev' ).replace( '.', 'd' ) ] = str( f_app1 )
                results[ name.format( 'Legendre' ).replace( '.', 'd' ) ] = str( f_app2 )


                

                for exponet in messure_exp:
                    mu = lambda x: exp( -1./( 1e-5+( x / l )**exponet ) )
                    N = Measure(D, mu)
                    U = OrthSystem([x], D, 'sympy')
                    # link the measure to U
                    U.SetMeasure(N)
                     # link B to U
                    U.Basis(B)
                     # generate the orthonormal basis
                    U.FormBasis()  
                    # extract the coefficients
                    Coeffs3 = U.Series(f)  
                    # form the approximation   
                    f_app3 = sum([U.OrthBase[i] * Coeffs3[i] for i in range(m)])
                    #function_ApproximationMethode_Interval_Degree[_MessureExp[_timestamp]]
                    name = '' + function + '_Too_' + str( l ) + '_' + str( degree ) + '_' + str( exponet )
                    if uniqueNames:
                        name += ( '_' + str( time.time() ) )
                    results[ name.replace( '.', 'd' ) ] = str( f_app3 )

                counter = counter + 1
                if polyFile is not None:            
                  polyFile.write("#Chebyshev-l"+ (str(l))+ "-d"+str(n)+":\n")

                if polyFile is not None:                
                    str1 = "def polyReLUInteg" + (str(counter))+ "(x):\n    return "
                    polyFile.write(str1)
                    str1 = str(integ_f_app2) + "\n\n"
                    polyFile.write(str1)

                if plotPath is not None:
                    #print "Chebyshev: ", f_app1
                    #print "Legendre: ", f_app2
                    #print "The other one: ", f_app3
                    G = Graphics('sympy', numpoints=100)
                    G.Plot2D(f_app1, (x, -1*l, l), color='red', legend='Chebyshev')
                    G.Plot2D(f_app2, (x, -1*l, l), color='green', legend='Legendre')
                    G.Plot2D(f_app3, (x, -1*l, l), color='pink', legend="Too")
                    G.Plot2D(f(x), (x,-1*l, l), color='blue', legend=function)
                    G.SetTitle("Approximations for {} of degree {} and meassure: {} ".format( function, degree, exponet ) )
                    G.save( plotPath + '/xxx' + function + '_' + str( l ) + '_' + str( degree ) + '_' + str( exponet ) + '.png' )
                    plt.clf()
  


    return results

if __name__ == '__main__':
   for name, function in polyGen( [100], 'relu', maxDegree = 5, outFile = None, messure_exp = [ 2, 4, 8 ], uniqueNames = True ).iteritems():
    print( '{} : {}'.format( name, function ) )
