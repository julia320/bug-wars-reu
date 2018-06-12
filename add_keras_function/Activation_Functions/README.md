This project provides polynomial approximations of the activations functions for
neural networks.

## Dependencies:

- Keras
- Tensorflow

If you want to find new approximations:
- sympy (get it here: https://github.com/sympy/sympy)
- pyProximation (get it here: https://github.com/mghasemi/pyProximation )

## Structure:

### functions.py

All activation functions are in functions.py . No other functions should be in 
in that file. When that file gets imported all functions defined will be added 
to the custom objects of Keras and can be used with their name.

Example:

    model.add( Activation( 'polyTanh269' ) )
    
If you want to add your own activation simply added it to the file. The importing
while be handled automacilly for you.

Example:

    def myFunction( x ): #not a very good name and function
        return x
        
The naming convention for the functions is that they should contain the name of
the function they are aproximating ( if they are ).
Currently the following functions are approximated:

- Sig (Sigmoid)
- ReLU
- Tanh


### tools.py

tools.py is where all helper functions should go. It provides functions to find
the best fitting approximation over a given interval, plotting, etc.
