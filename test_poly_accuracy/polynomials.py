# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:06:43 2018

@author: Julia

list of polynomial approximations for use as activation functions
"""

from __future__ import absolute_import
import keras.activations
import keras.utils.generic_utils
import sys, inspect


__name = 'polynomials'


###############################################################################################
#Interval = [-10,10]
#presicion = 2
#Chebyshev-l10-d2:
##Norm-2 of the approximation error Chebyshev = 3.347168
def polyReLUInteg2(x):
    return 0.0263068516521283*x**2 + 0.500000000000111*x

'''#Chebyshev-l10-d3:
##Norm-2 of the approximation error Chebyshev = 3.555147
def polyReLUInteg4(x):
    return -7.72574062998687e-5*x**4 + 0.0403273407849185*x**2 + 0.500000000000111*x
'''
#Interval = [-20,20]
#presicion = 2
#Chebyshev-l20-d2:
##Norm-2 of the approximation error Chebyshev = 10.141410
def polyReLUInteg6(x):
    return 3.19361719653008e-14*x**3 + 0.0131555717886246*x**2 + 0.499999999964823*x
'''
#Chebyshev-l20-d3:
##Norm-2 of the approximation error Chebyshev = 10.755274
def polyReLUInteg8(x):
    return -9.67670125926052e-6*x**4 + 3.19361719653008e-14*x**3 + 0.0201799900985077*x**2 + 0.499999999964823*x
'''
#Interval = [-30,30]
#presicion = 2
#Chebyshev-l30-d2:
##Norm-2 of the approximation error Chebyshev = 19.046269
def polyReLUInteg10(x):
    return 0.00877038133755076*x**2 + 0.500000000000006*x
'''
#Chebyshev-l30-d3:
##Norm-2 of the approximation error Chebyshev = 20.187448
def polyReLUInteg12(x):
    return -2.86717203622348e-6*x**4 + 0.0134533289889012*x**2 + 0.500000000000006*x
'''
#Interval = [-40,40]
#presicion = 2
#Chebyshev-l40-d2:
##Norm-2 of the approximation error Chebyshev = 29.644021
def polyReLUInteg14(x):
    return 0.00657778600317751*x**2 + 0.499999999999978*x
'''
#Chebyshev-l40-d3:
##Norm-2 of the approximation error Chebyshev = 31.410968
def polyReLUInteg16(x):
    return -1.20958820289945e-6*x**4 + 0.010089996742032*x**2 + 0.499999999999978*x
'''
#Interval = [-50,50]
#presicion = 2
#Chebyshev-l50-d2:
##Norm-2 of the approximation error Chebyshev = 41.697705
def polyReLUInteg18(x):
    return 0.005262228802542*x**2 + 0.500000000000118*x
'''
#Chebyshev-l50-d3:
##Norm-2 of the approximation error Chebyshev = 44.175309
def polyReLUInteg20(x):
    return -6.19309159882969e-7*x**4 + 0.00807199739361859*x**2 + 0.500000000000118*x
'''
#Interval = [-60,60]
#presicion = 2
#Chebyshev-l60-d2:
##Norm-2 of the approximation error Chebyshev = 55.048886
def polyReLUInteg22(x):
    return 0.004385190668785*x**2 + 0.499999999999988*x
'''
#Chebyshev-l60-d3:
##Norm-2 of the approximation error Chebyshev = 58.312902
def polyReLUInteg24(x):
    return -3.58396504562067e-7*x**4 + 0.00672666449468323*x**2 + 0.499999999999988*x
'''
#Interval = [-70,70]
#presicion = 2
#Chebyshev-l70-d2:
##Norm-2 of the approximation error Chebyshev = 69.581888
def polyReLUInteg26(x):
    return 0.00375873485895858*x**2 + 0.500000000000131*x
'''
#Chebyshev-l70-d3:
##Norm-2 of the approximation error Chebyshev = 73.701378
def polyReLUInteg28(x):
    return -2.25695757974866e-7*x**4 + 0.00576571242401349*x**2 + 0.500000000000131*x
'''
#Interval = [-80,80]
#presicion = 2
#Chebyshev-l80-d2:
##Norm-2 of the approximation error Chebyshev = 85.207418
def polyReLUInteg30(x):
    return 0.00328889300158875*x**2 + 0.500000000000058*x
'''
#Chebyshev-l80-d3:
##Norm-2 of the approximation error Chebyshev = 90.246265
def polyReLUInteg32(x):
    return -1.51198525362048e-7*x**4 + 0.00504499837101156*x**2 + 0.500000000000058*x
'''
#Interval = [-90,90]
#presicion = 2
#Chebyshev-l90-d2:
##Norm-2 of the approximation error Chebyshev = 101.853829
def polyReLUInteg34(x):
    return 0.00292346044585667*x**2 + 0.499999999999846*x
'''
#Chebyshev-l90-d3:
##Norm-2 of the approximation error Chebyshev = 107.871750
def polyReLUInteg36(x):
    return -1.06191556907323e-7*x**4 + 0.00448444299645614*x**2 + 0.499999999999846*x
'''
#Interval = [-100,100]
#presicion = 2
#Chebyshev-l100-d2:
##Norm-2 of the approximation error Chebyshev = 119.461967
def polyReLUInteg38(x):
    return 0.002631114401271*x**2 + 0.500000000000026*x
'''
#Chebyshev-l100-d3:
##Norm-2 of the approximation error Chebyshev = 126.515239
def polyReLUInteg40(x):
    return -7.7413644985429e-8*x**4 + 0.00403599869681035*x**2 + 0.500000000000026*x
'''
#Interval = [-110,110]
#presicion = 2
#Chebyshev-l110-d2:
##Norm-2 of the approximation error Chebyshev = 137.981915
def polyReLUInteg42(x):
    return 0.00239192218297364*x**2 + 0.500000000000092*x
'''
#Chebyshev-l110-d3:
##Norm-2 of the approximation error Chebyshev = 146.123910
def polyReLUInteg44(x):
    return -5.81620172693485e-8*x**4 + 0.00366908972437426*x**2 + 0.500000000000092*x
'''
#Interval = [-120,120]
#presicion = 2
#Chebyshev-l120-d2:
##Norm-2 of the approximation error Chebyshev = 157.370811
def polyReLUInteg46(x):
    return 0.0021925953343925*x**2 + 0.499999999999792*x
'''
#Chebyshev-l120-d3:
##Norm-2 of the approximation error Chebyshev = 166.652401
def polyReLUInteg48(x):
    return -4.47995630702801e-8*x**4 + 0.00336333224734218*x**2 + 0.499999999999792*x
'''
#Interval = [-130,130]
#presicion = 2
#Chebyshev-l130-d2:
##Norm-2 of the approximation error Chebyshev = 177.591326
def polyReLUInteg50(x):
    return 0.00202393415482385*x**2 + 0.499999999999871*x
'''
#Chebyshev-l130-d3:
##Norm-2 of the approximation error Chebyshev = 188.061205
def polyReLUInteg52(x):
    return -3.52360696338031e-8*x**4 + 0.00310461438216232*x**2 + 0.499999999999871*x
'''
#Interval = [-140,140]
#presicion = 2
#Chebyshev-l140-d2:
##Norm-2 of the approximation error Chebyshev = 198.610566
def polyReLUInteg54(x):
    return 0.00187936742947929*x**2 + 0.500000000000057*x
'''
#Chebyshev-l140-d3:
##Norm-2 of the approximation error Chebyshev = 210.315511
def polyReLUInteg56(x):
    return -2.82119697469024e-8*x**4 + 0.00288285621200831*x**2 + 0.500000000000057*x
'''
#Interval = [-150,150]
#presicion = 2
#Chebyshev-l150-d2:
##Norm-2 of the approximation error Chebyshev = 220.399259
def polyReLUInteg58(x):
    return 0.001754076267514*x**2 + 0.500000000000232*x
'''
#Chebyshev-l150-d3:
##Norm-2 of the approximation error Chebyshev = 233.384338
def polyReLUInteg60(x):
    return -2.29373762919736e-8*x**4 + 0.00269066579787334*x**2 + 0.500000000000232*x
'''
#Interval = [-160,160]
#presicion = 2
#Chebyshev-l160-d2:
##Norm-2 of the approximation error Chebyshev = 242.931134
def polyReLUInteg62(x):
    return 0.00164444650079438*x**2 + 0.499999999999742*x
'''
#Chebyshev-l160-d3:
##Norm-2 of the approximation error Chebyshev = 257.239881
def polyReLUInteg64(x):
    return -1.88998156702542e-8*x**4 + 0.0025224991855057*x**2 + 0.499999999999742*x
'''
#Interval = [-170,170]
#presicion = 2
#Chebyshev-l170-d2:
##Norm-2 of the approximation error Chebyshev = 266.182442
def polyReLUInteg66(x):
    return 0.00154771435368882*x**2 + 0.499999999999908*x
'''
#Chebyshev-l170-d3:
##Norm-2 of the approximation error Chebyshev = 281.857003
def polyReLUInteg68(x):
    return -1.5756899040392e-8*x**4 + 0.00237411688047685*x**2 + 0.499999999999908*x
'''
#Interval = [-180,180]
#presicion = 2
#Chebyshev-l180-d2:
##Norm-2 of the approximation error Chebyshev = 290.131573
def polyReLUInteg70(x):
    return 0.00146173022292833*x**2 + 0.49999999999968*x
'''
#Chebyshev-l180-d3:
##Norm-2 of the approximation error Chebyshev = 307.212832
def polyReLUInteg72(x):
    return -1.3273944613416e-8*x**4 + 0.0022422214982281*x**2 + 0.49999999999968*x
'''
#Interval = [-190,190]
#presicion = 2
#Chebyshev-l190-d2:
##Norm-2 of the approximation error Chebyshev = 314.758759
def polyReLUInteg74(x):
    return 0.00138479705330053*x**2 + 0.499999999999999*x
'''
#Chebyshev-l190-d3:
##Norm-2 of the approximation error Chebyshev = 333.286445
def polyReLUInteg76(x):
    return -1.12864331513885e-8*x**4 + 0.00212420984042597*x**2 + 0.499999999999999*x
'''
#Interval = [-200,200]
#presicion = 2
#Chebyshev-l200-d2:
##Norm-2 of the approximation error Chebyshev = 340.045819
def polyReLUInteg78(x):
    return 0.0013155572006355*x**2 + 0.500000000000167*x
'''
#Chebyshev-l200-d3:
##Norm-2 of the approximation error Chebyshev = 360.058598
def polyReLUInteg80(x):
    return -9.67670562318616e-9*x**4 + 0.00201799934840572*x**2 + 0.500000000000167*x
'''
#Interval = [-210,210]
#presicion = 2
#Chebyshev-l210-d2:
##Norm-2 of the approximation error Chebyshev = 365.975963
def polyReLUInteg82(x):
    return 0.00125291161965286*x**2 + 0.500000000000191*x
'''
#Chebyshev-l210-d3:
##Norm-2 of the approximation error Chebyshev = 387.511520
def polyReLUInteg84(x):
    return -8.35910214722701e-9*x**4 + 0.00192190414133861*x**2 + 0.500000000000191*x
'''
#Interval = [-220,220]
#presicion = 2
#Chebyshev-l220-d2:
##Norm-2 of the approximation error Chebyshev = 392.533621
def polyReLUInteg86(x):
    return 0.00119596109148682*x**2 + 0.499999999999917*x
'''
#Chebyshev-l220-d3:
##Norm-2 of the approximation error Chebyshev = 415.628729
def polyReLUInteg88(x):
    return -7.27025215867412e-9*x**4 + 0.00183454486218763*x**2 + 0.499999999999917*x
'''
#Interval = [-230,230]
#presicion = 2
#Chebyshev-l230-d2:
##Norm-2 of the approximation error Chebyshev = 419.704302
def polyReLUInteg90(x):
    return 0.0011439627831613*x**2 + 0.499999999999779*x
'''
#Chebyshev-l230-d3:
##Norm-2 of the approximation error Chebyshev = 444.394888
def polyReLUInteg92(x):
    return -6.36259102370715e-9*x**4 + 0.00175478204209158*x**2 + 0.499999999999779*x
'''
#Interval = [-240,240]
#presicion = 2
#Chebyshev-l240-d2:
##Norm-2 of the approximation error Chebyshev = 447.474474
def polyReLUInteg94(x):
    return 0.00109629766719625*x**2 + 0.499999999999933*x
'''
#Chebyshev-l240-d3:
##Norm-2 of the approximation error Chebyshev = 473.795673
def polyReLUInteg96(x):
    return -5.59994538378401e-9*x**4 + 0.00168166612367099*x**2 + 0.499999999999933*x
'''
#Interval = [-250,250]
#presicion = 2
#Chebyshev-l250-d2:
##Norm-2 of the approximation error Chebyshev = 475.831465
def polyReLUInteg98(x):
    return 0.0010524457605084*x**2 + 0.500000000000122*x
'''
#Chebyshev-l250-d3:
##Norm-2 of the approximation error Chebyshev = 503.817671
def polyReLUInteg100(x):
    return -4.95447327905646e-9*x**4 + 0.00161439947872289*x**2 + 0.500000000000122*x
'''
#Interval = [-260,260]
#presicion = 2
#Chebyshev-l260-d2:
##Norm-2 of the approximation error Chebyshev = 504.763373
def polyReLUInteg102(x):
    return 0.00101196707741192*x**2 + 0.499999999999935*x
'''
#Chebyshev-l260-d3:
##Norm-2 of the approximation error Chebyshev = 534.448283
def polyReLUInteg104(x):
    return -4.40450870421744e-9*x**4 + 0.00155230719108018*x**2 + 0.499999999999935*x
'''
#Interval = [-270,270]
#presicion = 2
#Chebyshev-l270-d2:
##Norm-2 of the approximation error Chebyshev = 534.258993
def polyReLUInteg106(x):
    return 0.000974486815285561*x**2 + 0.499999999999794*x
'''
#Chebyshev-l270-d3:
##Norm-2 of the approximation error Chebyshev = 565.675649
def polyReLUInteg108(x):
    return -3.93302062621258e-9*x**4 + 0.0014948143321541*x**2 + 0.499999999999794*x
'''
#Interval = [-280,280]
#presicion = 2
#Chebyshev-l280-d2:
##Norm-2 of the approximation error Chebyshev = 564.307751
def polyReLUInteg110(x):
    return 0.000939683714739644*x**2 + 0.500000000000092*x
'''
#Chebyshev-l280-d3:
##Norm-2 of the approximation error Chebyshev = 597.488576
def polyReLUInteg112(x):
    return -3.52649621835954e-9*x**4 + 0.00144142810600369*x**2 + 0.500000000000092*x
'''
#Interval = [-290,290]
#presicion = 2
#Chebyshev-l290-d2:
##Norm-2 of the approximation error Chebyshev = 594.899646
def polyReLUInteg114(x):
    return 0.000907280828024483*x**2 + 0.499999999999856*x
'''
#Chebyshev-l290-d3:
##Norm-2 of the approximation error Chebyshev = 629.876477
def polyReLUInteg116(x):
    return -3.17412132458978e-9*x**4 + 0.00139172368855524*x**2 + 0.499999999999856*x
'''
#Interval = [-300,300]
#presicion = 2
#Chebyshev-l300-d2:
##Norm-2 of the approximation error Chebyshev = 626.025200
def polyReLUInteg118(x):
    return 0.000877038133757*x**2 + 0.50000000000001*x
'''
#Chebyshev-l300-d3:
##Norm-2 of the approximation error Chebyshev = 662.829321
def polyReLUInteg120(x):
    return -2.86717203649778e-9*x**4 + 0.00134533289893685*x**2 + 0.50000000000001*x
'''
#Interval = [-310,310]
#presicion = 2
#Chebyshev-l310-d2:
##Norm-2 of the approximation error Chebyshev = 657.675416
def polyReLUInteg122(x):
    return 0.000848746581055162*x**2 + 0.500000000000258*x
'''
#Chebyshev-l310-d3:
##Norm-2 of the approximation error Chebyshev = 696.337583
def polyReLUInteg124(x):
    return -2.59855812109206e-9*x**4 + 0.00130193506348735*x**2 + 0.500000000000258*x
'''
#Interval = [-320,320]
#presicion = 2
#Chebyshev-l320-d2:
##Norm-2 of the approximation error Chebyshev = 689.841736
def polyReLUInteg126(x):
    return 0.000822223250397189*x**2 + 0.500000000000026*x
'''
#Chebyshev-l320-d3:
##Norm-2 of the approximation error Chebyshev = 730.392206
def polyReLUInteg128(x):
    return -2.36247695877931e-9*x**4 + 0.00126124959275239*x**2 + 0.500000000000026*x
'''
#Interval = [-330,330]
#presicion = 2
#Chebyshev-l330-d2:
##Norm-2 of the approximation error Chebyshev = 722.516010
def polyReLUInteg130(x):
    return 0.000797307394324546*x**2 + 0.500000000000032*x
'''
#Chebyshev-l330-d3:
##Norm-2 of the approximation error Chebyshev = 764.984559
def polyReLUInteg132(x):
    return -2.15414878775141e-9*x**4 + 0.00122302990812431*x**2 + 0.500000000000032*x
'''
#Interval = [-340,340]
#presicion = 2
#Chebyshev-l340-d2:
##Norm-2 of the approximation error Chebyshev = 755.690460
def polyReLUInteg134(x):
    return 0.000773857176844413*x**2 + 0.500000000000289*x
'''
#Chebyshev-l340-d3:
##Norm-2 of the approximation error Chebyshev = 800.106412
def polyReLUInteg136(x):
    return -1.96961238004966e-9*x**4 + 0.00118705844023857*x**2 + 0.500000000000289*x
'''
#Interval = [-350,350]
#presicion = 2
#Chebyshev-l350-d2:
##Norm-2 of the approximation error Chebyshev = 789.357654
def polyReLUInteg138(x):
    return 0.000751746971791714*x**2 + 0.500000000000101*x
'''
#Chebyshev-l350-d3:
##Norm-2 of the approximation error Chebyshev = 835.749897
def polyReLUInteg140(x):
    return -1.80556606380028e-9*x**4 + 0.001153142484803*x**2 + 0.500000000000101*x
'''
#Interval = [-360,360]
#presicion = 2
#Chebyshev-l360-d2:
##Norm-2 of the approximation error Chebyshev = 823.510485
def polyReLUInteg142(x):
    return 0.000730865111464167*x**2 + 0.500000000000025*x
'''
#Chebyshev-l360-d3:
##Norm-2 of the approximation error Chebyshev = 871.907490
def polyReLUInteg144(x):
    return -1.65924307667747e-9*x**4 + 0.00112111074911416*x**2 + 0.500000000000025*x
'''
#Interval = [-370,370]
#presicion = 2
#Chebyshev-l370-d2:
##Norm-2 of the approximation error Chebyshev = 858.142143
def polyReLUInteg146(x):
    return 0.000711112000343514*x**2 + 0.499999999999539*x
'''
#Chebyshev-l370-d3:
##Norm-2 of the approximation error Chebyshev = 908.571983
def polyReLUInteg148(x):
    return -1.52831313022853e-9*x**4 + 0.00109081045859757*x**2 + 0.499999999999539*x
'''
#Interval = [-380,380]
#presicion = 2
#Chebyshev-l380-d2:
##Norm-2 of the approximation error Chebyshev = 893.246095
def polyReLUInteg150(x):
    return 0.000692398526650263*x**2 + 0.499999999999859*x
'''
#Chebyshev-l380-d3:
##Norm-2 of the approximation error Chebyshev = 945.736462
def polyReLUInteg152(x):
    return -1.41080414392485e-9*x**4 + 0.00106210492021332*x**2 + 0.499999999999859*x
'''
#Interval = [-390,390]
#presicion = 2
#Chebyshev-l390-d2:
##Norm-2 of the approximation error Chebyshev = 928.816073
def polyReLUInteg154(x):
    return 0.000674644718274616*x**2 + 0.499999999999893*x
'''
#Chebyshev-l390-d3:
##Norm-2 of the approximation error Chebyshev = 983.394288
def polyReLUInteg156(x):
    return -1.30503961606742e-9*x**4 + 0.00103487146072095*x**2 + 0.499999999999893*x
'''
#Interval = [-400,400]
#presicion = 2
#Chebyshev-l400-d2:
##Norm-2 of the approximation error Chebyshev = 964.846047
def polyReLUInteg158(x):
    return 0.00065777860031775*x**2 + 0.499999999999946*x
'''
#Chebyshev-l400-d3:
##Norm-2 of the approximation error Chebyshev = 1021.539082
def polyReLUInteg160(x):
    return -1.20958820289749e-9*x**4 + 0.00100899967420263*x**2 + 0.499999999999946*x
'''
#Interval = [-410,410]
#presicion = 2
#Chebyshev-l410-d2:
##Norm-2 of the approximation error Chebyshev = 1001.330219
def polyReLUInteg162(x):
    return 0.000641735219822196*x**2 + 0.499999999999859*x
'''
#Chebyshev-l410-d3:
##Norm-2 of the approximation error Chebyshev = 1060.164703
def polyReLUInteg164(x):
    return -1.12322289266695e-9*x**4 + 0.000984389926051616*x**2 + 0.499999999999859*x
'''
#Interval = [-420,420]
#presicion = 2
#Chebyshev-l420-d2:
##Norm-2 of the approximation error Chebyshev = 1038.263003
def polyReLUInteg166(x):
    return 0.000626455809826429*x**2 + 0.499999999999905*x
'''
#Chebyshev-l420-d3:
##Norm-2 of the approximation error Chebyshev = 1099.265240
def polyReLUInteg168(x):
    return -1.04488776840257e-9*x**4 + 0.000960952070669045*x**2 + 0.499999999999905*x
'''
#Interval = [-430,430]
#presicion = 2
#Chebyshev-l430-d2:
##Norm-2 of the approximation error Chebyshev = 1075.639014
def polyReLUInteg170(x):
    return 0.000611887070063024*x**2 + 0.499999999999824*x
'''
#Chebyshev-l430-d3:
##Norm-2 of the approximation error Chebyshev = 1138.834993
def polyReLUInteg172(x):
    return -9.73670808676759e-10*x**4 + 0.000938604348095574*x**2 + 0.499999999999824*x
'''
#Interval = [-440,440]
#presicion = 2
#Chebyshev-l440-d2:
##Norm-2 of the approximation error Chebyshev = 1113.453058
def polyReLUInteg174(x):
    return 0.000597980545743409*x**2 + 0.50000000000028*x
'''
#Chebyshev-l440-d3:
##Norm-2 of the approximation error Chebyshev = 1178.868462
def polyReLUInteg176(x):
    return -9.087815198322e-10*x**4 + 0.000917272431093083*x**2 + 0.50000000000028*x
'''
#Interval = [-450,450]
#presicion = 2
#Chebyshev-l450-d2:
##Norm-2 of the approximation error Chebyshev = 1151.700118
def polyReLUInteg178(x):
    return 0.000584692089171334*x**2 + 0.4999999999999*x
'''
#Chebyshev-l450-d3:
##Norm-2 of the approximation error Chebyshev = 1219.360336
def polyReLUInteg180(x):
    return -8.49532455258715e-10*x**4 + 0.000896888599291275*x**2 + 0.4999999999999*x
'''
#Interval = [-460,460]
#presicion = 2
#Chebyshev-l460-d2:
##Norm-2 of the approximation error Chebyshev = 1190.375345
def polyReLUInteg182(x):
    return 0.000571981391580653*x**2 + 0.500000000000139*x
'''
#Chebyshev-l460-d3:
##Norm-2 of the approximation error Chebyshev = 1260.305485
def polyReLUInteg184(x):
    return -7.95323877964179e-10*x**4 + 0.000877391021046094*x**2 + 0.500000000000139*x
'''
#Interval = [-470,470]
#presicion = 2
#Chebyshev-l470-d2:
##Norm-2 of the approximation error Chebyshev = 1229.474052
def polyReLUInteg186(x):
    return 0.000559811574738511*x**2 + 0.500000000000007*x
'''
#Chebyshev-l470-d3:
##Norm-2 of the approximation error Chebyshev = 1301.698944
def polyReLUInteg188(x):
    return -7.45630977581736e-10*x**4 + 0.000858723126981079*x**2 + 0.500000000000007*x
'''
#Interval = [-480,480]
#presicion = 2
#Chebyshev-l480-d2:
##Norm-2 of the approximation error Chebyshev = 1268.991700
def polyReLUInteg190(x):
    return 0.000548148833598126*x**2 + 0.499999999999835*x
'''
#Chebyshev-l480-d3:
##Norm-2 of the approximation error Chebyshev = 1343.535909
def polyReLUInteg192(x):
    return -6.99993172973158e-10*x**4 + 0.00084083306183556*x**2 + 0.499999999999835*x
'''
#Interval = [-490,490]
#presicion = 2
#Chebyshev-l490-d2:
##Norm-2 of the approximation error Chebyshev = 1308.923894
def polyReLUInteg194(x):
    return 0.000536962122708367*x**2 + 0.499999999999987*x
'''
#Chebyshev-l490-d3:
##Norm-2 of the approximation error Chebyshev = 1385.811729
def polyReLUInteg196(x):
    return -6.58005125291425e-10*x**4 + 0.000823673203430615*x**2 + 0.499999999999987*x
'''
#Interval = [-500,500]
#presicion = 2
#Chebyshev-l500-d2:
##Norm-2 of the approximation error Chebyshev = 1349.266374
def polyReLUInteg198(x):
    return 0.0005262228802542*x**2 + 0.500000000000141*x
'''
#Chebyshev-l500-d3:
##Norm-2 of the approximation error Chebyshev = 1428.521893
def polyReLUInteg200(x):
    return -6.19309159883781e-10*x**4 + 0.000807199739362227*x**2 + 0.500000000000141*x
'''
#Interval = [-510,510]
#presicion = 2
#Chebyshev-l510-d2:
##Norm-2 of the approximation error Chebyshev = 1390.015010
def polyReLUInteg202(x):
    return 0.000515904784562941*x**2 + 0.500000000000313*x
'''
#Chebyshev-l510-d3:
##Norm-2 of the approximation error Chebyshev = 1471.662029
def polyReLUInteg204(x):
    return -5.83588853347323e-10*x**4 + 0.000791372293492035*x**2 + 0.500000000000313*x
'''
#Interval = [-520,520]
#presicion = 2
#Chebyshev-l520-d2:
##Norm-2 of the approximation error Chebyshev = 1431.165793
def polyReLUInteg206(x):
    return 0.000505983538705962*x**2 + 0.49999999999996*x
'''
#Chebyshev-l520-d3:
##Norm-2 of the approximation error Chebyshev = 1515.227893
def polyReLUInteg208(x):
    return -5.50563588028005e-10*x**4 + 0.000776153595540496*x**2 + 0.49999999999996*x
'''
#Interval = [-530,530]
#presicion = 2
#Chebyshev-l530-d2:
##Norm-2 of the approximation error Chebyshev = 1472.714830
def polyReLUInteg210(x):
    return 0.000496436679485095*x**2 + 0.499999999999893*x
'''
#Chebyshev-l530-d3:
##Norm-2 of the approximation error Chebyshev = 1559.215365
def polyReLUInteg212(x):
    return -5.1998391279692e-10*x**4 + 0.000761509188077555*x**2 + 0.499999999999893*x
'''
#Interval = [-540,540]
#presicion = 2
#Chebyshev-l540-d2:
##Norm-2 of the approximation error Chebyshev = 1514.658340
def polyReLUInteg214(x):
    return 0.000487243407642778*x**2 + 0.499999999999766*x
'''
#Chebyshev-l540-d3:
##Norm-2 of the approximation error Chebyshev = 1603.620442
def polyReLUInteg216(x):
    return -4.91627578275428e-10*x**4 + 0.000747407166076439*x**2 + 0.499999999999766*x
'''
#Interval = [-550,550]
#presicion = 2
#Chebyshev-l550-d2:
##Norm-2 of the approximation error Chebyshev = 1556.992648
def polyReLUInteg218(x):
    return 0.000478384436594727*x**2 + 0.499999999999959*x
'''
#Chebyshev-l550-d3:
##Norm-2 of the approximation error Chebyshev = 1648.439234
def polyReLUInteg220(x):
    return -4.6529613815463e-10*x**4 + 0.000733817944874766*x**2 + 0.499999999999959*x
'''
#Interval = [-560,560]
#presicion = 2
#Chebyshev-l560-d2:
##Norm-2 of the approximation error Chebyshev = 1599.714178
def polyReLUInteg222(x):
    return 0.000469841857369822*x**2 + 0.500000000000008*x
'''
#Chebyshev-l560-d3:
##Norm-2 of the approximation error Chebyshev = 1693.667955
def polyReLUInteg224(x):
    return -4.40812027295657e-10*x**4 + 0.000720714053002255*x**2 + 0.500000000000008*x
'''
#Interval = [-570,570]
#presicion = 2
#Chebyshev-l570-d2:
##Norm-2 of the approximation error Chebyshev = 1642.819453
def polyReLUInteg226(x):
    return 0.000461599017766843*x**2 + 0.499999999999783*x
'''
#Chebyshev-l570-d3:
##Norm-2 of the approximation error Chebyshev = 1739.302923
def polyReLUInteg228(x):
    return -4.18016042644645e-10*x**4 + 0.000708069946809027*x**2 + 0.499999999999783*x
'''
#Interval = [-580,580]
#presicion = 2
#Chebyshev-l580-d2:
##Norm-2 of the approximation error Chebyshev = 1686.305085
def polyReLUInteg230(x):
    return 0.000453640414012241*x**2 + 0.499999999999866*x
'''
#Chebyshev-l580-d3:
##Norm-2 of the approximation error Chebyshev = 1785.340554
def polyReLUInteg232(x):
    return -3.96765165573818e-10*x**4 + 0.000695861844277676*x**2 + 0.499999999999866*x
'''
#Interval = [-590,590]
#presicion = 2
#Chebyshev-l590-d2:
##Norm-2 of the approximation error Chebyshev = 1730.167777
def polyReLUInteg234(x):
    return 0.000445951593435763*x**2 + 0.500000000000345*x
'''
#Chebyshev-l590-d3:
##Norm-2 of the approximation error Chebyshev = 1831.777357
def polyReLUInteg236(x):
    return -3.76930674438425e-10*x**4 + 0.000684067575730754*x**2 + 0.500000000000345*x
'''


"""
list of all function names
"""
polynomials = {}

###################################################################

#__this = Functions()
this = sys.modules[ __name ]
objects = keras.utils.generic_utils.get_custom_objects()
for name, data in inspect.getmembers( this, inspect.isfunction ):
    objects[ name ] = data
    polynomials[ name ] = data
print( 'polynomial functions imported' )