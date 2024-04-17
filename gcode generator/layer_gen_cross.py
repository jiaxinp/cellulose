import os
import numpy as np

L = 20 #square size
d = 0.12 #line spacing
N = 1 #no. of layers
filename = 'cross_layer_20mm.gcode'
Ys = np.arange(0,L,d)
Ys = np.flip(Ys)
X = 0
E = 0
odd = True
with open(filename, 'w') as f:
    header_file = open("header.txt", "r")
    print(header_file.read(), file = f)
    for i in range(N):
        for Y in np.flip(Ys):
            if odd: 
                print("G1 X%f Y%f E%f" %(0, Y, E ),file = f)
                E+= d
                print("G1 X%f Y%f E%f" %(L, Y, E ),file = f)
                
                odd = False
            else:
                print("G1 X%f Y%f E%f" %(L, Y, E ),file = f)
                E+= d
                print("G1 X%f Y%f E%f" %(0, Y, E ),file = f)
                odd = True
            E+=L
        for X in Ys:
            if odd: 
                print("G1 X%f Y%f E%f" %(X, 0, E ),file = f)
                E+= d
                print("G1 X%f Y%f E%f" %(X, L, E ),file = f)
                odd = False
            else:
                print("G1 X%f Y%f E%f" %(X, L, E ),file = f)
                E+= d
                print("G1 X%f Y%f E%f" %(X, 0, E ),file = f)
                odd = True
            E+=L

    print("G0 F9000 X%f Y%f" %(L,L), file = f)
    print("G1 F2700 E%f"%E,file = f )
         

