import os
import numpy as np

L = 40 #square size
d = 5 #line spacing
N = 1 #no. of layers
S = 100 #speed
filename = "gcodes/"+ "s-" +str(S) + "_d-" + str(d) + "_lay-" + str(N) + "_w-" + str(L)+"mm" + ".gcode"
Ys = np.arange(0,L,d)
Ys = np.flip(Ys)
X = 0
E = 0
odd = True

with open(filename, 'w') as f:
    header_file = open("templates/header.txt", "r")
    print(header_file.read(), file = f)
    # print speed
    print("G1 F%f E0" %S , file = f) 
    # Print horizontal
    for i in range(N):
        for Y in np.flip(Ys):
            if odd: 
                print("G1 X%f Y%f E%f" %(0, Y, E ),file = f)
                print("G1 X%f Y%f E%f" %(L, Y, E ),file = f)
                
                odd = False
            else:
                print("G1 X%f Y%f E%f" %(L, Y, E ),file = f)
                print("G1 X%f Y%f E%f" %(0, Y, E ),file = f)
                odd = True
            E+=L
    # Print vertical        
    for i in range(N):
        for X in np.flip(Ys):
            if odd: 
                print("G1 X%f Y%f E%f" %(X, 0, E ),file = f)
                print("G1 X%f Y%f E%f" %(X, L, E ),file = f)
                
                odd = False
            else:
                print("G1 X%f Y%f E%f" %(X, L, E ),file = f)
                print("G1 X%f Y%f E%f" %(X, 0, E ),file = f)
                odd = True
            E+=L
    

    # print("G0 F9000 X%f Y%f" %(L,L), file = f)
    # print("G1 F2700 E%f"%E,file = f )

    #alarm for end of filament printing

    alarm_file= open("templates/alarm.txt", "r")
    print(alarm_file.read(), file = f)



    #Print end
    ender_file= open("templates/ender.txt", "r")
    print(ender_file.read(), file = f)
         

