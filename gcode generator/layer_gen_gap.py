import os
import numpy as np

L = 30 #square size
d = 0.24 #line spacing
gap= 5 #gap before going back
N = 1 #no. of layers
S = 50 #speed
filename = "gcodes/"+ "s-" +str(S) + "_d-" + str(d) + "_gap-" + str(gap) + "_lay-" + str(N) + "_w-" + str(L)+"mm" + ".gcode"
Ys = np.arange(0,L,gap)
ys =np.arange(0,gap,d)
Ys = np.flip(Ys)
X = 0
E = 0
odd = True
wash = False
with open(filename, 'w') as f:
    header_file = open("templates/header.txt", "r")
    print(header_file.read(), file = f)
    # print speed
    print("G1 F%f E0" %S , file = f) 
    for i in range(N):
        
        for y in ys:
            for Y in np.flip(Ys):
                Y = y+Y
                if odd: 
                    print("G1 X%f Y%f E%f" %(0, Y, E ),file = f)
                    print("G1 X%f Y%f E%f" %(L, Y, E ),file = f)
                    
                    odd = False
                else:
                    print("G1 X%f Y%f E%f" %(L, Y, E ),file = f)
                    print("G1 X%f Y%f E%f" %(0, Y, E ),file = f)
                    odd = True
                E+=L
    

    # print("G0 F9000 X%f Y%f" %(L,L), file = f)
    # print("G1 F2700 E%f"%E,file = f )

    #alarm for end of filament printing

    alarm_file= open("templates/alarm.txt", "r")
    print(alarm_file.read(), file = f)

    #Print washing
    if wash:
        print(";Continue washing", file = f)
        Wash_range = np.arange(L,L+10,d)
        for i in range(N):
            for W in Wash_range:
                if odd: 
                    print("G1 X%f Y%f E%f" %(0, W, E ),file = f)
                    print("G1 X%f Y%f E%f" %(L, W, E ),file = f)
                    
                    odd = False
                else:
                    print("G1 X%f Y%f E%f" %(L, W, E ),file = f)
                    print("G1 X%f Y%f E%f" %(0, W, E ),file = f)
                    odd = True
                E+=L


    #Print end
    ender_file= open("templates/ender.txt", "r")
    print(ender_file.read(), file = f)
         

