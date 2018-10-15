#this solves second order ODE's with linear coefficients arbitrary right hand side
import numpy as np
from matplotlib.pyplot import *
import scipy
from scipy.sparse import diags
def solve(co, Domain, N, BC, fun, ND):
        # co is the list of coefficients of the second order equation: co[0]y+co[1]y'+co[2]y''=fun
        # Domain is a list of two elements such that x(the independent variable) in (Domain[0],Domain[1])
        #N is the number of intervals to be considered. I.e. there are N+1 points, and N-1 inner points
        #BC is a list with the two values of the boundary conditions left to right
        #ND specifies the types of boundary conditions
        #If ND=0 then we have Diriclet BC, if ND=1 we have Neumann

        h=(Domain[-1]-Domain[0])/float(N) #grid spacing
        rhs=np.zeros(N+1)
        rhs[-1]=BC[-1]
        rhs[0]=BC[0]
        x=np.linspace(Domain[0],Domain[-1],N+1)
        
        for i in range(1,N):
            rhs[i]=fun(x[i])

        D2=1/h**2*diags([1, -2, 1], [-1, 0, 1], shape=(N+1, N+1)).toarray() #Differential operator for y''
        D1=1/(2*h)*diags([1, 0, -1], [-1, 0, 1], shape=(N+1, N+1)).toarray() #Differential operator for y'
        D0=diags([1], [0], shape=(N+1,N+1)).toarray() #identity operator - add y
        #Remove values at boundaries
        D2[0,:]=0 
        D2[-1,:]=0
        D1[0,:]=0
        D1[-1,:]=0
        D0[0,:]=0
        D0[-1,:]=0
        #-------------------------
        Operator=co[0]*D0+co[1]*D1+co[2]*D2
        #check BC type and assign
        if ND==0: #Diriclet
            Operator[0,0]=1
            Operator[-1,-1]=1
        elif ND==1: #Neumann
            Operator[0,0]=-3/2/h
            Operator[0,1]=2/h
            Operator[0,2]=-0.5/h
            Operator[-1,-1]=3/2/h
            Operator[-1,-2]=-2/h
            Operator[-1,-3]=0.5/h
        #-----------------------------

        #we now have that Operator*Solution=rhs
        #invert for the solution.
        y=np.linalg.solve(Operator,rhs)
        xy=np.concatenate((x,y),0)
        plot(x,y)
        savefig("resultofsolve.pdf")
        
        return xy
        
