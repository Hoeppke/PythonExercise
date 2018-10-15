# this solves second order ODE's with linear coefficients arbitrary right hand
# side
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import Fun


def solve(co, Domain, N, BC, fun, ND):
    """: Solves a 1D boundary value problem using finite difference method.
    :co: is the list of coefficients of the second order equation:
        co[0]y+co[1]y'+co[2]y''=fun :Domain: is a list of two elements such
        that x(the independent variable) in (Domain[0],Domain[1])
    :N: is the number of intervals to be considered. I.e. there are N+1 points,
        and N-1 inner points
    :BC: is a list with the two values of the boundary conditions left to right
    :ND: specifies the types of boundary conditions If ND=0 then we have
         Diriclet BC, if ND=1 we have Neumann.
    """
    h = (Domain[-1]-Domain[0])/float(N)  # grid spacing
    rhs = np.zeros(N+1)
    rhs[-1] = BC[-1]
    rhs[0] = BC[0]
    x = np.linspace(Domain[0], Domain[-1], N+1)
    for i in range(1, N):
        rhs[i] = fun(x[i])

    # Differential operator for y''
    D2 = 1/h**2*diags([1, -2, 1], [-1, 0, 1], shape=(N+1, N+1)).toarray()
    # Differential operator for y'
    D1 = 1/(2*h)*diags([1, 0, -1], [-1, 0, 1], shape=(N+1, N+1)).toarray()
    # identity operator - add y
    D0 = diags([1], [0], shape=(N+1, N+1)).toarray()
    # Remove values at boundaries
    D2[0, :] = 0
    D2[-1, :] = 0
    D1[0, :] = 0
    D1[-1, :] = 0
    D0[0, :] = 0
    D0[-1, :] = 0
    # -------------------------
    operator = co[0]*D0+co[1]*D1+co[2]*D2
    # check BC type and assign
    if ND.lower() == "dirichlet":  # Dirichlet BC
        operator[0, 0] = 1
        operator[-1, -1] = 1
    elif ND.lower() == "neumann":  # Neumann BC
        operator[0, 0] = -3/2/h
        operator[0, 1] = 2/h
        operator[0, 2] = -0.5/h
        operator[-1, -1] = 3/2/h
        operator[-1, -2] = -2/h
        operator[-1, -3] = 0.5/h
    else:
        # Default to using D'let bc.
        operator[0, 0] = 1
        operator[-1, -1] = 1

    # we now have that Operator*Solution=rhs
    # invert for the solution.
    y = np.linalg.solve(operator, rhs)
    xy = np.concatenate((x, y), 0)
    plt.plot(x, y)
    plt.savefig("resultofsolve.pdf")
    plt.show()

    return xy


if __name__ == "__main__":
    # Main function that is executed when python3 DiffSolve.py is called.

    # Define the coefficients used to solve the first example equation.
    coefs = [0, 0, 1]

    # Define the domain by left and right boundary
    dom = [0, 1]

    # Define the boundary conditions
    bc = [0, 0.1]

    # Chose the type of boundary condition to be dirichlet:
    bc_type = "dirichlet"

    # Choose the number of grid points:
    N = 256

    # Choose the function for the RHS:
    fun = Fun.funct

    solve(coefs, dom, N, bc, fun, bc_type)
