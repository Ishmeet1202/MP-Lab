import numpy as np
import matplotlib.pyplot as plt

# def func(y,x):
#     derivative = np.zeros(2)
#     derivative[0] = y[1]
#     derivative[1] = (-9.8/1)*y[0]
#     return derivative

def func(y,x):
    derivative = np.zeros(3)
    derivative[0] = y[2]
    derivative[1] = y[1]
    derivative[2] = -2*y[0]
    return derivative
    
def rk4(y,x,n):
    h = (t[-1] - t[0])/len(x)
    y_arr = []
    y_in = y
    x_in = [x[0]]
    
    for i in range(len(x)):
        y_arr.append(y_in)
        
        k1 = [h * ele for ele in func(y_in,x_in)]
        
        yn = [e1 + e2/2 for e1,e2 in zip(y_in,k1)]
        xn = [e1 + h/2 for e1 in x_in]
        k2 = [h * ele for ele in func(yn,xn)]
        
        yn = [e1 + e2/2 for e1,e2 in zip(y_in,k2)]
        xn = [e1 + h/2 for e1 in x_in]
        k3 = [h * ele for ele in func(yn,xn)]
        
        yn = [e1 + e2 for e1,e2 in zip(y_in,k2)]
        xn = [e1 + h for e1 in x_in]
        k4 = [h * ele for ele in func(yn,xn)]
        
        yf = [initial_y + (e1 + 2 * (e2 + e3) + e4) / 6 for (initial_y,e1,e2,e3,e4) in zip(y_in, k1, k2, k3, k4)]
        
        y_in = yf
        #x_in = [e1 + h/2 for e1 in xn]

    y_arr=np.array(y_arr).reshape(-1,n)

    return(y_arr)
        
        
    
if __name__ == "__main__":
    yo = [2,0,1]
    t = np.linspace(0,20,500)
    sol = rk4(yo,t,3)

    plt.plot(t,sol[:,0])
    plt.plot(t,sol[:,1])
    plt.plot(t,sol[:,2])
    plt.show()