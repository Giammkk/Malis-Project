
import numpy as np
import matplotlib.pyplot as plt
class SolveMinProbl:
    def __init__(self, y, x): 
        self.x = x
        self.Np = y.shape[0] # number of rows
        self.Nf = x.shape[1] # number of columns
        
        self.y = y # column vector y
        self.w = np.zeros((self.Nf,1), dtype=float) #column vector w

        self.err = np.zeros((1,1), dtype=float)
        return
    
    def plot_w(self, title='Weights'): # method to plot w
        w = self.w
        n = np.arange(self.Nf)
        plt.figure() 
        plt.plot(n, w, '.--r')
        plt.xlabel('features')
        plt.ylabel('w') 
        plt.title(title) 
        plt.grid() 
        #fig = plt.gcf() # get a reference to the current figure
        plt.show()
        #fig.savefig('w.png', dpi=100) # save the figure in a file
        return
    
    def computeError(self, yhat, y):
        A = self.x
        
        dif = yhat - y
        mse = np.power(dif, 2)
        self.err = np.mean(mse)
        
        return self.err
        
    def estimate(self, mean, std):
        yhat = np.dot(self.x, self.w)
        yhat = yhat * std
        yhat = yhat + mean
        
        yhat = yhat + 0.5 
        return yhat.astype(int) # approx to integer
    
    
class LLS(SolveMinProbl):
    
    def run(self): 
        """
        Method that finds w that minimize ||y-Xw||^2
        """
        A = self.x
        y = self.y
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y) # w=(A*AT)^-1*AT*y
        self.w = w


class conjugateGrad(SolveMinProbl):
    """
    Implementation of the Conjugate Gradient algorithm
    """
    def run(self):
        A = self.x
        Q = np.dot(A.T,A) # Q=A^T*A
        y=self.y
        b = (np.dot(A.T,y)) # b=A*y
        w = np.zeros((self.Nf,1), dtype=float)
        g = -b
        d = b
        
        for it in range(self.Nf):
            alfa = -np.dot(d.T,g)/np.dot(np.dot(d.T,Q),d) # alfa=d^T*b/(d^T*Q*d)
            
            w = w + alfa*d
            g = g + alfa*np.dot(Q,d) # g=g+alfa*Q*d
            
            # beta = g*Q*d/(d^T*Q*d)
            beta = np.dot(np.dot(g.T,Q),d)/np.dot(np.dot(d.T,Q),d)
            
            d = -g+beta*d
            
        self.w = w