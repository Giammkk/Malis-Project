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
    
    def plot_w(self, title='Weights'): #method to plot w
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
        w = self.w
        num_samples = A.shape[0]
        self.err = np.linalg.norm((yhat - y)**2)/num_samples
        
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