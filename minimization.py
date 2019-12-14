
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
    
    def plot_w(self, title=''): # method to plot w
        w = self.w
        n = np.arange(self.Nf)
        plt.figure() 
        plt.plot(n, w, '.--r')
        plt.xlabel('features')
        plt.ylabel('weight') 
        plt.title(title) 
        plt.grid() 
        #fig = plt.gcf() # get a reference to the current figure
        plt.show()
        #fig.savefig('w.png', dpi=100) # save the figure in a file
        return
    
    def computeError(self, yhat, y):
        dif = yhat - y
        mse = np.power(dif, 2)
        self.err = np.mean(mse)
        return self.err
    
    def accuracy(self, yhat, y):
        acc = 0
        for i in range(len(yhat)):
            a = self.classify(yhat.item(i))
            b = self.classify(y.item(i))
            
            if a == b:
                acc += 1
                
        return acc/len(yhat)
    
    def classify(self, n):
        if n <= 50:
            return 0
        if n <= 100:
            return 1
        if n <= 150:
            return 2
        else:
            return 4
     
    def estimate(self, mean, std):
        yhat = np.dot(self.x, self.w)
        yhat = yhat * std
        yhat = yhat + mean
        
        yhat = yhat + 0.5 
        return yhat.astype(int) # approx to integer
    
    def ploty(self, yhat, y):
        plt.figure() 
        plt.plot(range(len(y)), y, '.--r')
        plt.plot(range(len(yhat)), yhat, '.--b')
        plt.xlabel('features')
        plt.ylabel('weight') 
        plt.title('yhat (blue) vs y(red)') 
        plt.grid() 
        #fig = plt.gcf() # get a reference to the current figure
        plt.show()
        #fig.savefig('w.png', dpi=100) # save the figure in a file
        return
    
    
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