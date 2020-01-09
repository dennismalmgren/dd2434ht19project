#Todo: Andree will implement SVM

"""
Module containing the implementation of svm classifier.
"""
import gin
from kernels import ClusterKernel

#Dependencies for SVM
import numpy as np
from scipy.optimize import minimize

import sklearn

@gin.configurable
class SVM(object):
    """Support vector machine with various kernels."""

    def __init__(self, kernel_name='linear_kernel'):
        super(SVM, self).__init__()
        if kernel_name == 'linear_kernel':
            self.kernel = self.linear_kernel
        elif kernel_name == 'poly_kernel':
            self.kernel = self.poly_kernel
        elif kernel_name == 'rbf_kernel':
            self.kernel = self.rbf_kernel
     
        self.test_data_in = False
        self.test_data_out = False
        self.training_data_in = False
        self.training_data_out = False
        self.solution_found = False


    def linear_kernel(self, x, y):
        #Linear kernel
        return np.dot(x,y)

    def poly_kernel(self, x,y):
        #Polinomial kernel
        P=3
        return (np.dot(x,y)+1)**P

    def rbf_kernel(self, x,y):
        #Radial basis function kernel. 
        sigma = 0.2
        return np.exp(-((np.linalg.norm(x-y))**2)/(2*(sigma**2)))

    def set_kernel(self, function):
        #Give the SVM a custom kernel
        self.kernel = function

    def give_test_data(self, test_data_in, test_data_out):
        
        #Check for existing data, add accordingly
        if self.test_data_in:
            self.test_data_in 	= np.concatenate((self.test_data_in, test_data_in))
        else:
            self.test_data_in 	= test_data_in

        #Check for existing data, add accordingly
        if self.test_data_out:
            self.test_data_out 	= np.concatenate((self.test_data_out, test_data_out))
        else:
            self.test_data_out 	= test_data_out

    def give_training_data(self, training_data_in, training_data_out):
        #Check that data is the same length
        if len(training_data_in) != len(training_data_out):
            print("Data in and Data out have different lengths.")
        
        #Check for existing data, add accordingly
        if self.training_data_in:
            self.training_data_in 	= np.concatenate((self.training_data_in, training_data_in))
        else:
            self.training_data_in 	= training_data_in

        #Check for existing data, add accordingly
        if self.training_data_out:
            self.training_data_out 	= np.concatenate((self.training_data_out, training_data_out))
        else:
            self.training_data_out 	= training_data_out
        
    def train(self):
            
        #Save constants
        start = np.zeros(len(self.training_data_in))

        #Calculate P
        self.calculate_P()

        #Determine constraints
        C = 400
        B=[(0,C) for s in range(len(self.training_data_in))]
        XC={'type':'eq','fun':self.zerofun}

        #Minimize alpha
        ret = minimize(self.objective, start, bounds = B, constraints=XC)
        if ret['success']:
            alpha = ret['x']
            self.solution_found = True
            self.nonzero = self.extract(alpha)
            self.b = self.calculateB()
            #plot(classA,classB)

        else:
            print("No solution found.")
            
    def calculate_P(self):
        #Check if the length is correct
        #Pre-calculate P in order to optimise the calculations later
        P = np.outer(self.training_data_out, self.training_data_out).astype('float64')
        kmat = self.kernel(self.training_data_in, np.transpose(self.training_data_in))
        P = np.multiply(P, kmat)
        self.P = P

    def zerofun(self,alpha):
        #This was the culprit.
        return np.dot(alpha, self.training_data_out) 


    def objective(self, alpha):
        return 0.5*np.dot(alpha, np.dot(alpha, self.P)) - np.sum(alpha)

    def extract(self, alpha):
        #return [(alpha[i], x[i], y[i]) for i in range(len(x)) if abs(alpha[i]) > 10e-5]		#Changed - probably wrong
        temp = []
        for i in range(len(self.training_data_in)):
            if abs(alpha[i]) > 10e-5:
                temp.append([alpha[i],self.training_data_in[i],self.training_data_out[i],i])
            else:
                pass
        return np.array(temp)

    def calculateB(self):
        summa = 0
        for data in self.nonzero:
            summa += data[0] * data[2] * self.kernel(data[1], self.nonzero[0][1]) 
        return summa - self.nonzero[0][2]						#Changed according to equation 6

    def classify_point(self,index):
        if self.solution_found:
            summa = 0
            for i in range(len(self.nonzero)):
                summa += self.nonzero[i][0]*self.nonzero[i][2]*self.kernel(index,self.nonzero[i][3])
            indicator = summa- self.b
            classification = np.sign(indicator)
            if np.abs(indicator)>=1.0:
                confidence = 1.0
            else:
                confidence = 1-(np.abs(classification-indicator)/2)
            return classification, confidence
            
        else:
            print("No solution was found... sry ")

    def classify_dataset(self,data):
        if self.solution_found:
            output = np.zeros(shape=(len(data),2))

            for n, datapoint_index in enumerate(data):
                output[n,0] , output[n,1] = self.classify_point(datapoint_index)
            return output
            
        else:
            print("No solution was found... sry ")

    def analyze(self):
        results = self.classify_dataset(self.test_data_in)
        classifications = results[:,0]
        misclassification = np.abs(np.sum(classifications-self.test_data_out))/(2*len(classifications))
        print("Misclassification",misclassification)
        return misclassification

def rbf(x,y):
    sigma=0.4
    return np.exp(-((np.linalg.norm(x-y))**2)/(2*(sigma**2)))

def generate_data():
    #Generate data classes
    class_pos = np.random.randn(20,2)*0.4+[0.2, 0.5]
    class_neg = np.random.randn(20,2)*0.4+[-1,-0.5]

    #Combine classes into an input
    data_in = np.concatenate((class_pos , class_neg)) 

    #Catagorize classes
    data_out = np.concatenate( 
        (np.ones(len(class_pos)) , 
        -np.ones(len(class_neg)))) 

    N = len(data_in) # Number of rows (samples) 

    #Shuffle data
    permute=list(range(N)) 
    np.random.shuffle(permute) 
    data_in = data_in[permute , :] 
    data_out = data_out[permute] 

    #Return the results
    return [class_pos,class_neg, data_in, data_out]


if __name__ == '__main__':
    #Generate training data
    [_, _, training_data_in, training_data_out] = generate_data()
    #Generate test data
    [_, _, test_data_in, test_data_out] = generate_data()

    #Initiate SVM
    svm = SVM()
    svm.set_kernel(rbf)

    #Train the SVM
    svm.give_training_data(training_data_in, training_data_out)
    svm.train()

    #Test the SVM
    svm.give_test_data(test_data_in, test_data_out)
    svm.analyse()
    
