#Todo: Andree will implement SVM

"""
Module containing the implementation of svm classifier.
"""
import gin
from kernels import ClusterKernel
from utils import LINEAR

#Dependencies for SVM
import numpy as np
from scipy.optimize import minimize


@gin.configurable
class SVM(object):
    """Support vector machine with various kernels."""

    def __init__(self, kernel_name='linear_kernel'):
        super(SVM, self).__init__()
        if kernel_name == 'linear_kernel':
            self.kernel = self.linear_kernel
        elif kernel_name == 'poly_kernel':
            self.kernel = self.linear_kernel
        elif kernel_name == 'rbf_kernel':
            self.kernel = self.rbf_kernel
        else:
            self.kernel = self.linear_kernel
            
        self.data_in = False
        self.data_out = False
        self.dataset1 = False
        self.dataset2 = False
        self.solution_found = False


    def linear_kernel(self, x, y):
        #Linear kernel
        return np.dot(x,y)

    def poly_kernel(x,y):
        #Polinomial kernel
        P=3
        return (np.dot(x,y)+1)**P

    def rbf_kernel(x,y):
        #Radial basis function kernel. 
        sigma = 0.2
        return np.exp(-((np.linalg.norm(x-y))**2)/(2*(sigma**2)))

    def set_kernel(self, function):
        #Give the SVM a custom kernel
        self.kernel = function

    def give_test_data(self, dataset1, dataset2):
        
        #Check for existing data, add accordingly
        if self.dataset1:
            self.dataset1 	= np.concatenate((self.dataset1, dataset1))
        else:
            self.dataset1 	= dataset1

        #Check for existing data, add accordingly
        if self.dataset2:
            self.dataset2 	= np.concatenate((self.dataset2, dataset2))
        else:
            self.dataset2 	= dataset2

    def give_training_data(self, data_in, data_out):
        #Check that data is the same length
        if data_in.shape[0] != data_out.shape[0]:
            print("Data in and Data out have different lengths.")
        
        #Check for existing data, add accordingly
        if self.data_in:
            self.data_in 	= np.concatenate((self.data_in, data_in))
        else:
            self.data_in 	= data_in

        #Check for existing data, add accordingly
        if self.data_out:
            self.data_out 	= np.concatenate((self.data_out, data_out))
        else:
            self.data_out 	= data_out
        
    def train(self):
            
        #Save constants
        start = np.zeros(self.data_in.shape[0])

        #Calculate P
        self.calculate_P()

        #Determine constraints
        C = 400
        B=[(0,C) for s in range(len(self.data_in.shape[0]))]
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
        P=np.zeros(shape=(len(self.data_in),len(self.data_in)))
        for i in range(len(self.data_in)):
            for j in range(len(self.data_in)):
                P[i][j]=self.data_out[i]*self.data_out[j]*self.kernel(self.data_in[i],self.data_in[j])
        self.P = P

    def zerofun(self,alpha):
        #This was the culprit.
        return np.dot(alpha, self.data_out) 


    def objective(self, alpha):
        return 0.5*np.dot(alpha, np.dot(alpha, self.P)) - np.sum(alpha)

    def extract(self, alpha):
        #return [(alpha[i], x[i], y[i]) for i in range(len(x)) if abs(alpha[i]) > 10e-5]		#Changed - probably wrong
        temp = []
        for i in range(len(self.data_in)):
            if abs(alpha[i]) > 10e-5:
                temp.append([alpha[i],self.data_in[i],self.data_out[i]])
            else:
                pass
        return np.array(temp)

    def calculateB(self):
        summa = 0
        for data in self.nonzero:
            summa += data[0] * data[2] * self.kernel(data[1], self.nonzero[0][1]) 
        return summa - self.nonzero[0][2]						#Changed according to equation 6

    def classify_point(self,datapoint):
        if self.solution_found:
            summa = 0
            for i in range(len(self.nonzero)):
                #summa += alpha[i]*y[i]*kernel(np.array([pointA,pointB]),x[i])+1					#Old 
                summa += self.nonzero[i][0]*self.nonzero[i][2]*self.kernel(datapoint,self.nonzero[i][1])
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
            for n, datapoint in enumerate(data):
                output[n,0] , output[n,1] = self.classify_point(datapoint)
            return output
            
        else:
            print("No solution was found... sry ")

    def analyse(self):
        result_of_class_1 = self.classify_dataset(self.dataset1)
        result_of_class_2 = self.classify_dataset(self.dataset2)

        avg_confidence_1 = np.sum(result_of_class_1[:,1])/result_of_class_1.shape[0]
        avg_confidence_2 = np.sum(result_of_class_2[:,1])/result_of_class_2.shape[0]

        print("Analysis complete.")
        print("Class 1 has the average classification confidence of", avg_confidence_1)
        print("Class 2 has the average classification confidence of", avg_confidence_2)

        print("\nFor a more detailed description, call the function classify_dataset")

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
        (np.ones(class_pos.shape[0]) , 
        -np.ones(class_neg.shape[0]))) 

    N = data_in.shape[0] # Number of rows (samples) 

    #Shuffle data
    permute=list(range(N)) 
    np.random.shuffle(permute) 
    data_in = data_in[permute , :] 
    data_out = data_out[permute] 

    #Return the results
    return [class_pos,class_neg, data_in, data_out]


if __name__ == '__main__':
    #Generate training data
    [nothing1, nothing2, data_in, data_out] = generate_data()
    #Generate test data
    [class_pos, class_neg, nothing1, nothing2] = generate_data()

    #Initiate SVM
    svm = SVM()
    svm.set_kernel(rbf)

    #Train the SVM
    svm.give_training_data(data_in, data_out)
    svm.train()

    #Test the SVM
    svm.give_test_data(class_pos, class_neg)
    svm.analyse()
    
