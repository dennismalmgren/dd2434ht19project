#Dependencies for SVM
import numpy as np
from scipy.optimize import minimize

class SVM(object):
	"""Support vector machine with various kernels."""

	def __init__(self):
		super(SVM,self).__init__()
		self.kernel 	= self.rbf_kernel
		self.data_in 	= False
		self.data_out 	= False
		self.dataset1 	= False
		self.dataset2 	= False
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
		if len(data_in) != len(data_out):
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
		start = np.zeros(len(self.data_in))

		#Calculate P
		self.calculate_P()

		#Determine constraints
		C = 400
		B=[(0,C) for s in range(len(self.data_in))]
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
				P[i][j]=self.data_out[i]*self.data_out[j]*rbf(self.data_in[i],self.data_in[j])
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
		    summa += data[0] * data[2] * rbf(data[1], self.nonzero[0][1]) 
		return summa - self.nonzero[0][2]						#Changed according to equation 6

	def classify_point(self,datapoint):
		if self.solution_found:
			summa = 0
			for i in range(len(self.nonzero)):
				#summa += alpha[i]*y[i]*kernel(np.array([pointA,pointB]),x[i])+1					#Old 
				summa += self.nonzero[i][0]*self.nonzero[i][2]*rbf(datapoint,self.nonzero[i][1])
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
	def analyze(self):
		results = self.classify_dataset(self.dataset1)
		classifications = results[:,0]
		misclassification = np.sum(np.abs(classifications-self.dataset2))/(2*len(classifications))
		print("Misclassification",misclassification)
		return misclassification

def rbf(x,y):
	sigma=0.55
	return np.exp(-((np.linalg.norm(x-y))**2)/(2*(sigma**2)))
