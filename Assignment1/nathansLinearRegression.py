class linearRegression:
	def __init__(self):
		self.training_input = np.matrix('0;0')
		self.training_output = np.matrix('0;0')
		self.testing_input = np.matrix('0;0')
		self.testing_output = np.matrix('0;0')
		self.weight_vector = np.matrix('0;0')
		self.loaded = 0

	def load(self, training_file, testing_file, adjustVal=1, randomValues=0):
		with open(training_file, 'r') as f:
			lines = [line.rstrip('\n').split() for line in f.readlines()]
		if adjustVal == 1:
			X_string = [['1'] + a[:-1] for a in lines] # The first elements with '1' on the start and ';' at the end
		else:
			X_string = [a[:-1] for a in lines]
		X_string = [np.array(a, dtype=float) for a in X_string]
		Y_string = [a[-1] for a in lines]  # The last element of each line
		Y_string = np.array(Y_string, dtype=float)
		self.training_input = np.matrix(X_string)
		self.training_output = np.matrix(Y_string).transpose()

		with open(testing_file, 'r') as f:
			lines = [line.rstrip('\n').split() for line in f.readlines()]
		if adjustVal == 1:
			X_string = [['1'] + a[:-1] for a in lines] # The first elements with '1' on the start and ';' at the end
		else:
			X_string = [a[:-1] for a in lines]
		X_string = [np.array(a, dtype=float) for a in X_string]
		Y_string = [a[-1] for a in lines]  # The last element of each line
		Y_string = np.array(Y_string, dtype=float)
		self.testing_input = np.matrix(X_string)
		self.testing_output = np.matrix(Y_string).transpose()
		self.loaded = 1

	def calc_weight_vector(self):
		# w = (XT*X)^-1XTY
		matrix1 = self.training_input.transpose().dot(self.training_input).getI()  # ( X^T * X ) ^-1
		self.weight_vector = matrix1.dot(self.training_input.transpose()).dot(self.training_output)  # * X^T * Y
		self.loaded = 2

	def avg_sq_error(self):
		if self.loaded != 2:
			print "Please load data [load(filename)] and calculate weight_vector first"

		# Calculate projected y value
		projected_values = self.training_input.dot(self.weight_vector)
		# Take the difference from real y value
		diff = projected_values - self.training_output
		diff = np.square(diff)
		# Average their squared distance
		MSE_train = diff.sum(0).item(0) / float(diff.shape[0])

		# Calculate projected y value
		projected_values = self.testing_input.dot(self.weight_vector)
		# Take the difference from real y value
		diff = projected_values - self.testing_output
		diff = np.square(diff)
		# Average their squared distance
		MSE_test = diff.sum(0).item(0) / float(diff.shape[0])
		print "AVERAGE SQUARED ERROR (Training File): " + str(MSE_train)
		print "AVERAGE SQUARED ERROR (Testing File): " + str(MSE_test)
