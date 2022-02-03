class Array

	def mean
		return self.inject(0){|sum, n | sum + n}.fdiv(self.length)
	end

	def standard_deviation
		x_mean = self.mean
		variance = self.inject(0){|sum, n | sum + (n - x_mean)**2 }.fdiv(self.length)
		return Math.sqrt(variance)
	end

end
