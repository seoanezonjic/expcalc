class Array

	def mean
		return self.inject(0){|sum, n | sum + n}.fdiv(self.length)
	end

	def standard_deviation
		x_mean = self.mean
		variance = self.inject(0){|sum, n | sum + (n - x_mean)**2 }.fdiv(self.length)
		return Math.sqrt(variance)
	end

	def get_quantiles(position=0.5)
	  self.sort!
	  n_items = self.size
	  quantile_coor = n_items * position - 1
	  if n_items % 2 == 0
	    quantile_value = (self[quantile_coor.to_i] + self[quantile_coor.to_i + 1]).fdiv(2)   
	  else
	    quantile_value = self[quantile_coor.ceil]
	  end
	  return quantile_value
	end
end
