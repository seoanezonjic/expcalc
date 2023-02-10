class Array

	def mean
		return self.inject(0){|sum, n | sum + n}.fdiv(self.length)
	end

  def variance(population=false)
    x_mean = self.mean
    size = self.length
    size -= 1 if !population
    variance = self.inject(0){|sum, n | sum + (n - x_mean)**2 }.fdiv(size)
    return variance
  end

	def standard_deviation(population = false)
		return Math.sqrt(self.variance(population))
	end
	
	def get_quantiles(position=0.5, decreasing_sort = false) 
      sorted_array = self.sort
      sorted_array.reverse! if decreasing_sort
      quantile_value = nil

      n_items = self.length
      quantile_coor = (n_items - 1) * position
      f_qcoor = quantile_coor.floor.to_i
      c_qcoor = quantile_coor.ceil.to_i
      if f_qcoor == c_qcoor
        quantile_value = sorted_array[f_qcoor]
      else
        quantile_value = (sorted_array[f_qcoor] + sorted_array[c_qcoor]).fdiv(2)
      end
      return quantile_value
    end
end
