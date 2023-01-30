class Array

	def mean
		return self.inject(0){|sum, n | sum + n}.fdiv(self.length)
	end

	def standard_deviation
		x_mean = self.mean
		variance = self.inject(0){|sum, n | sum + (n - x_mean)**2 }.fdiv(self.length)
		return Math.sqrt(variance)
	end
	
	def get_quantiles(position=0.5, increasing_sort= false)
      self.sort!
      self.reverse! if !increasing_sort
      quantile_value = nil

      n_items = self.size
      quantile_coor = (n_items - 1) * position
      f_qcoor = quantile_coor.floor
      c_qcoor = quantile_coor.ceil
      if f_qcoor == c_qcoor
        quantile_value = array[f_qcoor.to_i]
      else
        quantile_value = (array[f_qcoor.to_i] + array[c_qcoor.to_i]).fdiv(2)
      end
      return quantile_value
    end
end
