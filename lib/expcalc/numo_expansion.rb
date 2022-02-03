#require 'nmatrix'
#require 'nmatrix/lapacke'
require 'numo/narray'
require 'numo/linalg'
require 'cmath'
require 'time'
require 'npy'
#require 'pp'

# Tb meter leer matrices con npy
class Hash
	def get_hash_values_idx
	  x_names_indx = {}
	  i = 0
	  self.each do |k, values|
	    values.each do |val_id|
	      query = x_names_indx[val_id]
	      if query.nil?
	        x_names_indx[val_id] = i
	        i += 1
	      end
	    end
	  end
	  return x_names_indx
	end

	def to_bmatrix 
	  x_names_indx = self.get_hash_values_idx
	  y_names = self.keys
	  x_names = x_names_indx.keys
	   # row (y), cols (x)
	  matrix = Numo::DFloat.zeros(self.length, x_names.length)
	  i = 0
	  self.each do |id, items|
	    items.each do |item_id|
	      matrix[i, x_names_indx[item_id]] = 1
	    end
	    i += 1
	  end
	  return matrix, y_names, x_names
	end


	# TODO: Only works if the resultin matrix will be squared. Replace implementacion taking into account to_bmatrix and its output
	def to_wmatrix
	  element_names = self.keys
	  matrix = Numo::DFloat.zeros(element_names.length, element_names.length)
	  i = 0
	  self.each do |elementA, relations|
	    element_names.each_with_index do |elementB, j|
	      if elementA != elementB
	        query = relations[elementB]
	        if !query.nil?
	          matrix[i, j] = query
	        else
	          matrix[i, j] = self[elementB][elementA]
	        end
	      end
	    end
	    i += 1
	  end
	  return matrix, element_names
	end
end

module Numo
	class NArray
		def save(matrix_filename, x_axis_names, x_axis_file, y_axis_names=nil, y_axis_file=nil)
		  File.open(x_axis_file, 'w'){|f| f.print x_axis_names.join("\n") }
		  File.open(y_axis_file, 'w'){|f| f.print y_axis_names.join("\n") } if !y_axis_names.nil?
		  Npy.save(matrix_filename, self)
		end


		def div(second_mat) #Matrix division A/B => A.dot(B.pinv) #https://stackoverflow.com/questions/49225693/matlab-matrix-division-into-python
			return self.dot(second_mat.pinv)
		end

		def div_by_vector(vector, by=:col)
			new_matrix =  self.new_zeros
			if by == :col
				self.shape.last.times do |n|
					vector.each_with_indices do |val, i, j|
						new_matrix[i, n] = self[i, n].fdiv(val)
					end
				end
			elsif by == :row

			end
			return new_matrix
		end

		def frobenius_norm
			fro = 0.0
			self.each do |value|
				fro += value.abs ** 2
			end
			return fro ** 0.5
		end

		def  max_norm #https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html, ord parameter = 1
			sums = self.abs.sum(1)
			return sums.max[0, 0]
		end

		def vector_product(vec_b)
			product = 0.0
			self.each_with_indices do |val, i, j|
				product += val * vec_b[i, j]
			end
			return product
		end

		def vector_self_product
			product = 0.0
			self.each_stored_with_indices do |val, i, j|
				product +=  val ** 2
			end
			return product
		end

		def max_eigenvalue(n=100, error = 10e-15) # do not set error too low or the eigenvalue cannot stabilised around the real one
			max_eigenvalue = 0.0
			length = self.shape.last
			v = Numo::DFloat.new(length).rand
			# http://web.mit.edu/18.06/www/Spring17/Power-Method.pdf 
			last_max_eigenvalue = nil
			n.times do
				v = Numo::Linalg.dot(self, v) 
				v = v / Numo::Linalg.norm(v) 
				max_eigenvalue = Numo::Linalg.dot(v, Numo::Linalg.dot(self, v)) / Numo::Linalg.dot(v,v) #Rayleigh quotient
				break if !last_max_eigenvalue.nil? && (last_max_eigenvalue - max_eigenvalue).abs <= error
				last_max_eigenvalue = max_eigenvalue
			end
			return max_eigenvalue
		end

		def min_eigenvalue(n=100, error = 10e-12)
			return  Numo::Linalg.inv(self).max_eigenvalue(n, error)
		end

		def expm
			return compute_py_method{|mat| expm(mat)}
			#return compute_py_method(self){|mat| expm(mat)}
			##################################################
			# matlab pade aproximation 
			################################################
			### toolbox/matlab/demos/expmdemo1.m (Golub and Van Loan, Matrix Computations, Algorithm 11.3-1.)		
			
			#fraction, exponent = Math.frexp(max_norm)
			#s = [0, exponent+1].max
			#a = self/2**s

			## Pade approximation for exp(A)
			#x = a
			#c = 0.5
			#ac = a*c
			#e = NMatrix.identity(a.shape, dtype: a.dtype) + ac
			#d = NMatrix.identity(a.shape, dtype: a.dtype) - ac
			#q = 6
			#p = true
			#(2..q).each do |k|
			#	c = c * (q-k+1) / (k*(2*q-k+1))
			#	x = a.dot(x)
			#	cX =  x * c
			#	e = e + cX
			#	if p
			#		d = d + cX
			#	else
			#		d = d - cX
			#	end
			#	p = !p
			#end
			#e = d.solve(e) #solve

			## Undo scaling by repeated squaring
			#(1..s).each do
			#	e = e.dot(e) 
			#end
			#return e

			###################################
			## Old python Pade aproximation
			###################################
			#### Pade aproximation: https://github.com/rngantner/Pade_PyCpp/blob/master/src/expm.py
			#a_l1 = max_norm
			#n_squarings = 0
			#if self.dtype == :float64 || self.dtype == :complex128
			#	if a_l1 < 1.495585217958292e-002
			#		u,v = _pade3(self)
		        #elsif a_l1 < 2.539398330063230e-001
			#		u,v = _pade5(self)
		        #elsif a_l1 < 9.504178996162932e-001
			#		u,v = _pade7(self)
		        #elsif a_l1 < 2.097847961257068e+000
			#		u,v = _pade9(self)
			#	else
			#		maxnorm = 5.371920351148152
			#		n_squarings = [0, Math.log2(a_l1 / maxnorm).ceil].max
			#		mat = self / 2**n_squarings
			#		u,v = _pade13(mat)
			#	end
			#elsif self.dtype == :float32 || self.dtype == :complex64
			#	if a_l1 < 4.258730016922831e-001
			#		u,v = _pade3(self)
			#    elsif a_l1 < 1.880152677804762e+000
			#		u,v = _pade5(self)
			#	else
			#		maxnorm = 3.925724783138660
			#		n_squarings = [0, Math.log2(a_l1 / maxnorm).ceil].max
			#		mat = self / 2**n_squarings
			#		u,v = _pade7(mat)
			#	end
			#end
			#p = u + v
			#q = -u + v
			#r = q.solve(p)
			#n_squarings.times do
			#	r = r.dot(r)
			#end
			#return r

			######################
			# Exact computing
			######################
			#####expm(matrix) = V*diag(exp(diag(D)))/V; V => eigenvectors(right), D => eigenvalues (right). # https://es.mathworks.com/help/matlab/ref/expm.html
			#eigenvalues, eigenvectors = NMatrix::LAPACK.geev(self, :right)
			#eigenvalues.map!{|val| Math.exp(val)}
			#numerator = eigenvectors.dot(NMatrix.diagonal(eigenvalues, dtype: self.dtype))
			#matrix_exp = numerator.div(eigenvectors)
			#return matrix_exp
		end 

		def cosine_normalization
			normalized_matrix =  NMatrix.zeros(self.shape, dtype: self.dtype)
			#normalized_matrix =  NMatrix.zeros(self.shape, dtype: :complex64)
			self.each_with_indices do |val, i, j|
				norm = val/CMath.sqrt(self[i, i] * self[j,j])
				#abort("#{norm} has non zero imaginary part" ) if norm.imag != 0
				normalized_matrix[i, j] = norm#.real
			end
			return normalized_matrix
		end
		

		private
		def copy_array_like(ary1, ary2)
			length = ary2.shape[0] #ruby numo::array
			length.times do |i|
				length.times do |j|
					ary2[i,j] = ary1[i,j] #python array
				end
			end
		end

		def compute_py_method
			require 'pycall/import'
			self.class.class_eval do # To include Pycall into Numo::array
				include PyCall::Import
			end
			#Python
			pyfrom 'scipy.linalg', import: :expm
			pyimport :numpy, as: :np
		
			b = np.empty(self.shape)
			start = Time.now
			copy_array_like(self, b)
			STDERR.puts "Time cpy rb in py array #{(Time.now - start)/60}"		

			a = nil
			start = Time.now
			PyCall.without_gvl do
				a = yield(b) # Code block from ruby with python code
				#a = expm(b)
			end
			##
			STDERR.puts "Time computing #{(Time.now - start)/60}"		

			start = Time.now
			result_matrix =  self.new_zeros
			copy_array_like(a, result_matrix)
			STDERR.puts "Time cpy py in rb array #{(Time.now - start)/60}"
			return result_matrix
		end

		def _pade3(a)
			b = [120.0, 60.0, 12.0, 1.0]
			a2 = a.dot(a)
			ident = NMatrix.identity(a.shape, dtype: a.dtype)
			u = a.dot(a2 * b[3] + ident * b[1])
			v = a2 * b[2] + ident * b[0]
			return u,v 
		end

		def _pade5(a)
			b = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0]
			a2 = a.dot(a)
			a4 = a2.dot(a2)
			ident = NMatrix.identity(a.shape, dtype: a.dtype)
			u = a.dot(a4 * b[5] + a2 * b[3] + ident * b[1])
			v = a4 * b[4] + a2 * b[2] + ident * b[0]
			return u,v 
		end

		def _pade7(a)
			b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0]
			a2 = a.dot(a)
			a4 = a2.dot(a2)
			a6 = a4.dot(a2)	
			ident = NMatrix.identity(a.shape, dtype: a.dtype)
			u = a.dot(a6 * b[7] + a4 * b[5] + a2 * b[3] + ident * b[1])
			v = a6 * b[6] + a4 * b[4] + a2 * b[2] + ident * b[0]
			return u,v 
		end

		def _pade9(a)
			b = [17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
	2162160.0, 110880.0, 3960.0, 90.0, 1.0]
			a2 = a.dot(a)
			a4 = a2.dot(a2)
			a6 = a4.dot(a2)	
			a8 = a6.dot(a2)	
			ident = NMatrix.identity(a.shape, dtype: a.dtype)
			u = a.dot(a8 * b[9] + a6 * b[7] + a4 * b[5] + a2 * b[3] + ident * b[1])
			v = a8 * b[8] + a6 * b[6] + a4 * b[4] + a2 * b[2] + ident * b[0]
			return u,v 
		end

		def _pade13(a)
			b = [64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
				1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
				33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0]
			a2 = a.dot(a)
			a4 = a2.dot(a2)
			a6 = a4.dot(a2)	
			ident = NMatrix.identity(a.shape, dtype: a.dtype)
			submat = a6 * b[13] + a4 * b[11] + a2 * b[9]
			u = a.dot(a6.dot(submat) + a6 * b[7] + a4 * b[5] + a2 * b[3] + ident * b[1])
			v = a6.dot(a6 * b[12] + a4 * b[10] + a2 * b[8] ) + a6 * b[6] + a4 * b[4] + a2 * b[2] + ident * b[0]
			return u,v 
		end
	end
end
