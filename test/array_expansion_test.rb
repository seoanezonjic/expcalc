#!/usr/bin/env ruby

ROOT_PATH = File.dirname(__FILE__)
require File.join(ROOT_PATH, 'test_helper.rb')

class ArrayTest < Minitest::Test

	def setup
		@data = [1,2,3,7,2,1,4,9]
	end

	# Statisticas from array
	def test_mean
		assert_equal(3.625 , @data.mean.round(3))
	end

	def test_variance
		assert_equal(8.554, @data.variance.round(3))
	end

	def test_standard_deviation
		assert_equal(2.925 ,@data.standard_deviation.round(3))
	end

	def test_get_quantiles
		assert_equal(1.50 ,@data.get_quantiles(0.25).round(2))
		assert_equal(2.50 ,@data.get_quantiles(0.5).round(2))
		assert_equal(5.50 ,@data.get_quantiles(0.75).round(2))
	end

end