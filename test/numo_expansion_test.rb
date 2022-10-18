#!/usr/bin/env ruby

ROOT_PATH = File.dirname(__FILE__)
require File.join(ROOT_PATH, 'test_helper.rb')

class NumoTest < Minitest::Test

	def setup
		@x_names = ['S1', 'S2', 'S3', 'S4']
		@y_names = ['M1', 'M2', 'M3', 'M4']
		@bmatrix_squared = Numo::NArray.load(File.join(ROOT_PATH, 'bmatrix_squared.txt'))
		@result_bmatrix_squared_to_hash = Numo::NArray.load(File.join(ROOT_PATH, 'result_bmatrix_squared_to_hash.txt'))

		@bmatrix_rectangular = Numo::NArray.load(File.join(ROOT_PATH, 'bmatrix_rectangular.txt'))
		@wmatrix_squared = Numo::NArray.load(File.join(ROOT_PATH, 'wmatrix_squared.txt'))
		@wmatrix_rectangular = Numo::NArray.load(File.join(ROOT_PATH, 'wmatrix_rectangular.txt'))
		@bhash_squared = {'S1' => ['S3'], 'S2' => ['S3', 'S4'], 'S3' => ['S1'], 'S4' => ['S2', 'S3']}
		@bhash_rectangular = {'S1' => ['M1', 'M2'], 'M1' => ['S1', 'S3'], 'M2' => ['S1', 'S2'], 'S2' => ['M2'], 'S3' => ['M1', 'M4'], 'M4' => ['S3', 'S4'], 'S4' => ['M3', 'M4'], 'M3' => ['S4']}
		@whash_squared = {'S1' => {'S2' => 3.0}, 'S2' => {'S1' => 3.0, 'S3' => 1.0, 'S4' => 2.0}, 'S3' => {'S2' => 1.0}, 'S4' => {'S2' => 2.0}}
		@whash_rectangular = {'M1' => {'S1'=> 3}, 'S1' => {'M1' => 3, 'M3' => 2}, 'M2' => {'S2' => 5, 'S3' => 1}, 'S2' => {'M2' => 5, 'M4' => 3}, 'M3' => {'S1' => 2, 'S3' => 4}, 'S3' => {'M2' => 1, 'M3' => 4}, 'M4' => {'S2' => 3, 'S4' => 2}, 'S4' => {'M4' => 2}}
	end


	# HASH TO MATRIX

	def test_to_bmatrix
		test_bmatrix_squared = @bhash_squared.to_bmatrix
		assert_equal(@result_bmatrix_squared_to_hash, test_bmatrix_squared.first)
	end

	def test_to_wmatrix_squared
		test_wmatrix_squared = @whash_squared.to_wmatrix(squared: true, symm: false)
		assert_equal(@wmatrix_squared, test_wmatrix_squared.first)
	end

	def test_to_wmatrix_rectangular
		test_wmatrix_rectangular = @whash_rectangular.to_wmatrix(squared: false, symm: false)
		assert_equal(@wmatrix_rectangular, test_wmatrix_rectangular.first)
	end


	# MATRIX TO HASH

	def test_bmatrix_squared_to_hash
		test_bmatrix_squared_to_hash = @bmatrix_squared.bmatrix_squared_to_hash(@x_names)
		assert_equal(@bhash_squared, test_bmatrix_squared_to_hash)
	end

	def test_bmatrix_rectangular_to_hash
		test_bmatrix_rectangular_to_hash = @bmatrix_rectangular.bmatrix_rectangular_to_hash(@x_names, @y_names)
		assert_equal(@bhash_rectangular, test_bmatrix_rectangular_to_hash)
	end

	def test_wmatrix_squared_to_hash
		test_wmatrix_squared_to_hash = @wmatrix_squared.wmatrix_squared_to_hash(@x_names)
		assert_equal(@whash_squared, test_wmatrix_squared_to_hash)
	end

	def test_wmatrix_rectangular_to_hash
		test_wmatrix_rectangular_to_hash = @wmatrix_rectangular.wmatrix_rectangular_to_hash(@y_names, @x_names)
		assert_equal(@whash_rectangular, test_wmatrix_rectangular_to_hash)
	end


end
