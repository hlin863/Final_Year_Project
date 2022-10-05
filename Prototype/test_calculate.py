"""

A file to run unit tests on the calculator.py file.

"""


import calculate


class Test:

    def test_add(self):
        """Test the add function"""
        assert calculate.add(2, 3) == 5

    def test_subtract(self):
        """Test the subtract function"""
        assert calculate.subtract(5, 3) == 2

    def test_multiply(self):
        """Test the multiply function"""
        assert calculate.multiply(5, 3) == 15

    def test_divide(self):
        """Test the divide function"""
        assert calculate.divide(6, 3) == 2

    def test_power(self):
        """Test the power function"""
        assert calculate.power(2, 3) == 8
