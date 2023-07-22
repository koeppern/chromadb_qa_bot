# 2023-07-12, J. Köppern

import unittest

from chromadb_qa_bot.app.app import load_api_key
from chromadb_qa_bot.app.app import square_input

class TestSquareInput(unittest.TestCase):
    def test_positive_numbers(self):
        self.assertEqual(square_input(2), 4)
        self.assertEqual(square_input(5), 25)
        self.assertEqual(square_input(10), 100)

    def test_negative_numbers(self):
        self.assertEqual(square_input(-2), 4)
        self.assertEqual(square_input(-5), 25)
        self.assertEqual(square_input(-10), 100)

    def test_zero(self):
        self.assertEqual(square_input(0), 0)

    def test_decimal_numbers(self):
        self.assertEqual(square_input(1.5), 2.25)
        self.assertEqual(square_input(2.5), 6.25)
        self.assertEqual(square_input(-1.5), 2.25)
        self.assertEqual(square_input(-2.5), 6.25)

    def test_large_numbers(self):
        self.assertEqual(square_input(1000000), 1000000000000)
        self.assertEqual(square_input(999999999), 999999998000000001)
        
# class TestFnAss(unittest.TestCase):
#     def test_fn_ass(self):
#         self.assertEqual(fn_ass(2, 3), 5)
#         self.assertEqual(fn_ass(-1, 1), 0)
#         self.assertEqual(fn_ass(0, 0), 0)

if __name__ == '__main__':
    unittest.main()
