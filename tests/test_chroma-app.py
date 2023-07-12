# 2023-07-12, J. Köppern

import unittest

from chromadb_qa_bot.app.app import import_api_key

class TestFnAss(unittest.TestCase):
    def test_fn_ass(self):
        self.assertEqual(fn_ass(2, 3), 5)
        self.assertEqual(fn_ass(-1, 1), 0)
        self.assertEqual(fn_ass(0, 0), 0)

if __name__ == '__main__':
    unittest.main()
