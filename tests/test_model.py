import unittest

import numpy as np

from src.model import NaiveBayes
from src.preprocessing import tokenize


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.model = NaiveBayes().fit(
            [["good", "food"], ["good"], ["bad", "food"]], [5, 5, 1]
        )

    def test_hand_calculation(self):
        # Vocabulary: bad, food, good. Class 1: 2 tokens; class 5: 3 tokens.
        np.testing.assert_allclose(np.exp(self.model.log_prior), [1 / 3, 2 / 3])
        np.testing.assert_allclose(np.exp(self.model.log_likelihood),
                                   [[2 / 5, 2 / 5, 1 / 5], [1 / 6, 2 / 6, 3 / 6]])
        expected = np.log([1 / 3, 2 / 3]) + 2 * np.log([1 / 5, 3 / 6])
        np.testing.assert_allclose(self.model.scores([["good", "good"]])[0], expected)
        self.assertEqual(self.model.predict([["good"]])[0], 5)
        self.assertEqual(self.model.predict([["bad", "bad"]])[0], 1)

    def test_unknown_and_empty_use_prior_without_growing_vocabulary(self):
        before = self.model.vocabulary.copy()
        np.testing.assert_allclose(self.model.scores([[], ["unseen"]]),
                                   np.tile(self.model.log_prior, (2, 1)))
        self.assertEqual(self.model.vocabulary, before)

    def test_five_classes_and_long_document(self):
        model = NaiveBayes().fit([[str(i)] for i in range(1, 6)], list(range(1, 6)))
        self.assertEqual(model.predict([["4"] * 10000])[0], 4)
        self.assertTrue(np.isfinite(model.scores([["4"] * 10000])).all())

    def test_thai_negation(self):
        words = tokenize("อาหารไม่อร่อย !!! https://example.com")
        self.assertIn("ไม่", words)
        self.assertIn("อร่อย", words)
        self.assertNotIn("!!!", words)


if __name__ == "__main__":
    unittest.main()
