import sys
import types
import importlib
import unittest

class DummyEstimator:
    def fit(self, X, y):
        self.fit_called = True
    def predict(self, X):
        return [0] * len(X)

def load_morp():
    if 'numpy' not in sys.modules:
        np = types.SimpleNamespace(
            zeros=lambda shape: [[0] * shape[1] for _ in range(shape[0])],
            hstack=lambda a, b: a + b,
            array=lambda x: x,
        )
        sys.modules['numpy'] = np
    if 'scipy.sparse' not in sys.modules:
        sparse_mod = types.SimpleNamespace(
            csr_matrix=lambda x: x,
            vstack=lambda x, y: x + y,
        )
        sys.modules['scipy'] = types.SimpleNamespace(sparse=sparse_mod)
        sys.modules['scipy.sparse'] = sparse_mod
    if 'sklearn.svm' not in sys.modules:
        svm = types.SimpleNamespace(LinearSVC=lambda **k: DummyEstimator())
        nb = types.SimpleNamespace(GaussianNB=lambda **k: DummyEstimator())
        lm = types.SimpleNamespace(SGDClassifier=lambda **k: DummyEstimator())
        sys.modules['sklearn'] = types.SimpleNamespace(
            svm=svm,
            naive_bayes=nb,
            linear_model=lm,
        )
        sys.modules['sklearn.svm'] = svm
        sys.modules['sklearn.naive_bayes'] = nb
        sys.modules['sklearn.linear_model'] = lm
    Morp = importlib.import_module('Morp.morp').Morp
    return Morp


class TestMorp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Morp = load_morp()

    def setUp(self):
        self.analyzer = self.Morp(estimator=DummyEstimator())

    def test_ngram(self):
        self.assertEqual(self.analyzer.ngram('abcd', 2), ['ab', 'bc', 'cd'])

    def test_get_type(self):
        gt = self.analyzer.get_type
        self.assertEqual(gt('ぁ'), 0)
        self.assertEqual(gt('ア'), 1)
        self.assertEqual(gt('亜'), 2)
        self.assertEqual(gt('A'), 3)
        self.assertEqual(gt('1'), 4)
        self.assertEqual(gt('?'), 5)

    def test_get_types_single(self):
        self.assertEqual(self.analyzer.get_types('ア'), '1')

    def test_get_position_part(self):
        text = '君|の|名-前|を|教えて|くれ'
        self.assertEqual(self.analyzer.get_position_part(text), [1, 2, 3, 4, 5, 8])

    def test_get_teacher(self):
        self.assertEqual(
            self.analyzer.get_teacher('私 は 元気 です'),
            [1, 1, 0, 1, 0],
        )

    def test_get_teacher_part(self):
        text = '私|は|お-金|が|好-き|で-す'
        self.assertEqual(
            self.analyzer.get_teacher_part(text),
            [1, 1, 0, 1, 1, 0, 1, 0],
        )


if __name__ == '__main__':
    unittest.main()
