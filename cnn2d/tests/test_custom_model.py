import unittest
import torch


class TestCustomModel(unittest.TestCase):
    def setUp(self):
        self.model = CustomModel()

    def test_output_sizes(self):
        input_size = (1, 1, 128, 151)
        input_tensor = torch.randn(input_size)

        output_sizes = self.model(input_tensor)

        expected_sizes = [(1, 32, 63, 74), (1, 64, 61, 72)]
        self.assertEqual(len(output_sizes), len(expected_sizes))
        for i, size in enumerate(output_sizes):
            self.assertEqual(size, expected_sizes[i])

if __name__ == '__main__':
    unittest.main()
