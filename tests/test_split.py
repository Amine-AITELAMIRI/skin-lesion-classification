import tempfile
import unittest
from pathlib import Path

from split_data import make_train_val, split_indices


class SplitTests(unittest.TestCase):
    def test_fixed_class_balanced_split(self):
        with tempfile.TemporaryDirectory() as temp:
            train_dir = Path(temp) / "train"
            for label, count in (("Benign", 10), ("Malignant", 8)):
                folder = train_dir / label
                folder.mkdir(parents=True)
                for index in range(count):
                    (folder / f"{index:03}.jpg").touch()

            train_indices, val_indices = split_indices(train_dir)
            self.assertEqual((train_indices, val_indices), split_indices(train_dir))
            self.assertFalse(set(train_indices) & set(val_indices))
            self.assertEqual(len(train_indices), 14)
            self.assertEqual(len(val_indices), 4)

            train, val = make_train_val(train_dir, None, None)
            self.assertEqual(train.indices, train_indices)
            self.assertEqual(val.indices, val_indices)
            self.assertEqual([y for i in val.indices for _, y in [val.dataset.samples[i]]].count(0), 2)
            self.assertEqual([y for i in val.indices for _, y in [val.dataset.samples[i]]].count(1), 2)

    def test_test_folder_cannot_be_used_for_validation(self):
        with self.assertRaisesRegex(ValueError, "train folder"):
            split_indices(Path("melanoma-cancer-dataset/test"))


if __name__ == "__main__":
    unittest.main()
