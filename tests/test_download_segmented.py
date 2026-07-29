import unittest

from scripts.download_segmented import segment_ranges


class DownloadSegmentedTests(unittest.TestCase):
    def test_ranges_cover_payload_without_gaps(self) -> None:
        ranges = segment_ranges(10, 3)
        self.assertEqual(ranges, [(0, 3), (4, 7), (8, 9)])
        flattened = [value for start, end in ranges for value in range(start, end + 1)]
        self.assertEqual(flattened, list(range(10)))

    def test_rejects_non_positive_values(self) -> None:
        with self.assertRaises(ValueError):
            segment_ranges(0, 2)
        with self.assertRaises(ValueError):
            segment_ranges(10, 0)


if __name__ == "__main__":
    unittest.main()
