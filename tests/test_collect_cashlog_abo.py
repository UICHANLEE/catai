import io
import unittest

from PIL import Image

from scripts.collect_cashlog_abo import encode_image, product_types, select_candidates


class CollectCashlogAboTests(unittest.TestCase):
    def test_product_types_normalizes_values(self) -> None:
        row = {"product_type": [{"value": " shoes "}, "GROCERY", {"other": "ignored"}]}
        self.assertEqual(product_types(row), {"SHOES", "GROCERY"})

    def test_selection_is_deterministic_and_capped_per_leaf(self) -> None:
        candidates = {
            "meal_grocery": [
                {"image_id": f"grocery-{index}", "leaf_id": "meal_grocery"}
                for index in range(10)
            ],
            "fashion_clothes": [
                {"image_id": f"shoe-{index}", "leaf_id": "fashion_clothes"}
                for index in range(10)
            ],
        }
        first = select_candidates(candidates, per_leaf=3, seed=7)
        second = select_candidates(candidates, per_leaf=3, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)

    def test_encode_image_outputs_valid_jpeg_and_hash(self) -> None:
        source = io.BytesIO()
        Image.new("RGB", (640, 320), "white").save(source, format="PNG")
        encoded, width, height, digest = encode_image(io.BytesIO(source.getvalue()), 256)
        self.assertEqual((width, height), (256, 128))
        self.assertEqual(len(digest), 64)
        with Image.open(io.BytesIO(encoded)) as image:
            self.assertEqual(image.format, "JPEG")


if __name__ == "__main__":
    unittest.main()
