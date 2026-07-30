import numpy as np

from scripts.export_quantize_cashlog33_siglip import (
    assignment_matrix,
    encoder_layer_index,
    prompt_assignment_matrix,
)


def test_assignment_matrix_places_source_classes_in_taxonomy_order():
    matrix = assignment_matrix(["leaf_b", "leaf_a"], ["leaf_a", "leaf_b"])

    np.testing.assert_array_equal(
        matrix,
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_prompt_assignment_averages_each_leaf_prompts():
    matrix = prompt_assignment_matrix(
        {"leaf_a": [0, 2], "leaf_b": [1]},
        ["leaf_a", "leaf_b"],
        3,
    )

    np.testing.assert_array_equal(
        matrix,
        np.asarray(
            [[0.5, 0.0], [0.0, 1.0], [0.5, 0.0]],
            dtype=np.float32,
        ),
    )


def test_encoder_layer_index_only_matches_siglip_encoder_nodes():
    assert (
        encoder_layer_index(
            "/vision_model/encoder/layers.11/mlp/fc2/MatMul"
        )
        == 11
    )
    assert encoder_layer_index("/vision_model/head/mlp/fc1/MatMul") is None
