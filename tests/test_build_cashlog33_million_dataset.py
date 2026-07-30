from scripts.build_cashlog33_million_dataset import (
    assign_partitions,
    build_schedule,
    deduplicate_rows,
)


def row(sample_id, leaf_id, digest, source="openimages_v7_train"):
    return {
        "sample_id": sample_id,
        "leaf_id": leaf_id,
        "sha256": digest,
        "source": source,
    }


def test_deduplicate_rows_drops_cross_leaf_conflicts():
    accepted, stats = deduplicate_rows(
        [
            row("a", "one", "same"),
            row("b", "one", "same"),
            row("c", "one", "conflict"),
            row("d", "two", "conflict"),
        ]
    )

    assert [item["sha256"] for item in accepted] == ["same"]
    assert stats["duplicate_rows_dropped"] == 1
    assert stats["cross_leaf_conflict_rows_dropped"] == 2


def test_partition_and_schedule_are_exact_and_balanced():
    rows = [
        row(f"a-{index}", "one", f"a{index}") for index in range(10)
    ] + [row(f"b-{index}", "two", f"b{index}") for index in range(10)]
    partitioned = assign_partitions(rows, validation_ratio=0.2, seed=9)
    indexes, seeds, leaves = build_schedule(
        partitioned, target_views=100, minimum_originals_per_leaf=2, seed=9
    )

    assert len([item for item in partitioned if item["partition"] == "validation"]) == 4
    assert len(indexes) == len(seeds) == 100
    assert leaves == ["one", "two"]
    scheduled_labels = [partitioned[index]["leaf_id"] for index in indexes]
    assert scheduled_labels.count("one") == scheduled_labels.count("two") == 50


def test_schedule_uses_every_eligible_original_before_oversampling():
    rows = [
        {
            **row(f"a-{index}", "large", f"a{index}", source="other"),
            "partition": "train",
        }
        for index in range(80)
    ] + [
        {
            **row(f"b-{index}", "small", f"b{index}", source="other"),
            "partition": "train",
        }
        for index in range(8)
    ]

    indexes, _, leaves = build_schedule(
        rows, target_views=100, minimum_originals_per_leaf=2, seed=9
    )

    assert leaves == ["large", "small"]
    assert set(range(len(rows))) <= set(int(index) for index in indexes)
    scheduled_labels = [rows[index]["leaf_id"] for index in indexes]
    assert scheduled_labels.count("large") == 80
    assert scheduled_labels.count("small") == 20
