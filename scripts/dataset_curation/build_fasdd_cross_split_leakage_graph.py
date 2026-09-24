#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "cross_split_scoring"
    / "high_priority_pairs.csv"
)

OUT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "cross_split_graph"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMPONENTS_CSV = OUT_DIR / "components.csv"
MEMBERS_CSV = OUT_DIR / "members.csv"
EDGES_CSV = OUT_DIR / "edges.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"


# ============================================================
# UNION FIND
# ============================================================

class DSU:

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):

        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ============================================================
# COMPONENT TYPE
# ============================================================

def component_type(splits):

    s = set(splits)

    if s == {"train", "val", "test"}:
        return "TRAIN_VAL_TEST"

    if s == {"train", "test"}:
        return "TRAIN_TEST"

    if s == {"val", "test"}:
        return "VAL_TEST"

    if s == {"train", "val"}:
        return "TRAIN_VAL"

    return "OTHER"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 94)
    print("FASDD CROSS-SPLIT NEAR-DUPLICATE GRAPH")
    print("=" * 94)

    with INPUT.open(
        encoding="utf-8",
        newline="",
    ) as f:

        edges = list(
            csv.DictReader(f)
        )

    print(
        "High-priority edges:",
        f"{len(edges):,}"
    )

    if not edges:
        raise RuntimeError(
            "No high-priority edges found."
        )

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    dsu = DSU()

    node_split = {}
    adjacency = defaultdict(set)

    for row in edges:

        a = row["stem_a"]
        b = row["stem_b"]

        sa = row["split_a"]
        sb = row["split_b"]

        node_split[a] = sa
        node_split[b] = sb

        dsu.union(a, b)

        adjacency[a].add(b)
        adjacency[b].add(a)

    # ========================================================
    # COMPONENT MEMBERS
    # ========================================================

    component_nodes = defaultdict(list)

    for node in node_split:

        root = dsu.find(node)

        component_nodes[
            root
        ].append(node)

    ordered_components = sorted(
        component_nodes.values(),
        key=lambda nodes: (
            -len(nodes),
            sorted(nodes)[0],
        ),
    )

    node_component = {}

    for component_id, nodes in enumerate(
        ordered_components,
        1,
    ):

        for node in nodes:

            node_component[
                node
            ] = component_id

    # ========================================================
    # COMPONENT EDGES
    # ========================================================

    component_edges = defaultdict(list)

    for row in edges:

        cid = node_component[
            row["stem_a"]
        ]

        component_edges[
            cid
        ].append(row)

    # ========================================================
    # COMPONENT STATS
    # ========================================================

    components = []

    size_distribution = Counter()
    type_distribution = Counter()

    test_component_count = 0

    total_test_nodes = set()
    total_val_nodes = set()
    total_train_nodes = set()

    for component_id, nodes in enumerate(
        ordered_components,
        1,
    ):

        nodes = sorted(nodes)

        split_counts = Counter(
            node_split[n]
            for n in nodes
        )

        splits = sorted(
            split_counts
        )

        ctype = component_type(
            splits
        )

        cedges = component_edges[
            component_id
        ]

        n = len(nodes)
        m = len(cedges)

        possible_edges = (
            n * (n - 1) / 2
        )

        density = (
            m / possible_edges
            if possible_edges > 0
            else 0.0
        )

        priority_counts = Counter(
            e["leakage_priority"]
            for e in cedges
        )

        visual_counts = Counter(
            e["visual_tier"]
            for e in cedges
        )

        annotation_counts = Counter(
            e["annotation_state"]
            for e in cedges
        )

        relation_counts = Counter(
            e["relation"]
            for e in cedges
        )

        degrees = {
            node:
                len(adjacency[node])
            for node in nodes
        }

        max_degree = max(
            degrees.values(),
            default=0,
        )

        min_degree = min(
            degrees.values(),
            default=0,
        )

        mean_degree = (
            sum(degrees.values())
            / len(degrees)
            if degrees
            else 0.0
        )

        # ----------------------------------------------------
        # Direct test anchors
        # ----------------------------------------------------

        test_nodes = {
            n
            for n in nodes
            if node_split[n] == "test"
        }

        val_nodes = {
            n
            for n in nodes
            if node_split[n] == "val"
        }

        train_nodes = {
            n
            for n in nodes
            if node_split[n] == "train"
        }

        direct_to_test = set()

        for test_node in test_nodes:

            for neighbor in adjacency[
                test_node
            ]:

                if node_split[
                    neighbor
                ] != "test":

                    direct_to_test.add(
                        neighbor
                    )

        if test_nodes:

            test_component_count += 1

            total_test_nodes.update(
                test_nodes
            )

            total_val_nodes.update(
                val_nodes
            )

            total_train_nodes.update(
                train_nodes
            )

        components.append({
            "component_id":
                component_id,

            "component_type":
                ctype,

            "nodes":
                n,

            "edges":
                m,

            "density":
                density,

            "train_nodes":
                split_counts["train"],

            "val_nodes":
                split_counts["val"],

            "test_nodes":
                split_counts["test"],

            "very_high_edges":
                priority_counts[
                    "VERY_HIGH"
                ],

            "high_edges":
                priority_counts[
                    "HIGH"
                ],

            "extreme_edges":
                visual_counts[
                    "EXTREME"
                ],

            "ultra_edges":
                visual_counts[
                    "ULTRA"
                ],

            "annotation_exact":
                annotation_counts[
                    "EXACT"
                ],

            "annotation_strong":
                annotation_counts[
                    "STRONG"
                ],

            "annotation_moderate":
                annotation_counts[
                    "MODERATE"
                ],

            "annotation_conflict":
                annotation_counts[
                    "CONFLICT"
                ],

            "train_val_edges":
                relation_counts[
                    "train_val"
                ],

            "train_test_edges":
                relation_counts[
                    "train_test"
                ],

            "val_test_edges":
                relation_counts[
                    "val_test"
                ],

            "direct_non_test_nodes_to_test":
                len(
                    direct_to_test
                ),

            "max_degree":
                max_degree,

            "min_degree":
                min_degree,

            "mean_degree":
                mean_degree,
        })

        size_distribution[n] += 1
        type_distribution[ctype] += 1

    # ========================================================
    # SAVE COMPONENTS
    # ========================================================

    component_fields = [
        "component_id",
        "component_type",
        "nodes",
        "edges",
        "density",
        "train_nodes",
        "val_nodes",
        "test_nodes",
        "very_high_edges",
        "high_edges",
        "extreme_edges",
        "ultra_edges",
        "annotation_exact",
        "annotation_strong",
        "annotation_moderate",
        "annotation_conflict",
        "train_val_edges",
        "train_test_edges",
        "val_test_edges",
        "direct_non_test_nodes_to_test",
        "max_degree",
        "min_degree",
        "mean_degree",
    ]

    with COMPONENTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=component_fields,
        )

        writer.writeheader()

        for row in components:

            out = dict(row)

            out["density"] = (
                f"{row['density']:.8f}"
            )

            out["mean_degree"] = (
                f"{row['mean_degree']:.6f}"
            )

            writer.writerow(out)

    # ========================================================
    # SAVE MEMBERS
    # ========================================================

    with MEMBERS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "component_id",
            "stem",
            "split",
            "degree",
            "direct_test_neighbor",
            "test_neighbor_count",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for component_id, nodes in enumerate(
            ordered_components,
            1,
        ):

            for node in sorted(nodes):

                test_neighbors = [
                    neighbor
                    for neighbor
                    in adjacency[node]
                    if node_split[
                        neighbor
                    ] == "test"
                ]

                writer.writerow({
                    "component_id":
                        component_id,

                    "stem":
                        node,

                    "split":
                        node_split[node],

                    "degree":
                        len(
                            adjacency[node]
                        ),

                    "direct_test_neighbor":
                        int(
                            len(
                                test_neighbors
                            ) > 0
                        ),

                    "test_neighbor_count":
                        len(
                            test_neighbors
                        ),
                })

    # ========================================================
    # SAVE EDGES WITH COMPONENT ID
    # ========================================================

    edge_fields = [
        "component_id"
    ] + list(
        edges[0].keys()
    )

    with EDGES_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=edge_fields,
        )

        writer.writeheader()

        for row in edges:

            out = {
                "component_id":
                    node_component[
                        row["stem_a"]
                    ],

                **row,
            }

            writer.writerow(out)

    # ========================================================
    # SUMMARY
    # ========================================================

    total_nodes = len(
        node_split
    )

    summary = {
        "edges":
            len(edges),

        "nodes":
            total_nodes,

        "components":
            len(components),

        "component_type_distribution":
            dict(
                type_distribution
            ),

        "component_size_distribution":
            {
                str(k): v
                for k, v
                in sorted(
                    size_distribution.items()
                )
            },

        "components_containing_test":
            test_component_count,

        "nodes_inside_test_components": {
            "train":
                len(
                    total_train_nodes
                ),

            "val":
                len(
                    total_val_nodes
                ),

            "test":
                len(
                    total_test_nodes
                ),
        },

        "important":
            (
                "Connected components are diagnostic only. "
                "Do not remove all nodes in a component "
                "because similarity is not transitive."
            ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 94)
    print("NEAR-DUPLICATE GRAPH SUMMARY")
    print("=" * 94)

    print(
        "Edges      :",
        f"{len(edges):,}"
    )

    print(
        "Nodes      :",
        f"{total_nodes:,}"
    )

    print(
        "Components :",
        f"{len(components):,}"
    )

    print()
    print("COMPONENT TYPE")
    print("-" * 94)

    for name in (
        "TRAIN_VAL",
        "TRAIN_TEST",
        "VAL_TEST",
        "TRAIN_VAL_TEST",
        "OTHER",
    ):

        print(
            f"{name:18}: "
            f"{type_distribution[name]:,}"
        )

    print()
    print("COMPONENT SIZE DISTRIBUTION")
    print("-" * 94)

    for size in sorted(
        size_distribution
    ):

        print(
            f"size {size:4}: "
            f"{size_distribution[size]:,}"
        )

    print()
    print("TEST-CONTAINING COMPONENTS")
    print("-" * 94)

    print(
        "Components:",
        f"{test_component_count:,}"
    )

    print(
        "Train nodes:",
        f"{len(total_train_nodes):,}"
    )

    print(
        "Val nodes  :",
        f"{len(total_val_nodes):,}"
    )

    print(
        "Test nodes :",
        f"{len(total_test_nodes):,}"
    )

    if components:

        largest = max(
            components,
            key=lambda x:
                x["nodes"]
        )

        print()
        print("LARGEST COMPONENT")
        print("-" * 94)

        print(
            "ID      :",
            largest[
                "component_id"
            ]
        )

        print(
            "Type    :",
            largest[
                "component_type"
            ]
        )

        print(
            "Nodes   :",
            largest[
                "nodes"
            ]
        )

        print(
            "Edges   :",
            largest[
                "edges"
            ]
        )

        print(
            "Density :",
            f"{largest['density']:.6f}"
        )

        print(
            "Train   :",
            largest[
                "train_nodes"
            ]
        )

        print(
            "Val     :",
            largest[
                "val_nodes"
            ]
        )

        print(
            "Test    :",
            largest[
                "test_nodes"
            ]
        )

    print()
    print(
        "Components CSV:",
        COMPONENTS_CSV
    )

    print(
        "Members CSV   :",
        MEMBERS_CSV
    )

    print(
        "Edges CSV     :",
        EDGES_CSV
    )

    print(
        "Summary JSON  :",
        SUMMARY_JSON
    )

    print()
    print(
        "No manifest or dataset file was modified."
    )

    print("=" * 94)


if __name__ == "__main__":
    main()
