import streamlit as st


@st.cache_data
def get_summary(
    concept_file_map: dict, df: "pd.DataFrame", df_d: "pd.DataFrame"
) -> dict:
    import pandas as pd

    num_concept_files = len(concept_file_map)
    num_concepts = pd.concat([df["h"], df["r"], df["t"]]).nunique()
    relations = df["r"].unique()
    num_relations = len(relations)  # df["r"].nunique()
    num_relations_w_files = len(set(relations) & set(concept_file_map))
    num_entities = num_concepts - num_relations
    num_entities_w_files = num_concept_files - num_relations_w_files
    sources = sorted(df_d["source"].unique())
    iterations = sorted(df_d["iter"].unique())
    return {
        "num_concept_files": num_concept_files,
        "num_concepts": num_concepts,
        "num_entities": num_entities,
        "num_entities_w_files": num_entities_w_files,
        "num_relations": num_relations,
        "num_relations_w_files": num_relations_w_files,
        "sources": sources,
        "iterations": iterations,
    }


@st.cache_data
def generate_concept_table(
    concept_file_map: dict, df: list, df_d: list
) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    relations = np.unique(df["r"].values)
    entities = np.unique(df[["h", "t"]].values.ravel())
    entities_no_rel = np.setdiff1d(entities, relations)
    concepts = np.concatenate([relations, entities_no_rel])
    is_rel = np.concatenate(
        [
            np.ones(len(relations), dtype=bool),
            np.zeros(len(entities_no_rel), dtype=bool),
        ]
    )

    sort_idx = np.argsort(concepts)
    concepts_sorted = concepts[sort_idx]
    is_rel_sorted = is_rel[sort_idx]

    counts = (
        pd.concat(
            [
                df["h"].value_counts().rename("h_uniq_cnt"),
                df["r"].value_counts().rename("r_uniq_cnt"),
                df["t"].value_counts().rename("t_uniq_cnt"),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
    )
    counts["all_uniq_cnt"] = counts[["h_uniq_cnt", "r_uniq_cnt", "t_uniq_cnt"]].sum(
        axis=1
    )
    counts = counts.reindex(concepts_sorted, fill_value=0)

    count_raw_h = df_d.groupby("h")["frequency"].sum()
    count_raw_r = df_d.groupby("r")["frequency"].sum()
    count_raw_t = df_d.groupby("t")["frequency"].sum()
    count_raw = (
        pd.concat(
            [
                count_raw_h.rename("h_raw_cnt"),
                count_raw_r.rename("r_raw_cnt"),
                count_raw_t.rename("t_raw_cnt"),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
    )
    count_raw["all_raw_cnt"] = count_raw[["h_raw_cnt", "r_raw_cnt", "t_raw_cnt"]].sum(
        axis=1
    )
    count_raw = count_raw.reindex(concepts_sorted, fill_value=0)

    source_file = pd.Series(concept_file_map).reindex(concepts_sorted).values

    df_long_iter = df_d.melt(
        id_vars=["iter"], value_vars=["h", "r", "t"], value_name="concept"
    )[["concept", "iter"]]
    iter_stats = (
        df_long_iter.groupby("concept")["iter"]
        .agg(["min", "max"])
        .rename(columns={"min": "iter_min", "max": "iter_max"})
    )
    iter_stats = iter_stats.reindex(concepts_sorted, fill_value=np.nan)

    df_c = (
        pd.DataFrame(
            {
                "concept": concepts_sorted,
                "is_rel": is_rel_sorted,
                "file_location": source_file,
            }
        )
        .join(counts, on="concept")
        .join(count_raw, on="concept")
        .join(iter_stats, on="concept")
    )

    df_c = df_c[
        [
            "concept",
            "file_location",
            "is_rel",
            "all_uniq_cnt",
            "h_uniq_cnt",
            "r_uniq_cnt",
            "t_uniq_cnt",
            "all_raw_cnt",
            "h_raw_cnt",
            "r_raw_cnt",
            "t_raw_cnt",
            "iter_min",
            "iter_max",
        ]
    ]

    return df_c


def count_triples_by_iteration(
    df: "pd.DataFrame", df_d: "pd.DataFrame"
) -> "pd.DataFrame":
    triple_counts_per_iter = (
        df_d.groupby("iter")
        .agg(count=("frequency", "sum"))
        .reset_index()
        .sort_values("iter")
    )
    return triple_counts_per_iter


def count_new_triples_by_iter(
    df: "pd.DataFrame", df_d: "pd.DataFrame"
) -> "pd.DataFrame":
    new_triple_counts_per_iter = (
        df_d.groupby(["h", "r", "t"], as_index=False)["iter"]
        .min()
        .groupby("iter")
        .size()
        .reset_index(name="count")
    )
    return new_triple_counts_per_iter


def count_unique_triples_by_source(
    df: "pd.DataFrame", df_d: "pd.DataFrame"
) -> "pd.DataFrame":
    unique_triple_counts_per_source = (
        df_d.groupby("source")[["h", "r", "t"]]
        .apply(lambda g: g.drop_duplicates().shape[0])
        .reset_index(name="count")
        .sort_values("source")
    )
    return unique_triple_counts_per_source


def count_triples_by_source(df: "pd.DataFrame", df_d: "pd.DataFrame") -> "pd.DataFrame":
    triple_counts_per_source = (
        df_d.groupby("source")
        .agg(count=("frequency", "sum"))
        .reset_index()
        .sort_values("source")
    )
    return triple_counts_per_source


def count_triples_by_source_iter(
    df: "pd.DataFrame", df_d: "pd.DataFrame"
) -> "pd.DataFrame":
    triple_counts_per_source_iter = (
        df_d.groupby(["source", "iter"])
        .agg(count=("frequency", "sum"))
        .reset_index()
        .sort_values(["source", "iter"])
    )
    return triple_counts_per_source_iter


def get_avg_confidence_by_source_iter(
    df: "pd.DataFrame", df_d: "pd.DataFrame"
) -> "pd.DataFrame":
    avg_conf_per_source_iter = (
        df_d.groupby(["source", "iter"])[["confidence", "frequency"]]
        .apply(
            lambda g: (g["confidence"] * g["frequency"]).sum() / g["frequency"].sum()
        )
        .reset_index(name="weighted_avg_conf")
        .sort_values(["source", "iter"])
    )
    return avg_conf_per_source_iter


@st.cache_data
def _get_first_seen_triples(df_d: "pd.DataFrame") -> "pd.DataFrame":
    """Cache the expensive first-seen computation (done once, reused for all iterations)."""
    first_seen = df_d.groupby(["h", "r", "t"], as_index=False)["iter"].min()
    first_seen = first_seen.rename(columns={"iter": "first_iter"})
    return first_seen


def get_new_concepts_at_iter(
    df_concepts: "pd.DataFrame", iteration: int
) -> "pd.DataFrame":
    """Get new concepts discovered at a specific iteration using df_concepts.iter_min."""
    new_concepts = df_concepts[df_concepts["iter_min"] == iteration].copy()
    new_concepts = new_concepts[["concept", "is_rel", "all_raw_cnt"]].copy()
    new_concepts["type"] = new_concepts["is_rel"].apply(
        lambda x: "relation" if x else "entity"
    )
    new_concepts = new_concepts.rename(columns={"all_raw_cnt": "frequency"})
    new_concepts = (
        new_concepts[["concept", "type", "frequency"]]
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )
    return new_concepts


def get_new_triples_at_iter(
    df: "pd.DataFrame", df_d: "pd.DataFrame", iteration: int
) -> "pd.DataFrame":
    """Get new unique triples discovered at a specific iteration, sorted by frequency."""
    # Get cached first-seen data
    first_seen = _get_first_seen_triples(df_d)

    # Filter to triples first seen at the specified iteration
    new_at_iter = first_seen[first_seen["first_iter"] == iteration][["h", "r", "t"]]

    # Merge back with df_d to get frequency info for these triples at this iteration
    iter_data = df_d[df_d["iter"] == iteration]
    new_triples = new_at_iter.merge(
        iter_data[["h", "r", "t", "frequency", "confidence", "source"]],
        on=["h", "r", "t"],
        how="left",
    )

    # Aggregate frequency per triple (sum across sources if multiple)
    new_triples_agg = (
        new_triples.groupby(["h", "r", "t"], as_index=False)
        .agg(
            {
                "frequency": "sum",
                "confidence": "mean",
                "source": lambda x: ", ".join(sorted(set(x))),
            }
        )
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )

    return new_triples_agg
