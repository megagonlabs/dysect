import os

import altair as alt
import streamlit as st

from app.utils import (count_new_triples_by_iter, count_triples_by_iteration,
                       count_triples_by_source, count_triples_by_source_iter,
                       count_unique_triples_by_source,
                       get_avg_confidence_by_source_iter,
                       get_new_concepts_at_iter, get_new_triples_at_iter,
                       load_json)

summary_tab, iter_tab, source_tab, viewer_tab = st.tabs(
    [
        ":material/finance: Report",
        ":material/history: Iterations",
        ":material/smart_toy: Sources",
        ":material/data_object: Concept Viewer",
    ]
)

with viewer_tab:
    selected_concept = st.selectbox(
        "Select a concept:",
        sorted(st.session_state.concept_file_map.keys()),
    )

    if selected_concept:
        file_path = os.path.join(
            st.session_state.current_kb_full_path,
            "kb",
            st.session_state.concept_file_map[selected_concept],
        )
        st.markdown(
            f":paperclip: : `{st.session_state.concept_file_map[selected_concept]}`"
        )
        st.json(load_json(file_path))


with iter_tab:
    shared_iter_axis = alt.Axis(
        values=st.session_state.summary.get("iterations", []),
        labelOverlap=False,
        labelPadding=1,
    )
    if not st.session_state.df_triples.empty:
        df = st.session_state.df_triples
        df_d = st.session_state.df_triples_detail
        new_triple_counts_per_iter = count_new_triples_by_iter(df, df_d)
        triple_counts_per_iter = count_triples_by_iteration(df, df_d)
    col1, col2 = st.columns(2)
    col1.subheader(
        "New Unique Triples per Iteration",
        help="Number of newly discovered unique triples `unique([<h,r,t>])` in each iteration.",
    )
    if not st.session_state.df_triples.empty:
        chart1 = (
            alt.Chart(new_triple_counts_per_iter)
            .mark_line(point=True)
            .encode(
                x=alt.X("iter:Q", title="Iteration", axis=shared_iter_axis),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["iter", "count"],  # hover tooltips
            )
        )
        col1.altair_chart(chart1, use_container_width=True)

    col2.subheader(
        "Number of Triples per Iteration",
        help="Total number of triples `count([<h,r,t>])` (including duplicates) recorded in each iteration.",
    )
    if not st.session_state.df_triples.empty:
        chart2 = (
            alt.Chart(triple_counts_per_iter)
            .mark_line(point=True)
            .encode(
                x=alt.X("iter:Q", title="Iteration", axis=shared_iter_axis),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["iter", "count"],  # hover tooltips
            )
        )
        col2.altair_chart(chart2, use_container_width=True)

    # Add subtabs for each iteration showing new concepts and triples
    st.subheader("New Concepts and Triples by Iteration")
    iterations = st.session_state.summary.get("iterations", [])
    if iterations and not st.session_state.df_triples.empty:
        iter_subtabs = st.tabs([f"{i}" for i in iterations])
        for idx, iteration in enumerate(iterations):
            with iter_subtabs[idx]:
                col_concepts, col_triples = st.columns([3, 7])

                # New Concepts
                with col_concepts:
                    new_concepts_df = get_new_concepts_at_iter(
                        st.session_state.df_concepts, iteration
                    )
                    if not new_concepts_df.empty:
                        st.write(f"**{len(new_concepts_df)}** new concepts")
                        st.dataframe(
                            new_concepts_df,
                            column_config={
                                "concept": "Concept",
                                "type": "Type",
                                "total_freq": "Frequency",
                            },
                            width='stretch',
                            hide_index=True,
                        )
                    else:
                        st.info(f"No new concepts")

                # New Triples
                with col_triples:
                    new_triples_df = get_new_triples_at_iter(df, df_d, iteration)
                    if not new_triples_df.empty:
                        st.write(f"**{len(new_triples_df)}** new unique triples")
                        st.dataframe(
                            new_triples_df,
                            column_config={
                                "h": "Head",
                                "r": "Relation",
                                "t": "Tail",
                                "frequency": "Frequency",
                                "confidence": st.column_config.NumberColumn(
                                    "Avg Confidence", format="%.3f"
                                ),
                                "source": "Sources",
                            },
                            width='stretch',
                            hide_index=True,
                        )
                    else:
                        st.info(f"No new triples")


def shorten_sources_dict(sources, max_len=20):
    counts = {}
    mapping = {}

    for s in sources:
        # Truncate if longer than max_len and remove trailing spaces
        short = s[:max_len].rstrip() if len(s) > max_len else s

        # Count duplicates
        if short in counts:
            counts[short] += 1
            short_unique = f"{short} {counts[short]}".rstrip()
        else:
            counts[short] = 1
            short_unique = short

        mapping[s] = short_unique

    return mapping


with source_tab:
    st.subheader("Used Sources")
    sources = st.session_state.summary.get("sources", [])
    shorten_src_map = shorten_sources_dict(sources)
    for src in sorted(sources):
        st.markdown(f"`{shorten_src_map[src]}`: {src}")

    shared_x_axis = alt.Axis(labelAngle=-20, labelOverlap=False, labelPadding=1)
    shared_iter_axis = alt.Axis(
        values=st.session_state.summary.get("iterations", []),
        labelOverlap=False,
        labelPadding=1,
    )
    shared_color_scale = alt.Scale(
        domain=sorted(shorten_src_map.values()), scheme="tableau10"
    )
    if not st.session_state.df_triples.empty:
        df = st.session_state.df_triples
        df_d = st.session_state.df_triples_detail
        unique_triple_counts_per_source = count_unique_triples_by_source(df, df_d)
        unique_triple_counts_per_source["source"] = unique_triple_counts_per_source[
            "source"
        ].map(shorten_src_map)

        triple_counts_per_source = count_triples_by_source(df, df_d)
        triple_counts_per_source["source"] = triple_counts_per_source["source"].map(
            shorten_src_map
        )

        triple_counts_per_source_iter = count_triples_by_source_iter(df, df_d)
        triple_counts_per_source_iter["source"] = triple_counts_per_source_iter[
            "source"
        ].map(shorten_src_map)

        avg_conf_per_source_iter = get_avg_confidence_by_source_iter(df, df_d)
        avg_conf_per_source_iter["source"] = avg_conf_per_source_iter["source"].map(
            shorten_src_map
        )

        dist_conf = df_d[["source", "confidence", "frequency"]].copy()
        dist_conf["source"] = dist_conf["source"].map(shorten_src_map)

    col1, col2, col3 = st.columns([1, 1, 2])
    col1.subheader(
        "Number of Unique Triples",
        help="Number of unique triples `unique([<h,r,t>])` contributed by each source.",
    )
    if not st.session_state.df_triples.empty:
        chart1 = (
            alt.Chart(unique_triple_counts_per_source)
            .mark_bar()
            .encode(
                x=alt.X("source:N", title="Source", axis=shared_x_axis),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("source:N", scale=shared_color_scale, legend=None),
            )
        )
        col1.altair_chart(chart1, use_container_width=True)

    col2.subheader(
        "Number of Triples",
        help="Total number of triples `count([<h,r,t>])` (including duplicates) contributed by each source.",
    )
    if not st.session_state.df_triples.empty:
        chart2 = (
            alt.Chart(triple_counts_per_source)
            .mark_bar()
            .encode(
                x=alt.X("source:N", title="Source", axis=shared_x_axis),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("source:N", scale=shared_color_scale, legend=None),
            )
        )
        col2.altair_chart(chart2, use_container_width=True)

    col3.subheader(
        "Number of Triples Across Iterations",
        help="Total number of triples `count([<h,r,t>])` (including duplicates) contributed by each source across iterations.",
    )
    if not st.session_state.df_triples.empty:
        chart3 = (
            alt.Chart(triple_counts_per_source_iter)
            .mark_line(point=True)
            .encode(
                x=alt.X("iter:Q", title="Iteration", axis=shared_iter_axis),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("source:N", scale=shared_color_scale, legend=None),
                tooltip=["iter", "source", "count"],  # hover tooltips
            )
        )
        col3.altair_chart(chart3, use_container_width=True)

    col4, col5 = st.columns(2)
    col4.subheader(
        "Distribution of Triple Confidence",
        help="Distribution of triple confidence scores `conf(<h,r,t>)` for each source, with the width showing how many triples fall at a given confidence level.",
    )
    if not st.session_state.df_triples.empty:
        chart4 = (
            alt.Chart(dist_conf)
            .mark_bar(opacity=0.4, binSpacing=0)
            .encode(
                alt.X("confidence:Q", bin=alt.Bin(maxbins=100), title="Confidence"),
                alt.Y("sum(frequency):Q", stack=None, title="Count"),
                alt.Color("source:N", scale=shared_color_scale, legend=None),
            )
        )
        col4.altair_chart(chart4, use_container_width=True)

    col5.subheader(
        "Average Triple Confidence Across Iterations",
        help="Average confidence score of triples `avg([conf(<h,r,t>)])` from each source across iterations.",
    )
    if not st.session_state.df_triples.empty:
        chart5 = (
            alt.Chart(avg_conf_per_source_iter)
            .mark_line(point=True)
            .encode(
                x=alt.X("iter:Q", title="Iteration", axis=shared_iter_axis),
                y=alt.Y("weighted_avg_conf:Q", title="Avg(Confidence)"),
                color=alt.Color("source:N", scale=shared_color_scale, legend=None),
                tooltip=["iter", "source", "weighted_avg_conf"],  # hover tooltips
            )
        )
        col5.altair_chart(chart5, use_container_width=True)

with summary_tab:
    st.subheader("Summary")
    ct_s = st.container()
    with ct_s:
        res = st.session_state.summary
        st.write(
            "Total Concepts:",
            res.get("num_concepts"),
            "(with files: ",
            res.get("num_concept_files"),
            ")",
        )
        st.write(
            "Total Entities:",
            res.get("num_entities"),
            "(with files: ",
            res.get("num_entities_w_files"),
            ")",
        )
        st.write(
            "Total Relations:",
            res.get("num_relations"),
            "(with files: ",
            res.get("num_relations_w_files"),
            ")",
        )
        st.write("Sources:", ", ".join([f"`{src}`" for src in res.get("sources", [])]))
        st.write(
            "Iterations:", ", ".join([f"`{src}`" for src in res.get("iterations", [])])
        )
        if st.session_state.errors:
            st.write("Errors:", st.session_state.errors)

    st.subheader("Concepts")
    ct_c = st.container()
    type_options = ["All", "Entities", "Relations"]
    meta_options = ["All", "Meta-Only", "Non-Meta"]
    with ct_c:
        col1, col2, col3 = st.columns(3)
        concept_type = col1.segmented_control(
            "Concept Type",
            type_options,
            default=type_options[0],
            label_visibility="collapsed",
        )
        meta_type = col2.segmented_control(
            "Meta",
            meta_options,
            default=meta_options[0],
            label_visibility="collapsed",
            disabled=True,
        )
        with_files_only = col3.toggle("Show concepts with files")

    if not st.session_state.df_concepts.empty:
        df_c = st.session_state.df_concepts
        if concept_type == "Entities":
            df_c = df_c[df_c["is_rel"] == False]
        elif concept_type == "Relations":
            df_c = df_c[df_c["is_rel"] == True]
        if with_files_only:
            df_c = df_c[df_c["file_location"].notna()]
        ct_c.dataframe(
            df_c,
            column_config={
                "concept": st.column_config.TextColumn(
                    "Concept",
                    help="Name of the concept (entity or relation)",
                ),
                "file_location": st.column_config.TextColumn(
                    "File",
                    help="Path to the concept's JSON file in the KB",
                ),
                "is_rel": st.column_config.CheckboxColumn(
                    "Is Rel?",
                    help="Whether this concept is a relation (vs an entity)",
                ),
                "all_uniq_cnt": st.column_config.NumberColumn(
                    "Uniq.All",
                    help="Count of unique triples this concept appears in (as head, relation, or tail)",
                ),
                "h_uniq_cnt": st.column_config.NumberColumn(
                    "Uniq.H",
                    help="Count of unique triples where this concept is the head",
                ),
                "r_uniq_cnt": st.column_config.NumberColumn(
                    "Uniq.R",
                    help="Count of unique triples where this concept is the relation",
                ),
                "t_uniq_cnt": st.column_config.NumberColumn(
                    "Uniq.T",
                    help="Count of unique triples where this concept is the tail",
                ),
                "all_raw_cnt": st.column_config.NumberColumn(
                    "Cnt.All",
                    help="Total raw count of triple occurrences (including duplicates)",
                ),
                "h_raw_cnt": st.column_config.NumberColumn(
                    "Cnt.H",
                    help="Raw count as head (including duplicates)",
                ),
                "r_raw_cnt": st.column_config.NumberColumn(
                    "Cnt.R",
                    help="Raw count as relation (including duplicates)",
                ),
                "t_raw_cnt": st.column_config.NumberColumn(
                    "Cnt.T",
                    help="Raw count as tail (including duplicates)",
                ),
                "iter_min": st.column_config.NumberColumn(
                    "First Iter",
                    help="First iteration this concept appeared in",
                ),
                "iter_max": st.column_config.NumberColumn(
                    "Last Iter",
                    help="Last iteration this concept appeared in",
                ),
            },
            hide_index=True,
        )
