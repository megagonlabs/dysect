import pandas as pd
import streamlit as st

tab1, tab2, tab3 = st.tabs(
    [
        ":material/person_check: Annotate",
        ":material/add: Add New",
        ":material/checklist_rtl: Review & Export",
    ]
)


def add_annotation(h: str, r: str, t: str, conf: float):
    if h and r and t:
        # Check if annotation already exists
        existing = [
            a
            for a in st.session_state.annotations
            if a["h"] == h and a["r"] == r and a["t"] == t
        ]
        if existing:
            existing_conf = existing[0]["conf"]
            if existing_conf == conf:
                # Same button clicked again - remove annotation (un-mark)
                st.session_state.annotations = [
                    a
                    for a in st.session_state.annotations
                    if not (a["h"] == h and a["r"] == r and a["t"] == t)
                ]
            else:
                # Different button - update annotation
                for a in st.session_state.annotations:
                    if a["h"] == h and a["r"] == r and a["t"] == t:
                        a["conf"] = conf
        else:
            st.session_state.annotations.append({"h": h, "r": r, "t": t, "conf": conf})


def get_filtered_triples(
    df: pd.DataFrame, search_head: str, search_rel: str, search_tail: str
) -> pd.DataFrame:
    """Filter triples based on search terms (case-insensitive partial match)."""
    if df.empty:
        return df

    filtered = df.copy()
    if search_head:
        filtered = filtered[
            filtered["h"].str.contains(search_head, case=False, na=False)
        ]
    if search_rel:
        filtered = filtered[
            filtered["r"].str.contains(search_rel, case=False, na=False)
        ]
    if search_tail:
        filtered = filtered[
            filtered["t"].str.contains(search_tail, case=False, na=False)
        ]
    return filtered


def is_annotated(h: str, r: str, t: str) -> tuple[bool, float | None]:
    """Check if a triple is already annotated and return the confidence."""
    for a in st.session_state.annotations:
        if a["h"] == h and a["r"] == r and a["t"] == t:
            return True, a["conf"]
    return False, None


def clear_compose_selection():
    """Clear the compose triple selection (callback for button)."""
    st.session_state.compose_head = None
    st.session_state.compose_rel = None
    st.session_state.compose_tail = None


with tab1:
    st.subheader("Search Triples & Annotate")

    df = st.session_state.df_triples

    if df.empty:
        st.warning("No triples loaded. Please load a KB from the sidebar first.")
    else:
        # Search filters
        col_s1, col_s2, col_s3 = st.columns(3)
        search_head = col_s1.text_input(
            "Head (contains)", key="search_head", placeholder="e.g., person"
        )
        search_rel = col_s2.text_input(
            "Relation (contains)", key="search_rel", placeholder="e.g., has"
        )
        search_tail = col_s3.text_input(
            "Tail (contains)", key="search_tail", placeholder="e.g., location"
        )

        # View mode selection
        col_mode, col_count = st.columns([2, 1])
        view_mode = col_mode.radio(
            "View Mode",
            [
                "Most Confident",
                "Least Confident",
                "Most Frequent",
                "Annotated",
                "Search Results",
            ],
            horizontal=True,
            key="view_mode",
        )
        num_display = col_count.number_input(
            "Show top N", min_value=5, max_value=100, value=20, step=5
        )

        # Apply filters and sorting
        filtered_df = get_filtered_triples(df, search_head, search_rel, search_tail)

        if view_mode == "Most Confident":
            display_df = filtered_df.nlargest(num_display, "overall_confidence")
        elif view_mode == "Least Confident":
            display_df = filtered_df.nsmallest(num_display, "overall_confidence")
        elif view_mode == "Most Frequent":
            display_df = filtered_df.nlargest(num_display, "total_frequency")
        elif view_mode == "Annotated":
            # Show only annotated triples
            annotated_keys = {
                (a["h"], a["r"], a["t"]) for a in st.session_state.annotations
            }
            annotated_df = filtered_df[
                filtered_df.apply(
                    lambda row: (row["h"], row["r"], row["t"]) in annotated_keys, axis=1
                )
            ]
            display_df = annotated_df.head(num_display)
        else:  # Search Results
            display_df = filtered_df.head(num_display)

        st.markdown(
            f"**Showing {len(display_df)} of {len(filtered_df)} filtered triples** (Total: {len(df)})"
        )

        # Display triples with annotation buttons
        if not display_df.empty:
            for idx, row in display_df.iterrows():
                h, r, t = row["h"], row["r"], row["t"]
                conf = row["overall_confidence"]
                freq = row["total_frequency"]

                annotated, anno_conf = is_annotated(h, r, t)

                # Determine row style based on annotation
                if annotated:
                    if anno_conf == 1:
                        status_icon = ":white_check_mark:"
                    else:
                        status_icon = ":x:"
                else:
                    status_icon = ":grey_question:"

                col1, col2, col3 = st.columns([6, 1, 1], vertical_alignment="center")

                with col1:
                    st.markdown(
                        f"{status_icon} `{h}` → `{r}` → `{t}` "
                        f"<span style='color: gray; font-size: 0.85em;'>(conf: {conf:.3f}, freq: {freq})</span>",
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.button(
                        "",
                        icon=":material/check:",
                        type=(
                            "primary"
                            if not (annotated and anno_conf == 1)
                            else "secondary"
                        ),
                        key=f"valid_{idx}",
                        on_click=add_annotation,
                        args=(h, r, t, 1),
                        help="Mark as Valid",
                    )

                with col3:
                    st.button(
                        "",
                        icon=":material/close:",
                        type=(
                            "primary"
                            if not (annotated and anno_conf == 0)
                            else "secondary"
                        ),
                        key=f"invalid_{idx}",
                        on_click=add_annotation,
                        args=(h, r, t, 0),
                        help="Mark as Invalid",
                    )
        else:
            st.info("No triples match your search criteria.")

with tab2:
    st.subheader("Add New Triples & Annotate")
    st.markdown(
        "Create annotations for triples that don't exist in the KB using known concepts and relations."
    )

    concept_names = st.session_state.concept_names
    relation_names = st.session_state.relation_names

    if not concept_names or not relation_names:
        st.warning(
            "No concepts or relations loaded. Please load a KB from the sidebar first."
        )
    else:
        # Check if triple exists in KB
        def triple_exists(h: str, r: str, t: str) -> bool:
            df = st.session_state.df_triples
            if df.empty:
                return False
            return ((df["h"] == h) & (df["r"] == r) & (df["t"] == t)).any()

        col_h, col_r, col_t = st.columns(3)

        with col_h:
            head = st.selectbox(
                "Head Concept",
                options=concept_names,
                index=None,
                placeholder="Select head...",
                key="compose_head",
            )

        with col_r:
            rel = st.selectbox(
                "Relation",
                options=relation_names,
                index=None,
                placeholder="Select relation...",
                key="compose_rel",
            )

        with col_t:
            tail = st.selectbox(
                "Tail Concept",
                options=concept_names,
                index=None,
                placeholder="Select tail...",
                key="compose_tail",
            )

        # Show composed triple preview
        if head and rel and tail:
            exists_in_kb = triple_exists(head, rel, tail)
            annotated, anno_conf = is_annotated(head, rel, tail)

            st.divider()

            # Status display
            if exists_in_kb:
                if annotated:
                    status = "✅ Valid" if anno_conf == 1 else "❌ Invalid"
                    st.info(
                        f"ℹ️ This triple **exists** in the KB. Annotated as: {status}"
                    )
                else:
                    st.info("ℹ️ This triple **exists** in the KB.")
            else:
                if annotated:
                    status = "✅ Valid" if anno_conf == 1 else "❌ Invalid"
                    st.success(f"✨ New triple (not in KB). Annotated as: {status}")
                else:
                    st.warning(
                        "⚠️ This triple does **not** exist in the KB (unseen combination)."
                    )

            # Triple preview
            st.markdown(f"**Preview:** `{head}` → `{rel}` → `{tail}`")

            # Annotation buttons
            col_valid, col_invalid = st.columns(2)

            with col_valid:
                btn_type = "secondary" if (annotated and anno_conf == 1) else "primary"
                if st.button(
                    "Mark as Valid",
                    icon=":material/check:",
                    type=btn_type,
                    key="compose_valid",
                    width='stretch',
                ):
                    add_annotation(head, rel, tail, 1)
                    st.rerun()

            with col_invalid:
                btn_type = "secondary" if (annotated and anno_conf == 0) else "primary"
                if st.button(
                    "Mark as Invalid",
                    icon=":material/close:",
                    type=btn_type,
                    key="compose_invalid",
                    width='stretch',
                ):
                    add_annotation(head, rel, tail, 0)
                    st.rerun()

            # Clear button
            st.button(
                "Clear Selection",
                icon=":material/clear:",
                key="compose_clear",
                width='stretch',
                on_click=clear_compose_selection,
            )
        else:
            st.info("Select a head, relation, and tail to compose a triple.")

with tab3:
    st.subheader("Annotations")

    if not st.session_state.annotations:
        st.info(
            "No annotations yet. Use the Annotate tab to mark triples as valid or invalid."
        )
    else:
        # Summary stats
        valid_count = sum(1 for a in st.session_state.annotations if a["conf"] == 1)
        invalid_count = sum(1 for a in st.session_state.annotations if a["conf"] == 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Annotations", len(st.session_state.annotations))
        col2.metric("Valid ✅ ", valid_count)
        col3.metric("Invalid ❌", invalid_count)

        st.divider()

        # Filter annotations view
        filter_type = st.radio(
            "Filter",
            ["All", "Valid Only", "Invalid Only"],
            horizontal=True,
            key="anno_filter",
        )

        if filter_type == "Valid Only":
            display_annotations = [
                a for a in st.session_state.annotations if a["conf"] == 1
            ]
        elif filter_type == "Invalid Only":
            display_annotations = [
                a for a in st.session_state.annotations if a["conf"] == 0
            ]
        else:
            display_annotations = st.session_state.annotations

        # Display as table
        if display_annotations:
            df_annotations = pd.DataFrame(display_annotations)
            df_annotations["status"] = df_annotations["conf"].apply(
                lambda x: "✅ Valid" if x == 1 else "❌ Invalid"
            )
            st.dataframe(
                df_annotations[["h", "r", "t", "status"]],
                column_config={
                    "h": "Head",
                    "r": "Relation",
                    "t": "Tail",
                    "status": "Status",
                },
                width='stretch',
                hide_index=True,
            )

        st.divider()

        # Export options
        st.markdown("**Export Annotations**")
        col_export1, col_export2 = st.columns(2)

        with col_export1:
            csv_data = pd.DataFrame(st.session_state.annotations).to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                file_name="annotations.csv",
                mime="text/csv",
                icon=":material/download:",
                width='stretch',
            )

        with col_export2:
            json_data = pd.DataFrame(st.session_state.annotations).to_json(
                orient="records", indent=2
            )
            st.download_button(
                "Download JSON",
                json_data,
                file_name="annotations.json",
                mime="application/json",
                icon=":material/download:",
                width='stretch',
            )

        # Clear annotations button
        st.divider()
        if st.button(
            "Clear All Annotations", type="secondary", icon=":material/delete:"
        ):
            st.session_state.annotations = []
            st.rerun()
