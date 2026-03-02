import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import streamlit as st

from app.utils import (
    check_cache_status,
    generate_concept_table,
    get_cache_dir,
    get_summary,
    index_kb_concepts,
    load_pkl,
)


# get arg
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="Knowledge Base Path", default=None)
    parser.add_argument("--id", type=str, help="Knowledge Base ID", default=None)
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to save/load cached summary",
        default=".",
    )
    args = parser.parse_args()
    return args


args = parse_args()
print("Args:", args)


# config
st.set_page_config(layout="wide", page_title="KB Dashboard")
page_explore = st.Page("explore.py", title="KB Explorer", icon=":material/graph_3:")
page_annotate = st.Page("annotate.py", title="Human Feedback", icon=":material/rule:")
pg = st.navigation([page_explore, page_annotate], position="hidden")


# style
css = """
<style>
.stSidebar .stVerticalBlock {
    gap: .5rem;
}
.stSidebar hr {
    margin-top: .5rem;
    margin-bottom: 1.5rem;
}
.stSidebar .labelTop {
    margin-top: .8rem;
    margin-bottom: -.3rem;
    font-weight: bold;
}
.stSidebar .stPageLink [data-testid="stPageLink-NavLink"] span * {
    font-size: 1.5rem;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] {
    font-size: 1.5rem;
    margin: 0 5px;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# initialization
init_state = {
    # kb selection
    "kb_path": args.path,
    "kb_id": args.id,
    "cache_dir": args.cache_dir,
    # current kb source
    "current_kb_full_path": None,
    "current_cache_dir": None,
    "cache_file_path": None,
    # current kb data
    "concept_file_map": {},
    "df_triples": pd.DataFrame(),
    "df_triples_detail": pd.DataFrame(),
    "df_concepts": pd.DataFrame(),
    "summary": {},
    "concept_names": [],
    "relation_names": [],
    "errors": {},
    # feedback
    "annotations": [],
    "anno_head": None,
    "anno_rel": None,
    "anno_tail": None,
    # compose new triples
    "compose_head": None,
    "compose_rel": None,
    "compose_tail": None,
}
for key, value in init_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# callbacks
def set_session_vars(concept_file_map, df, df_d, errors):
    st.session_state.concept_file_map = concept_file_map
    st.session_state.df_triples = df
    st.session_state.df_triples_detail = df_d

    st.session_state.summary = get_summary(concept_file_map, df, df_d)
    df_c = generate_concept_table(concept_file_map, df, df_d)
    st.session_state.df_concepts = df_c
    st.session_state.concept_names = df_c["concept"].to_list()
    st.session_state.relation_names = df_c[df_c["is_rel"] == True]["concept"].to_list()

    st.session_state.errors = errors

    st.session_state.current_kb_full_path = kb_full_path
    st.session_state.current_cache_dir = get_cache_dir(
        str(kb_full_path), st.session_state.cache_dir
    )


def load_cache():
    t0 = time.time()
    data = load_pkl(st.session_state.cache_file_path)
    set_session_vars(
        data["concept_file_map"],
        pd.DataFrame(data["triples"]),
        pd.DataFrame(data["triples_detail"]),
        data["errors"],
    )
    t1 = time.time()
    print(f"Time taken to load cache: {t1 - t0:.6f} seconds")


def scan_kb():
    concept_file_map, triples, triples_detail, errors = index_kb_concepts(
        str(kb_full_path), st.session_state.cache_dir, pbar_placeholder
    )
    set_session_vars(
        concept_file_map, pd.DataFrame(triples), pd.DataFrame(triples_detail), errors
    )


def add_top_label(text: str):
    st.html(f"<p class='labelTop'>{text}</p>")


# @st.cache_data
def list_kb_dirs(kb_path: str):
    return sorted(
        [p.name for p in Path(kb_path).iterdir() if p.is_dir() and (p / "kb").is_dir()]
    )


# sidebar
with st.sidebar:
    add_top_label("Select KB")

    kb_path = st.text_input("KB path:", placeholder="Enter path", key="kb_path")

    options = list_kb_dirs(st.session_state.kb_path) if st.session_state.kb_path else []
    if args.id in options:
        idx = options.index(args.id)
    else:
        idx = 0
        if args.id:
            st.toast(f"KB ID {args.id} not existing", icon="⚠️")
    st.selectbox("KB ID:", options, index=idx, key="kb_id")

    kb_full_path = (
        Path(st.session_state.kb_path) / st.session_state.kb_id
        if st.session_state.kb_path and st.session_state.kb_id
        else ""
    )
    cache_status, st.session_state.cache_file_path = (
        check_cache_status(str(kb_full_path), st.session_state.cache_dir)
        if kb_full_path
        else ("none", None)
    )
    if cache_status == "up-to-date" or cache_status == "outdated":
        col1, col2 = st.columns(2)
        col1.button(
            "",
            icon=":material/play_circle:",
            width="stretch",
            type="primary",
            on_click=scan_kb,
        )
        col2.button(
            "",
            icon=":material/folder_open:",
            width="stretch",
            type="primary",
            on_click=load_cache,
        )
    else:
        st.button(
            "",
            icon=":material/play_circle:",
            width="stretch",
            type="primary",
            on_click=scan_kb,
        )

    pbar_placeholder = st.empty()

    add_top_label("Currently showing")
    st.markdown(f":material/database: `{str(st.session_state.current_kb_full_path)}`")
    st.markdown(f":material/backup: `{st.session_state.current_cache_dir}`")
    st.divider()

    st.page_link(page_explore, label="KB Explorer", icon=":material/graph_3:")
    st.page_link(page_annotate, label="Human Feedback", icon=":material/rule:")
    st.divider()


pg.run()
