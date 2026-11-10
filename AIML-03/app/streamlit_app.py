"""Demo UI.   streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.recommend import Recommender


@st.cache_resource
def load():
    return Recommender()


rec = load()
st.set_page_config(page_title="Two-Stage Recommender", layout="wide")
st.title("🎬 Two-Stage Personalized Recommendation Engine")
st.caption("FAISS retrieval (two-tower, TensorFlow Recommenders) → deep ranker → Redis cache")

with st.sidebar:
    mode = st.radio("Who is the user?", ["Existing user", "New user (cold start)"])
    ranker = st.selectbox("Ranking model (A/B arms)", ["dcn", "wide_deep"],
                          format_func=lambda k: {"dcn": "B: DCN", "wide_deep": "A: Wide & Deep"}[k])
    k = st.slider("How many recommendations", 5, 20, 10)
    st.write(f"Cache backend: **{rec.cache.backend}**")

user_id, genres = None, None
if mode == "Existing user":
    ids = sorted(rec.user_map)
    user_id = st.selectbox("User ID", ids[:500], index=0)
    uidx = rec.user_map[user_id]
    hist = rec.movies.loc[sorted(rec.seen.get(uidx, []))[:15]]
    with st.expander("A few movies this user has already watched"):
        st.dataframe(hist[["title", "genres"]], use_container_width=True)
else:
    all_genres = sorted({g for s in rec._genre_sets.values() for g in s})
    genres = st.multiselect("Pick a few genres you like (optional)", all_genres)

if st.button("Recommend", type="primary"):
    out = rec.recommend(user_id, k=k, ranker=ranker, genres=genres)
    c1, c2, c3 = st.columns(3)
    c1.metric("Latency", f"{out['latency_ms']} ms")
    c2.metric("Cache", "HIT ✅" if out["cache_hit"] else "MISS")
    c3.write(f"**Strategy:** {out['strategy']}")
    st.dataframe(out["items"], use_container_width=True)
    st.caption("Click Recommend again for the same user to see the cached (much faster) response.")
