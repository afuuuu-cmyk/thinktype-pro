# -*- coding: utf-8 -*-
"""ThinkType – Clean, user‑friendly typing personality app."""

import streamlit as st
import time
import difflib
from datetime import datetime
import uuid
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="ThinkType – Typing Personality", layout="centered")

st.markdown("<h2 style='text-align:center'>ThinkType</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Discover your mental state through how you type</p>", unsafe_allow_html=True)
st.write("This demo works best on desktop.")

st.divider()

# ---------- INITIAL SESSION STATE ----------
defaults = {
    "test_started": False,
    "start_time": None,
    "last_time": None,
    "hesitation_count": 0,
    "text": "",
    "history": []
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- SETTINGS ----------
st.sidebar.header("Settings")
pause_threshold = st.sidebar.slider("Hesitation threshold (seconds)", 0.5, 3.0, 1.5, 0.1)
show_live = st.sidebar.checkbox("Show live stats", value=True)

# ---------- SAMPLE SENTENCES ----------
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Streamlit makes data apps incredibly fast to build.",
    "Practice makes perfect, so keep typing to improve."
]
if st.sidebar.button("Pick random sentence"):
    st.session_state.random_key = uuid.uuid4().int
chosen = st.selectbox("Choose a sentence to type:", sentences, key=st.session_state.get("random_key", 0))

# ---------- START TEST ----------
if not st.session_state.test_started:
    if st.button("Start typing test"):
        st.session_state.test_started = True
        st.session_state.start_time = None
        st.session_state.last_time = None
        st.session_state.hesitation_count = 0
        st.session_state.text = ""
        st.experimental_rerun()
    st.stop()

# ---------- INPUT FIELD ----------
st.info(f"Type the sentence below and press Submit when done:\n\n**{chosen}**")
st.session_state.text = st.text_input("Start typing here:", value=st.session_state.text)

# ---------- TIMING & HESITATIONS ----------
if st.session_state.text:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    now = time.time()
    if st.session_state.last_time and now - st.session_state.last_time > pause_threshold:
        st.session_state.hesitation_count += 1
    st.session_state.last_time = now

# ---------- LIVE STATS ----------
if show_live:
    with st.expander("Live stats"):
        chars = len(st.session_state.text)
        elapsed = 0 if st.session_state.start_time is None else time.time() - st.session_state.start_time
        st.write(f"Characters typed: {chars}")
        st.write(f"Time elapsed: {elapsed:.1f} s")
        st.write(f"Hesitations: {st.session_state.hesitation_count}")

# ---------- PROGRESS ----------
progress_pct = min(len(st.session_state.text) / len(chosen), 1.0)
st.progress(progress_pct)

# ---------- MODEL ----------
@st.cache_resource
def get_model():
    df = pd.DataFrame({
        "avg_delay": [0.15,0.35,0.25,0.1,0.5,0.18],
        "total_time": [20,45,30,15,50,22],
        "hesitations": [1,5,3,0,6,2],
        "label": ["Focused","Stressed","Neutral","Focused","Stressed","Focused"]
    })
    clf = RandomForestClassifier(random_state=42)
    clf.fit(df[["avg_delay","total_time","hesitations"]], df["label"])
    return clf
model = get_model()

# ---------- SUBMIT ----------
if st.button("Submit"):
    if not st.session_state.text:
        st.warning("Please type something before submitting.")
        st.stop()

    total_time = time.time() - st.session_state.start_time
    avg_delay = total_time / len(st.session_state.text)
    hes = st.session_state.hesitation_count
    words = len(st.session_state.text.split())
    wpm = words / total_time * 60
    cpm = len(st.session_state.text) / total_time * 60
    acc = difflib.SequenceMatcher(None, st.session_state.text, chosen).ratio() * 100
    personality = model.predict(np.array([[avg_delay, total_time, hes]]))[0]

    desc = {
        "Focused": "Consistent typing with low hesitation shows good focus.",
        "Stressed": "Irregular typing and pauses may indicate stress.",
        "Neutral": "Average rhythm suggests a neutral state."
    }

    st.success("Typing test complete!")
    st.write(f"Personality: **{personality}** – {desc.get(personality,'')}")
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Delay (s)", f"{avg_delay:.3f}")
    col2.metric("Time (s)", f"{total_time:.1f}")
    col3.metric("Hesitations", hes)
    col4.metric("WPM", f"{wpm:.1f}")
    col5.metric("Accuracy %", f"{acc:.1f}")

    # Save history
    st.session_state.history.append({
        "timestamp": datetime.now(),
        "delay": avg_delay,
        "time": total_time,
        "hesitations": hes,
        "wpm": wpm,
        "accuracy": acc,
        "personality": personality
    })

    # Reset for next run
    st.session_state.test_started = False
    st.experimental_rerun()

# ---------- HISTORY ----------
if st.session_state.history:
    st.subheader("Session history")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)
    csv_data = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_data, "typing_history.csv")
