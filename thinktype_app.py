import streamlit as st
import time, uuid, difflib, io
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime

    # ---------- CONFIG ----------
    st.set_page_config(page_title="ThinkType – Typing Personality", layout="centered")
    st.markdown("<h1 style='text-align:center'>🧠 ThinkType</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center'>Discover your mental state through how you type</h4>", unsafe_allow_html=True)
    st.write("📱 Works best on desktop for now.")
    st.divider()

    # ---------- DEFAULT SESSION KEYS ----------
    if "init" not in st.session_state:
        st.session_state.update({
            "test_started": False,
            "start_time": None,
            "last_time": None,
            "hesitation_count": 0,
            "text_input": "",
            "sessions": []
        })

    # ---------- USER NAME ----------
    user_name = st.text_input("Enter your name (optional):", key="user_name")

    # ---------- SETTINGS ----------
    st.sidebar.header("⚙️ Settings")
    hesitation_threshold = st.sidebar.slider("Hesitation threshold (s)", 0.5, 3.0, 1.5, 0.1)
    show_live_stats = st.sidebar.toggle("Show live stats", True)

    # ---------- SAMPLE SENTENCES ----------
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Streamlit makes data apps incredibly fast to build.",
        "Practice makes perfect, so keep typing to improve.",
    ]
    if st.sidebar.button("🎲 Random sentence"):
        st.session_state["random_idx"] = uuid.uuid4().int
    sentence = st.selectbox("Choose a sentence:", sentences, key=st.session_state.get("random_idx", 0))

    # ---------- START BUTTON ----------
    if not st.session_state.test_started:
        st.button("🚀 Start Typing Test", on_click=lambda: st.session_state.update({
            "test_started": True,
            "start_time": None,
            "last_time": None,
            "hesitation_count": 0,
            "text_input": ""
        }))
        st.stop()

    # ---------- TEST AREA ----------
    st.info(f"👉 Type this sentence and press **Submit** when done:

**{sentence}**")
    st.session_state.text_input = st.text_input("📝 Start typing here:", value=st.session_state.text_input, key="text_input_main")

    # ---------- TIMING & HESITATION ----------
    if st.session_state.text_input:
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()
        now = time.time()
        if st.session_state.last_time and now - st.session_state.last_time > hesitation_threshold:
            st.session_state.hesitation_count += 1
        st.session_state.last_time = now

    # ---------- LIVE FEEDBACK ----------
    if show_live_stats:
        with st.expander("📊 Live stats"):
            chars = len(st.session_state.text_input)
            elapsed = 0 if st.session_state.start_time is None else time.time() - st.session_state.start_time
            st.write(f"Characters typed: **{chars}**")
            st.write(f"Time elapsed: **{elapsed:.1f} s**")
            st.write(f"Hesitations: **{st.session_state.hesitation_count}**")

    # ---------- PROGRESS BAR ----------
    st.progress(min(len(st.session_state.text_input) / len(sentence), 1.0))

    # ---------- MODEL (cached) ----------
    @st.cache_resource
    def load_model():
        data = {
            "avg_key_delay":[0.15,0.35,0.25,0.10,0.5,0.18],
            "total_time":[20,45,30,15,50,22],
            "hesitation_count":[1,5,3,0,6,2],
            "prediction":["Focused","Stressed","Neutral","Focused","Stressed","Focused"]
        }
        df = pd.DataFrame(data)
        X = df[["avg_key_delay","total_time","hesitation_count"]]
        y = df["prediction"]
        model = RandomForestClassifier(random_state=42)
        model.fit(X,y)
        return model
    model = load_model()

    # ---------- SUBMIT ----------
    if st.button("✅ Submit Typing"):
        if not st.session_state.text_input:
            st.warning("Please type something before submitting.")
            st.stop()

        total_time = time.time() - st.session_state.start_time
        avg_delay = total_time / len(st.session_state.text_input)
        hesitations = st.session_state.hesitation_count
        wpm = len(st.session_state.text_input.split()) / total_time * 60
        cpm = len(st.session_state.text_input) / total_time * 60
        accuracy = difflib.SequenceMatcher(None, st.session_state.text_input, sentence).ratio()*100
        personality = model.predict(np.array([[avg_delay,total_time,hesitations]]))[0]

        # Personality descriptions
        descriptions = {
            "Focused":"🧠 **Focused** – Consistent typing speed and low hesitation show calm, sustained attention.",
            "Stressed":"😬 **Stressed** – Irregular typing with pauses may reflect distraction or stress.",
            "Neutral":"😐 **Neutral** – Average typing rhythm indicates neither high focus nor high stress."
        }

        # Results card
        with st.container():
            st.subheader("🔍 Results")
            st.markdown(descriptions.get(personality,""))
            col1,col2,col3,col4,col5,col6 = st.columns(6)
            col1.metric("Delay (s)",f"{avg_delay:.3f}")
            col2.metric("Time (s)",f"{total_time:.1f}")
            col3.metric("Hesitations",hesitations)
            col4.metric("WPM",f"{wpm:.1f}")
            col5.metric("CPM",f"{cpm:.0f}")
            col6.metric("Accuracy %",f"{accuracy:.1f}")

        # Save to history
        st.session_state.sessions.append({
            "timestamp": datetime.now(),
            "name": user_name or "Anonymous",
            "delay": avg_delay,
            "time": total_time,
            "hesitations": hesitations,
            "wpm": wpm,
            "cpm": cpm,
            "accuracy": accuracy,
            "personality": personality
        })

        st.balloons()

        # Reset for another run
        st.session_state.update({
            "test_started": False,
            "start_time": None,
            "last_time": None,
            "hesitation_count": 0,
            "text_input": ""
        })
        st.stop()

    # ---------- SESSION HISTORY ----------
    if st.session_state.sessions:
        st.subheader("📜 Session History")
        hist_df = pd.DataFrame(st.session_state.sessions)
        st.dataframe(hist_df[["timestamp","name","wpm","accuracy","hesitations","personality"]], use_container_width=True)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV history", csv, "typing_history.csv")

        if st.checkbox("Show WPM & Accuracy trend chart"):
            chart_df = hist_df.reset_index()
            st.line_chart(chart_df.set_index("index")[["wpm","accuracy"]])

    # ---------- RESET HISTORY ----------
    if st.sidebar.button("Reset all history"):
        st.session_state.sessions = []
        st.toast("History cleared!")
