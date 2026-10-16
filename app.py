"""Mpox Buddy: a chat and voice agent that can also check a skin photo for monkeypox.

Run with:  streamlit run app.py   (from inside this folder)
Flow: greeting -> user types, speaks, taps a chip or attaches a photo -> agent replies and speaks.
"""
import streamlit as st

import agent
import detector
import ui

st.set_page_config(page_title="Mpox Buddy", page_icon="🩺", layout="centered")

AVATARS = {"assistant": "🩺", "user": "🧑"}
QUICK_ACTIONS = [
    ("📷 Check a photo", "I'd like to check a skin photo for monkeypox."),
    ("🩺 Symptom check", "Can we do a quick symptom check?"),
    ("❓ What is mpox?", "What is mpox and how does it spread?"),
    ("🛡 Stay safe", "How can I protect myself and others from mpox?"),
]


FOLLOW_UPS = [
    ("🤔 Why this result?", "Why did the models give this result? Please explain it simply."),
    ("🧭 What next?", "What should I do next based on this result?"),
    ("📊 What do the numbers mean?", "What do the percentages and the ring mean?"),
    ("🔍 Something else?", "Could this be something other than monkeypox?"),
]


class QueuedPrompt:
    """A tapped quick-action behaves like a typed message."""

    def __init__(self, text):
        self.text, self.files, self.audio = text, [], None


@st.cache_resource(show_spinner="Loading the three trained models...")
def load_detector():
    """Load the three models once and keep them in memory between reruns."""
    return detector.load_models()


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "greeting": True}]
    st.session_state.setdefault("results", [])  # newest last, at most 3


def language_code():
    return agent.LANGUAGES[st.session_state.get("lang_label", "English")][0]


def thinking(label):
    """Show animated dots in a slot. Call .empty() on the slot when done."""
    slot = st.empty()
    slot.markdown(ui.thinking_html(label), unsafe_allow_html=True)
    return slot


def play_voice(text):
    """Speak `text`. Autoplay works because the user just interacted with the page."""
    audio, mime, note = agent.speak(
        text, st.session_state.get("voice", agent.DEFAULT_VOICE), language_code()
    )
    if note:
        st.caption(note)
    if audio:
        st.markdown(ui.orb_html(agent.audio_seconds(audio, mime)), unsafe_allow_html=True)
        st.audio(audio, format=mime, autoplay=True)


def history_for_llm(messages):
    """Plain text history for the LLM (a photo is mentioned in words, not sent)."""
    history = []
    for m in messages:
        text = agent.greeting("en") if m.get("greeting") else m.get("text", "")
        if m.get("image"):
            text = (text + " " if text else "") + "[the user uploaded a skin photo]"
        history.append({"role": m["role"], "content": text})
    return history


def show_message(msg, index):
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg.get("image"):
            st.image(msg["image"], width=240)
        text = agent.greeting(language_code()) if msg.get("greeting") else msg.get("text", "")
        if text:
            st.markdown(("🎤 " if msg.get("voice") else "") + text)
        if msg.get("result"):
            st.markdown(ui.result_card_html(msg["result"]), unsafe_allow_html=True)
            st.download_button(
                "⬇ Download summary",
                data=ui.report_text(msg["result"]),
                file_name="mpox_check_summary.txt",
                mime="text/plain",
                key=f"download_{index}",
            )


def handle_submission(prompt, models, device):
    """One user turn: voice -> text, then either a photo check or a normal chat reply."""
    code = language_code()
    typed = (prompt.text or "").strip()
    files = list(prompt.files or [])
    image_bytes = files[0].getvalue() if files else None

    spoken = ""
    if prompt.audio is not None:
        slot = thinking("Listening...")
        try:
            spoken = agent.transcribe(prompt.audio.getvalue(), code)
        except agent.AgentError as exc:
            st.warning(f"Voice input problem: {exc}")
        slot.empty()
    user_text = " ".join(part for part in (typed, spoken) if part)
    if not user_text and image_bytes is None:
        return  # nothing usable was submitted

    history = history_for_llm(st.session_state.messages)  # before adding this turn
    user_msg = {"role": "user", "text": user_text, "image": image_bytes, "voice": bool(spoken)}
    st.session_state.messages.append(user_msg)
    show_message(user_msg, len(st.session_state.messages))

    text, warning, result = "", None, None
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        slot = thinking("Thinking...")
        if image_bytes is not None:
            if not models:
                text = "I can't check photos right now because my models did not load. The sidebar shows why."
            else:
                slot.markdown(ui.thinking_html("Checking your photo with three AI models..."), unsafe_allow_html=True)
                try:
                    result = detector.analyze(models, image_bytes, device)
                except detector.ImageError as exc:
                    text = f"I couldn't use that image. {exc} Could you try another photo?"
                except Exception as exc:  # keep the chat alive whatever happens
                    text = f"Something went wrong while checking the photo: {exc}"
                if result is not None:
                    earlier = list(st.session_state.results)
                    st.session_state.results = (earlier + [result])[-3:]
                    slot.markdown(ui.thinking_html("Writing up the result..."), unsafe_allow_html=True)
                    text, warning = agent.explain_result(result, user_text, history, code, earlier)
        else:
            text, warning = agent.chat_reply(history, user_text, st.session_state.results, code)
        slot.empty()

        if warning:
            st.warning(warning)
        st.markdown(text)
        if result is not None:
            st.markdown(ui.result_card_html(result), unsafe_allow_html=True)
            st.download_button(
                "⬇ Download summary",
                data=ui.report_text(result),
                file_name="mpox_check_summary.txt",
                mime="text/plain",
                key=f"download_{len(st.session_state.messages)}",
            )
        if st.session_state.get("speak", True):
            voice_slot = thinking("Getting my voice ready...")
            play_voice(text)
            voice_slot.empty()

    st.session_state.messages.append({"role": "assistant", "text": text, "result": result})


def render_sidebar(models, errors, device):
    st.sidebar.header("Settings")
    st.sidebar.toggle("Speak replies aloud", value=True, key="speak")
    st.sidebar.selectbox("Language", list(agent.LANGUAGES), key="lang_label")
    if language_code() != "en":
        st.sidebar.caption("The natural voice is English-only, so this language uses the basic voice.")
    st.sidebar.selectbox(
        "Voice", agent.VOICES, index=agent.VOICES.index(agent.DEFAULT_VOICE), key="voice"
    )
    problem = agent.settings_problem()
    if problem:
        st.sidebar.warning(problem)
    if st.get_option("theme.base") != "dark":
        st.sidebar.caption("Tip: run from the project folder so the dark theme file is picked up.")

    st.sidebar.subheader("Models")
    st.sidebar.caption(f"Device: {device}")
    for name in detector.MODEL_FILES:
        st.sidebar.write(("✅ " if name in models else "❌ ") + name)
    if errors:
        with st.sidebar.expander("Model errors"):
            for name, message in errors.items():
                st.error(f"{name}: {message}")

    if st.sidebar.button("Reload models"):
        load_detector.clear()
        st.rerun()
    if st.sidebar.button("New chat"):
        del st.session_state["messages"]
        st.session_state.results = []
        st.rerun()


def render_quick_actions(actions, prefix):
    columns = st.columns(len(actions))
    for column, (label, text) in zip(columns, actions):
        with column:
            if st.button(label, key=f"{prefix}_{label}"):
                st.session_state["queued"] = text
                st.rerun()


def main():
    st.markdown(ui.CSS_TAG, unsafe_allow_html=True)
    models, errors, device = load_detector()
    init_state()
    render_sidebar(models, errors, device)

    label = st.session_state.get("lang_label", "English")
    st.markdown(
        ui.hero_html(len(models), len(detector.MODEL_FILES), st.session_state.get("speak", True), label),
        unsafe_allow_html=True,
    )

    for index, msg in enumerate(st.session_state.messages):
        show_message(msg, index)

    if len(st.session_state.messages) == 1:  # only the greeting so far
        if st.button("🔊 Hear me say hi"):  # a click lets the browser play sound
            voice_slot = thinking("Getting my voice ready...")
            play_voice(agent.greeting(language_code()))
            voice_slot.empty()
    if st.session_state.results:  # there is a result to talk about
        st.caption("Ask me about your result:")
        render_quick_actions(FOLLOW_UPS, "followup")
    elif len(st.session_state.messages) <= 3:
        render_quick_actions(QUICK_ACTIONS, "quick")

    prompt = st.chat_input(
        "Type, tap the mic to talk, or attach a skin photo with the paperclip",
        accept_file=True,
        file_type=["jpg", "jpeg", "png"],
        accept_audio=True,
    )
    queued = st.session_state.pop("queued", None)
    if queued:
        prompt = QueuedPrompt(queued)
    if prompt is not None:
        handle_submission(prompt, models, device)


main()
