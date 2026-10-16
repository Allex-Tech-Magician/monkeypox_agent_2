"""The agent's brain: chat with an LLM (LangChain + Groq), turn voice into text
(Groq Whisper) and turn replies into speech (gTTS).

No Streamlit code lives here. Settings come from the .env file next to this file.
"""
import io
import os
import re
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"  # override with GROQ_MODEL in .env
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"  # Groq speech-to-text, override with GROQ_STT_MODEL
DEFAULT_TTS_MODEL = "canopylabs/orpheus-v1-english"  # Groq text-to-speech, override with GROQ_TTS_MODEL
VOICES = ["hannah", "autumn", "diana", "austin", "daniel", "troy"]  # Orpheus English voices
_env_voice = os.getenv("GROQ_TTS_VOICE", "").strip().lower()
DEFAULT_VOICE = _env_voice if _env_voice in VOICES else "hannah"
TTS_CHAR_LIMIT = 190  # Groq allows at most 200 characters per speech request
PLACEHOLDER_KEY = "your_groq_api_key_here"
MAX_HISTORY = 12  # how many earlier messages the LLM remembers

GREETING = (
    "Hi, I'm Mpox Buddy! I can chat about anything, and I can check a skin photo for signs "
    "that look like monkeypox, also called mpox. When you're ready, tap the paperclip and upload "
    "a clear, well-lit photo of the skin area, or just talk to me with the mic. Quick note: "
    "I'm a learning tool, not a doctor, so my results are never a diagnosis."
)

GREETING_TA = (
    "வணக்கம், நான் Mpox Buddy! என்னுடன் எதைப் பற்றியும் பேசலாம், மேலும் தோல் புகைப்படத்தில் "
    "குரங்கு அம்மை (mpox) போன்ற அறிகுறிகள் தெரிகிறதா என்றும் சோதிக்கலாம். தயாரானதும், "
    "பேப்பர்கிளிப் பொத்தானை அழுத்தி, தெளிவான, நல்ல வெளிச்சத்தில் எடுத்த தோல் புகைப்படத்தைப் "
    "பதிவேற்றுங்கள், அல்லது மைக் மூலம் என்னுடன் பேசுங்கள். ஒரு சிறிய குறிப்பு: நான் ஒரு கற்றல் "
    "கருவி, மருத்துவர் அல்ல, அதனால் என் முடிவுகள் ஒருபோதும் மருத்துவ உறுதிப்படுத்தல் ஆகாது."
)

GREETING_HI = (
    "नमस्ते, मैं Mpox Buddy हूँ! आप मुझसे किसी भी विषय पर बात कर सकते हैं, और त्वचा की तस्वीर की जाँच "
    "भी करवा सकते हैं कि उसमें मंकीपॉक्स (एमपॉक्स) जैसे संकेत दिखते हैं या नहीं। जब आप तैयार हों, "
    "पेपरक्लिप दबाकर त्वचा की एक साफ़, अच्छी रोशनी वाली तस्वीर अपलोड करें, या माइक से बात करें। "
    "एक ज़रूरी बात: यह एक सीखने का टूल है, डॉक्टर नहीं, इसलिए इसके नतीजे कभी भी निदान नहीं होते।"
)

# label shown in the sidebar -> (code for speech and voice, name given to the LLM)
LANGUAGES = {
    "English": ("en", "English"),
    "தமிழ் (Tamil)": ("ta", "Tamil"),
    "हिन्दी (Hindi)": ("hi", "Hindi"),
}
LANGUAGE_NAMES = {code: name for code, name in LANGUAGES.values()}
GREETINGS = {"en": GREETING, "ta": GREETING_TA, "hi": GREETING_HI}
PERCENT_WORD = {"en": " percent", "ta": " சதவீதம்", "hi": " प्रतिशत"}


def greeting(language="en"):
    return GREETINGS.get(language, GREETING)


OFFLINE_REPLY = (
    "Sorry, I can't reach my language model right now, so I can't chat at the moment. "
    "I can still check a photo if you upload one."
)

SYSTEM_PROMPT = """You are Mpox Buddy, a warm, friendly voice and chat assistant.

What you can do:
- Chat normally about anything: small talk, questions, explanations.
- Help with a monkeypox (mpox) image check. The user attaches a skin photo with the paperclip button, and three trained AI models analyse it. The app gives you the results in LATEST RESULT. You never see the image yourself.
- Answer follow-up questions about the results: why the models disagree, what the percentages and the ring mean, what to do next, whether it could be something else, how the app works.

ABOUT THIS APP (use it when asked how things work): Three image models look at the photo: a custom CNN, a ResNeXt-50 and an SE-ResNeXt-50. They learned from one dataset of about 8,500 skin photos, half monkeypox and half other conditions. Each model gives a probability. The app takes a majority vote, shows the average probability in the ring, and judges how well the models agree with each other. The photo is analysed on the user's own computer and is never sent to the language model. The models can be overconfident, and they may do worse on photos that look different from what they learned from. The percentages are not calibrated and are not the chance that the person has the disease.

Rules:
1. If the user wants a monkeypox test or check, or asks whether something is monkeypox, ask them to upload a clear, well-lit photo of the skin area with the paperclip button. Do not judge from a text description.
2. When talking about a result, use ONLY the numbers and wording in LATEST RESULT. Never invent or change them, and never describe what the image looks like.
3. Answer directly. When asked whether it is monkeypox, give the plain-language lean from LATEST RESULT first, for example "my models lean towards not monkeypox, but they disagree, so I can't be sure", then the reason. Never say the person definitely has, or definitely does not have, monkeypox. A photo cannot confirm it, and only a healthcare professional with an exam and tests can. An Others result does not rule out monkeypox or other conditions.
4. For health worries, suggest seeing a healthcare professional. For severe symptoms such as trouble breathing, a high fever, lesions that spread fast or are very painful, or symptoms near the eyes, tell the user to seek urgent care.
5. For general mpox facts, stick to well-established basics (cause, common symptoms, how it spreads, prevention) and say so if you are unsure. Never invent studies, statistics or links.
6. Talk like a real person on a phone call: short sentences, contractions, a warm and natural tone. Usually 2 to 5 sentences, then, when it fits, ask one short follow-up question. No lists, no markdown, no emojis, no asterisks. Say numbers the way you would say them aloud and round them, for example "about ninety-eight percent". Never read out file names or technical details.
7. The user's words may come from speech recognition. If a message is unclear or looks garbled, kindly ask them to say it again.
8. Always reply in {reply_language}, because that is the language the user chose. Keep the name Mpox Buddy and numbers as they are.
9. Vary your wording so you do not sound scripted. Be kind. You may joke lightly, but never about someone's health worries.
10. If the user asks for a symptom check, ask ONE short question at a time: how long the rash has been there, where it is, any fever or swollen glands, recent close contact or travel, and pain or itching. After three to five answers, give general guidance only: this cannot diagnose anything, they should see a healthcare professional, and you should mention the urgent-care warning signs.
11. If the user has uploaded more than one photo, LATEST RESULT lists them. Say which photo you mean when it matters.

LATEST RESULT:
{latest_result}"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{user_input}"),
    ]
)


class AgentError(Exception):
    """Something went wrong with the LLM or voice service. The message is safe to show."""


# ---------------------------------------------------------------- settings
def _float_env(name, default):
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _settings():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == PLACEHOLDER_KEY:
        raise AgentError("GROQ_API_KEY is missing. Copy .env.example to .env and add your Groq key.")
    return {
        "key": key,
        "model": os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL,
        "temperature": _float_env("LLM_TEMPERATURE", 0.4),
    }


def settings_problem():
    """None if the API key is set, otherwise a message for the sidebar."""
    try:
        _settings()
    except AgentError as exc:
        return str(exc)
    return None


@lru_cache(maxsize=2)
def _build_llm(key, model, temperature):
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise AgentError("langchain-groq is not installed. Run: uv pip install -r requirements.txt") from exc
    return ChatGroq(model=model, temperature=temperature, api_key=key, timeout=60, max_retries=2)


# ---------------------------------------------------------------- helpers
def clean_text(text):
    """Remove hidden <think>...</think> reasoning that some models print."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def friendly_error(exc):
    """Turn provider and network exceptions into one short message (matched by class name)."""
    name = type(exc).__name__
    detail = str(exc)[:200]
    if "terms acceptance" in str(exc).lower():
        return (
            "Groq needs the account owner to accept this voice model's terms once. Open "
            "https://console.groq.com/playground?model=canopylabs/orpheus-v1-english, "
            "sign in, and accept the terms."
        )
    if name == "AuthenticationError":
        return "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
    if name == "PermissionDeniedError":
        return f"Groq denied permission (check your account, key or model access). Details: {detail}"
    if name == "RateLimitError":
        return "Groq's rate limit was reached. Wait a moment and try again."
    if name in ("APIConnectionError", "ConnectError"):
        return "Could not reach Groq. Check your internet connection."
    if name in ("APITimeoutError", "TimeoutException", "ReadTimeout"):
        return "Groq took too long to answer. Try again."
    if name == "NotFoundError":
        return f"That model was not found on Groq. Check GROQ_MODEL in .env. ({detail})"
    if name in ("ImportError", "ModuleNotFoundError"):
        return "A required package is missing. Run: uv pip install -r requirements.txt"
    return f"The AI request failed ({name}): {detail}"


def _format_one(results):
    lines = [
        f"Plain-language lean: {results['verdict']['headline']} "
        f"(agreement between models: {results['verdict']['agreement']}; {results['verdict']['note']})"
        if results.get("verdict")
        else f"Final classification (majority vote): {results['final']}",
        f"Majority vote: {results['final']}, {results['agreement']} models agree",
        f"Average Monkeypox probability across models: {results['avg_monkeypox_pct']}%",
    ]
    for m in results["models"]:
        lines.append(
            f"- {m['name']}: {m['label']} (Monkeypox {m['monkeypox_pct']}%, Others {m['others_pct']}%)"
        )
    return "\n".join(lines)


def format_result(results):
    """The image result(s) as plain text for the system prompt. Accepts one result or a list."""
    if not results:
        return "none yet (the user has not uploaded a photo)"
    if isinstance(results, dict):
        return _format_one(results)
    blocks = []
    for number, item in enumerate(results, start=1):
        tag = "LATEST photo" if number == len(results) else "Earlier photo"
        blocks.append(f"Photo {number} of {len(results)} ({tag}):\n{_format_one(item)}")
    return "\n\n".join(blocks)


def _to_messages(history):
    """[{'role': 'user'|'assistant', 'content': str}] -> LangChain messages."""
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in history[-MAX_HISTORY:]
    ]


def _ask(user_input, history, last_result, fallback, language="en"):
    """Run prompt | LLM | parser. Returns (text, warning); warning is None on success."""
    try:
        s = _settings()
        chain = _PROMPT | _build_llm(s["key"], s["model"], s["temperature"]) | StrOutputParser()
        raw = chain.invoke(
            {
                "latest_result": format_result(last_result),
                "reply_language": LANGUAGE_NAMES.get(language, "English"),
                "history": _to_messages(history),
                "user_input": user_input,
            }
        )
        text = clean_text(raw or "")
        if not text:
            raise AgentError("The language model returned an empty answer.")
        return text, None
    except AgentError as exc:
        return fallback, str(exc)
    except Exception as exc:  # provider and network errors vary by package version
        return fallback, friendly_error(exc)


# ---------------------------------------------------------------- the two things the agent says
def chat_reply(history, user_text, last_result=None, language="en"):
    """A normal conversation turn. Returns (text, warning)."""
    return _ask(user_text, history, last_result, OFFLINE_REPLY, language)


def fallback_explanation(results):
    """Used when the LLM is unavailable: a plain explanation built from the real numbers."""
    verdict = results.get("verdict")
    if verdict:
        opening = f"My models say: {verdict['headline'].lower()}. {verdict['note']}"
    elif results["final"] == "Tie":
        opening = "The three models did not agree, so the result is uncertain."
    else:
        opening = f"The majority vote says {results['final']}, with {results['agreement']} models agreeing."
    extra = (
        " That does not rule out monkeypox or other conditions."
        if results["final"] == "Others"
        else ""
    )
    return (
        f"{opening} The average Monkeypox probability across the models was "
        f"{results['avg_monkeypox_pct']} percent.{extra} A photo cannot confirm monkeypox, so please see a "
        "healthcare professional if you are worried."
    )


def explain_result(results, user_text, history, language="en", earlier=None):
    """Explain a fresh image result like a person would. Returns (text, warning)."""
    said = f' The user also said: "{user_text}".' if user_text else ""
    instruction = (
        "The user just uploaded a skin photo and the three models have analysed it." + said +
        " Talk to them in 4 to 6 short spoken sentences, using only LATEST RESULT. "
        "First give a direct answer to the question on their mind, 'is it monkeypox?': say which way my models lean "
        "(use the plain-language lean) and how sure they are, in one or two sentences. "
        "If the models disagree, say so plainly and what that means. Mention the average Monkeypox probability rounded "
        "to a whole number. Say clearly that a photo cannot confirm monkeypox and a healthcare professional is the only "
        "way to know, and give the most useful next step. If the lean is not monkeypox, say that does not rule it out. "
        "Finish by asking one short question, for example whether they want to hear why the models disagree, "
        "what to do next, or whether they have other symptoms."
    )
    all_results = list(earlier or []) + [results]
    return _ask(instruction, history, all_results, fallback_explanation(results), language)


# ---------------------------------------------------------------- voice
def transcribe(audio_bytes, language=None):
    """Speech to text with Groq Whisper. Raises AgentError with a friendly message.

    `language` is a code such as en, ta or hi. If omitted, GROQ_STT_LANGUAGE (default en) is used.
    """
    s = _settings()
    stt_model = os.getenv("GROQ_STT_MODEL", "").strip() or DEFAULT_STT_MODEL
    try:
        from groq import Groq

        client = Groq(api_key=s["key"], timeout=60)
        options = {}
        language = (language or os.getenv("GROQ_STT_LANGUAGE", "en")).strip().lower()
        if language and language != "auto":  # a fixed language avoids wrong auto-detection
            options["language"] = language
        result = client.audio.transcriptions.create(
            file=("voice.wav", audio_bytes), model=stt_model, **options
        )
    except Exception as exc:
        raise AgentError(friendly_error(exc)) from exc
    text = (getattr(result, "text", "") or "").strip()
    if not text:
        raise AgentError("I could not hear any speech in that recording.")
    return text


def clean_for_speech(text, language="en"):
    """Remove markdown symbols and emojis, and say % as a word, so the voice reads it well."""
    text = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf]", "", text)
    text = text.replace("%", PERCENT_WORD.get(language, " percent"))
    text = re.sub(r"[*_#`>|]", "", text)
    return re.sub(r"\s+", " ", text).strip()[:600]


def split_for_tts(text, limit=TTS_CHAR_LIMIT):
    """Split text into chunks of at most `limit` characters, breaking at sentence ends."""
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sentence) > limit:  # a very long sentence: cut it at the last space
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks


def _read_wav(blob):
    """Read a WAV file: returns (channels, bytes_per_sample, sample_rate, pcm_bytes).

    Streamed audio often declares its length as 'unknown' (0xFFFFFFFF), which Python's own
    wave module cannot write back out. So this reads the chunks by hand and, when the
    declared size is impossible, simply takes the rest of the file as audio.
    """
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(blob):
        chunk_id = blob[pos:pos + 4]
        size = struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            _, channels, rate, _, _, bits = struct.unpack("<HHIIHH", blob[body:body + 16])
            fmt = (channels, bits // 8, rate)
        elif chunk_id == b"data":
            end = body + size if body + size <= len(blob) else len(blob)
            pcm = blob[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("incomplete WAV file")
    return (*fmt, pcm)


def _merge_wavs(parts):
    """Join WAV clips (same format) into one clip with a correct header."""
    try:
        decoded = [_read_wav(blob) for blob in parts]
    except ValueError:
        if len(parts) == 1:
            return parts[0]  # a single clip we cannot parse is still played as it is
        raise
    channels, width, rate, _ = decoded[0]
    pcm = b"".join(d[3] for d in decoded)
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * channels * width, channels * width, width * 8)
        + b"data" + struct.pack("<I", len(pcm))
    )
    return header + pcm


def _groq_speech(text, voice):
    """Natural voice: Groq Orpheus. Chunks are generated in parallel, then joined."""
    s = _settings()
    model = os.getenv("GROQ_TTS_MODEL", "").strip() or DEFAULT_TTS_MODEL
    from groq import Groq

    client = Groq(api_key=s["key"], timeout=60)

    def one(chunk):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "part.wav")
            client.audio.speech.create(
                model=model, voice=voice, input=chunk, response_format="wav"
            ).write_to_file(path)
            with open(path, "rb") as handle:
                return handle.read()

    chunks = split_for_tts(text)
    with ThreadPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(one, chunks))  # keeps the original order
    return _merge_wavs(parts)


def _gtts_speech(text, language="en"):
    """Basic backup voice (Google gTTS, needs internet). Returns mp3 bytes; raises on failure."""
    from gtts import gTTS

    buffer = io.BytesIO()
    gTTS(text=text, lang=language).write_to_fp(buffer)
    return buffer.getvalue()


def _gtts_reason(exc):
    if isinstance(exc, ImportError):
        return "gTTS is not installed. Run: uv pip install gTTS"
    return f"{type(exc).__name__}: {str(exc)[:120]}"


def audio_seconds(audio, mime):
    """Length of a spoken clip in seconds (exact for WAV, a rough guess for mp3)."""
    try:
        if mime == "audio/wav":
            channels, width, rate, pcm = _read_wav(audio)
            return len(pcm) / (rate * channels * width)
    except Exception:
        pass
    return max(2.0, len(audio) / 4000)


def speak(text, voice=DEFAULT_VOICE, language="en"):
    """Text to speech. Returns (audio_bytes, mime_type, note).

    English uses the natural Groq voice, then falls back to gTTS. Tamil and Hindi use gTTS,
    because the natural voice is English-only. `note` explains any downgrade so the app can
    show it; it is None when everything worked.
    """
    spoken = clean_for_speech(text, language)
    if not spoken:
        return None, None, None
    note = None
    if language == "en":
        try:
            return _groq_speech(spoken, voice), "audio/wav", None
        except AgentError as exc:
            reason = str(exc)
        except Exception as exc:
            reason = friendly_error(exc)
        note = f"Natural voice unavailable: {reason} Using the basic voice instead."
    try:
        return _gtts_speech(spoken, language), "audio/mp3", note
    except Exception as exc:
        prefix = f"{note} " if note else ""
        return None, None, f"{prefix}The basic voice failed too ({_gtts_reason(exc)}), so you only get text."
