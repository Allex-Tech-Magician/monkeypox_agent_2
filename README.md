# Mpox Buddy: chat and voice agent

Says hi, chats normally, and checks a skin photo for monkeypox with your three trained models.
Educational use only. It is not a medical diagnosis.

## Files (4 code files)

|File|Job|
|-|-|
|`app.py`|The chat page (Streamlit)|
|`detector.py`|The three models, image checks, prediction, majority vote|
|`agent.py`|The LLM chat (LangChain + Groq), voice-to-text, text-to-speech, languages|
|`ui.py`|The look: theme, animated result card, speaking orb, thinking dots|

The folder `.streamlit` holds `config.toml`, the dark theme. Keep it, and always run the app from inside this folder.

## Setup (Windows, Command Prompt)

1. Put this folder anywhere, for example `E:\\\\monkeypox\\\_agent`.
2. Copy your three weight files into the `models` folder: `cnn\\\_model.pth`, `resnext\\\_model.pth`, `seresnext\\\_model.pth`.
3. Copy your old `.env` into this folder (it has your Groq key and working model name). Or copy `.env.example` to `.env` and edit it.
4. Reuse your existing environment and add the one new package:

```cmd
E:\\\\monkeypox\\\_langchain\\\_app\\\\.venv\\\\Scripts\\\\activate.bat
uv pip install gTTS
cd /d E:\\\\monkeypox\\\_agent
streamlit run app.py
```

(For a fresh environment instead: `uv venv --python 3.11`, activate it, then `uv pip install -r requirements.txt`.)

## How the voice works

* **You talk:** tap the microphone in the chat box, speak, then send. Groq Whisper turns your voice into text. It is push-to-talk, not always-listening.
* **It talks back:** replies are spoken with Groq's Orpheus voice, which sounds natural and human. Pick a voice in the sidebar.
* **Cost note:** Orpheus is priced per character (about $22 per million characters, roughly one cent per reply). If it is unavailable on your account, the app falls back to a basic gTTS voice and tells you why.
* **First sound:** browsers block sound until you click something, so press "Hear me say hi" to hear the greeting.

## What it can do

* Chat normally, talk with the mic, or attach a skin photo.
* Quick chips: check a photo, symptom check, what is mpox, stay safe.
* A clear answer: the result card leads with a plain-language headline ("Looks like monkeypox", "Leaning not monkeypox, but unsure", and so on), an agreement level, what it means, and what to do next. It never claims a diagnosis: a photo cannot confirm monkeypox.
* Ask about your result: after a photo, chips such as "Why this result?" and "What next?" appear, and you can type or say any question. The agent knows the numbers, the verdict and how the app works. It remembers up to three photos.
* Animated result card: ring, per-model bars, vote dots, and a warning when the models disagree.
* Download a plain-text summary of any photo result.
* Languages: English, Tamil, Hindi (pick in the sidebar). Replies and speech recognition follow the choice. The natural voice is English-only, so Tamil and Hindi use the basic gTTS voice. Please have a native speaker check the Tamil and Hindi greetings before presenting.

## Using it

* Type, or tap the microphone in the chat box and talk (allow microphone access in the browser).
* Tap the paperclip to attach a JPG or PNG skin photo. The agent checks it with the three models, votes, and explains the result.
* The sidebar has a switch for spoken replies, model status, Reload models and New chat.

## Troubleshooting

* Sidebar shows a model with a cross: read the message under "Model errors" (missing file, or wrong checkpoint).
* "GROQ\_API\_KEY is missing": check `.env` sits next to `app.py`, then restart Streamlit.
* "model was not found": set `GROQ\\\_MODEL` in `.env` to a model your account can use.
* Voice sounds robotic: the natural voice failed and the basic one took over. The note under the reply says why (key, plan or rate limit).
* No voice reply at all: both voices need internet. The text reply still appears.
* Voice input fails: check microphone permission, internet and `GROQ\\\_STT\\\_MODEL`.

\## Screenshots



!\[Mpox Buddy home screen](screenshots/home.png)



!\[Photo result](screenshots/photo-result.png)



!\[Detailed result](screenshots/result-details.png)

