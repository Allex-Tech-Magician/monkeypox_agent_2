"""Look and feel: CSS, animated result card, speaking orb, thinking dots, summary text.

Pure Python, no Streamlit import. Every function returns a string, so it is easy to test.
HTML pieces are built on ONE line with no indentation, because Streamlit's markdown treats
indented lines or blank lines as code.
"""
from html import escape

AMBER, BLUE, GREY = "#f59e0b", "#60a5fa", "#94a3b8"
GLOW = {AMBER: "rgba(245,158,11,.14)", BLUE: "rgba(96,165,250,.14)", GREY: "rgba(148,163,184,.14)"}

CSS = """
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
.stApp {
  background:
    radial-gradient(900px 520px at 8% -8%, rgba(139,92,246,.32), transparent 60%),
    radial-gradient(800px 460px at 100% 0%, rgba(34,211,238,.20), transparent 55%),
    radial-gradient(700px 500px at 50% 112%, rgba(236,72,153,.15), transparent 60%),
    #0a0f1f;
}
.block-container { max-width: 860px; padding-top: 1.2rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(12,18,40,.88); backdrop-filter: blur(14px); border-right: 1px solid rgba(255,255,255,.08); }

.hero { text-align: center; padding: 6px 0 14px; }
.hero h1 {
  font-size: 3.1rem; font-weight: 800; margin: 0; letter-spacing: -1px; padding: 0;
  background: linear-gradient(90deg, #a78bfa, #22d3ee, #f472b6, #a78bfa); background-size: 300% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shine 8s linear infinite;
}
.hero p { color: #9fb0d6; margin: 4px 0 14px; font-size: 1.02rem; }
.chips { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.chip { padding: 5px 13px; border-radius: 999px; font-size: .8rem; color: #d6e0ff; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); backdrop-filter: blur(6px); }
.chip .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; animation: blink 2s ease-in-out infinite; }
.tiny { margin-top: 10px; font-size: .75rem; color: #7f90b8; }

[data-testid="stChatMessage"] {
  background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.10); border-radius: 18px;
  padding: 14px 16px; margin-bottom: 12px; backdrop-filter: blur(10px); animation: rise .45s ease both;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg, rgba(139,92,246,.24), rgba(34,211,238,.12));
}
[data-testid="stImage"] img { border-radius: 16px; border: 1px solid rgba(255,255,255,.16); }
audio { width: 100%; border-radius: 999px; }

.stButton > button, .stDownloadButton > button {
  width: 100%; border-radius: 999px; border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.06);
  color: #e8eefc; transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px); border-color: #a78bfa; box-shadow: 0 8px 22px rgba(139,92,246,.38);
}
[data-testid="stChatInput"] { border-radius: 18px; border: 1px solid rgba(167,139,250,.5); box-shadow: 0 0 26px rgba(139,92,246,.22); }

.rc {
  border-radius: 20px; padding: 18px; margin-top: 10px; color: #e8eefc;
  background: linear-gradient(160deg, rgba(22,31,64,.96), rgba(13,19,42,.96)); border: 1px solid rgba(255,255,255,.13);
  box-shadow: 0 12px 40px rgba(0,0,0,.38); animation: rise .6s ease both;
}
.rc-top { display: flex; gap: 22px; align-items: center; flex-wrap: wrap; }
.ring {
  --p: var(--target); position: relative; width: 132px; height: 132px; border-radius: 50%; flex: none; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.10) 0); animation: fill 1.5s cubic-bezier(.2,.8,.2,1) both;
}
.ring::before { content: ""; position: absolute; inset: 11px; border-radius: 50%; background: #0e1530; }
.ring .num { position: relative; text-align: center; line-height: 1.15; }
.ring .num b { font-size: 1.9rem; }
.ring .num span { display: block; font-size: .68rem; color: #9fb0d6; }
.verdict { flex: 1; min-width: 210px; }
.verdict .k { font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: #8ea0c9; }
.badge {
  display: inline-flex; align-items: center; gap: 8px; margin: 6px 0 4px; padding: 6px 15px; border-radius: 999px;
  font-weight: 700; font-size: 1.1rem; color: var(--c); background: var(--bg); border: 1px solid var(--c);
}
.votes { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
.vd { width: 24px; height: 24px; border-radius: 50%; background: var(--c); box-shadow: 0 0 12px var(--c); animation: pop .5s ease both; animation-delay: calc(var(--i) * .15s); }
.vt { font-size: .88rem; color: #c3d0f0; margin-left: 4px; }
.mr { margin-top: 15px; }
.mr-head { display: flex; justify-content: space-between; gap: 10px; font-size: .88rem; margin-bottom: 6px; }
.mr-head .lbl { font-weight: 700; }
.bar { height: 10px; border-radius: 999px; overflow: hidden; background: rgba(96,165,250,.28); }
.bar i { display: block; height: 100%; width: calc(var(--w) * 1%); background: linear-gradient(90deg, #f59e0b, #fcd34d); animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i) * .15s); }
.legend { display: flex; gap: 14px; margin-top: 12px; font-size: .75rem; color: #9fb0d6; }
.legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; background: var(--c); }
.warn { margin-top: 14px; padding: 10px 14px; border-radius: 12px; font-size: .9rem; color: #fde68a; background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.42); }
.foot { margin-top: 12px; font-size: .75rem; color: #8091ba; }

.chipx { display: inline-block; margin: 2px 0 0; padding: 3px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600; color: var(--c); border: 1px solid var(--c); background: rgba(255,255,255,.04); }
.meaning { margin-top: 16px; padding: 12px 14px; border-radius: 12px; font-size: .92rem; line-height: 1.5; color: #d6e0ff; background: rgba(255,255,255,.05); }
.next { margin-top: 14px; padding: 12px 16px; border-radius: 12px; font-size: .9rem; color: #d1fae5; background: rgba(52,211,153,.08); border: 1px solid rgba(52,211,153,.28); }
.next b { display: block; margin-bottom: 6px; color: #6ee7b7; }
.next ul { margin: 0; padding-left: 18px; }
.next li { margin: 4px 0; }
.orb { display: flex; align-items: center; gap: 14px; margin: 8px 0 12px 10px; color: #c4b5fd; font-size: .85rem; animation: orbfade .6s ease forwards; animation-delay: var(--dur); }
.orb .core { position: relative; width: 44px; height: 44px; flex: none; }
.orb .core i { position: absolute; inset: 0; border-radius: 50%; background: radial-gradient(circle at 35% 35%, #ddd6fe, #7c3aed 60%, #4c1d95); animation: pulse 1.1s ease-in-out var(--n); }
.orb .core i:nth-child(2) { opacity: .5; animation-delay: .25s; }
.orb .core i:nth-child(3) { opacity: .3; animation-delay: .5s; }
.dots { display: flex; gap: 7px; align-items: center; padding: 6px 2px; color: #9fb0d6; font-size: .9rem; }
.dots b { width: 9px; height: 9px; border-radius: 50%; background: #a78bfa; animation: bounce 1.2s ease-in-out infinite; }
.dots b:nth-child(2) { animation-delay: .15s; }
.dots b:nth-child(3) { animation-delay: .3s; }

@keyframes shine { to { background-position: 300% 0; } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }
@keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes fill { from { --p: 0; } }
@keyframes grow { from { width: 0; } }
@keyframes pop { from { transform: scale(0); } to { transform: scale(1); } }
@keyframes pulse { 0%, 100% { transform: scale(.72); opacity: .9; } 50% { transform: scale(1.3); opacity: .3; } }
@keyframes orbfade { to { opacity: 0; height: 0; margin: 0; } }
@keyframes bounce { 0%, 80%, 100% { transform: scale(.55); opacity: .4; } 40% { transform: scale(1); opacity: 1; } }
"""

CSS_TAG = f"<style>{CSS}</style>"


def hero_html(models_ready, models_total, voice_on, language):
    """Animated title with small status chips."""
    ok = "#34d399" if models_ready == models_total else AMBER
    chips = (
        f'<span class="chip"><span class="dot" style="background:{ok};box-shadow:0 0 8px {ok}"></span>'
        f"{models_ready}/{models_total} models ready</span>"
        f'<span class="chip">🎙 voice {"on" if voice_on else "off"}</span>'
        f'<span class="chip">🌐 {escape(language)}</span>'
        '<span class="chip">🔒 photos analysed on this computer</span>'
    )
    return (
        '<div class="hero"><h1>Mpox Buddy</h1>'
        "<p>Chat, talk, or show me a skin photo</p>"
        f'<div class="chips">{chips}</div>'
        '<div class="tiny">Educational use only. Not a medical diagnosis.</div></div>'
    )


def thinking_html(label):
    return f'<div class="dots"><b></b><b></b><b></b><span>{escape(label)}</span></div>'


def orb_html(seconds):
    """A pulsing orb that runs for about as long as the spoken clip, then fades away."""
    seconds = max(1.5, float(seconds))
    pulses = max(1, round(seconds / 1.1))
    return (
        f'<div class="orb" style="--dur:{seconds:.1f}s;--n:{pulses}">'
        '<div class="core"><i></i><i></i><i></i></div><span>speaking...</span></div>'
    )


LEAN_COLOR = {"monkeypox": AMBER, "not_monkeypox": BLUE, "unclear": GREY}
AGREE_COLOR = {"high": "#34d399", "moderate": AMBER, "low": "#f87171"}
AGREE_TEXT = {"high": "High agreement", "moderate": "Moderate agreement", "low": "Low agreement"}

URGENT = "Seek urgent care for trouble breathing, a high fever, fast-spreading or very painful lesions, or symptoms near the eyes."
NEXT_STEPS = {
    "monkeypox": [
        "Contact a doctor or your local health service soon for an exam and testing.",
        "Avoid close skin contact and keep the area covered until you are assessed.",
        URGENT,
    ],
    "not_monkeypox": [
        "If the rash is new, painful, spreading, or comes with fever or swollen glands, get it checked anyway.",
        "Only an exam and tests can confirm or rule out monkeypox.",
        URGENT,
    ],
    "unclear": [
        "Take a clearer, well-lit, close-up photo and try again.",
        "Have the rash checked by a healthcare professional.",
        URGENT,
    ],
}
MEANING = {
    "monkeypox": "The models found patterns that resemble the monkeypox photos they learned from.",
    "not_monkeypox": (
        "The models did not find patterns that match monkeypox. Other skin conditions can look similar, "
        "so this does not rule it out."
    ),
    "unclear": "The models could not agree, so this photo is borderline for them.",
}


def _verdict_info(result):
    """The verdict stored in the result, or a simple one derived from the vote."""
    if result.get("verdict"):
        return result["verdict"]
    labels = [m["label"] for m in result["models"]]
    unanimous = max(labels.count("Monkeypox"), labels.count("Others")) == len(labels)
    lean = {"Monkeypox": "monkeypox", "Others": "not_monkeypox"}.get(result["final"], "unclear")
    headline = {"monkeypox": "Leans monkeypox", "not_monkeypox": "Leans not monkeypox", "unclear": "Can't tell"}[lean]
    return {"lean": lean, "agreement": "high" if unanimous else "low", "headline": headline, "note": ""}


def _next_steps(verdict):
    steps = list(NEXT_STEPS[verdict["lean"]])
    if verdict["agreement"] == "low" and verdict["lean"] != "unclear":
        steps.append("The models disagree, so a clearer, well-lit close-up photo may help.")
    return steps


def _meaning(verdict):
    text = MEANING[verdict["lean"]]
    if verdict["agreement"] == "low" and verdict["lean"] != "unclear":
        text += " Because the models disagree, treat this as a weak signal."
    return text


def result_card_html(result):
    models = result["models"]
    n = len(models)
    verdict = _verdict_info(result)
    color = LEAN_COLOR[verdict["lean"]]
    labels = [m["label"] for m in models]
    agree = max(labels.count("Monkeypox"), labels.count("Others"))
    unanimous = agree == n
    avg = float(result["avg_monkeypox_pct"])
    ring_color = AMBER if avg >= 50 else BLUE

    dots = "".join(
        f'<span class="vd" style="--c:{AMBER if label == "Monkeypox" else BLUE};--i:{i}"></span>'
        for i, label in enumerate(labels)
    )
    sentence = (
        f"Vote: {agree} of {n} say {'Monkeypox' if result['final'] == 'Monkeypox' else 'Others'}"
        if result["final"] != "Tie"
        else f"The {n} models are split"
    )
    rows = "".join(
        '<div class="mr"><div class="mr-head">'
        f'<span class="lbl">{escape(m["name"])} '
        f'<em style="color:{AMBER if m["label"] == "Monkeypox" else BLUE};font-style:normal">{escape(m["label"])}</em></span>'
        f'<span>{float(m["monkeypox_pct"]):.1f}% Monkeypox</span></div>'
        f'<div class="bar"><i style="--w:{float(m["monkeypox_pct"])};--i:{i}"></i></div></div>'
        for i, m in enumerate(models)
    )
    note = f" ({verdict['note'].rstrip('.')})" if verdict.get("note") and not unanimous else ""
    warn = (
        ""
        if unanimous
        else f'<div class="warn">⚠ The models disagree{escape(note)}, so treat this result as uncertain.</div>'
    )
    steps = "".join(f"<li>{escape(t)}</li>" for t in _next_steps(verdict))
    agree_color = AGREE_COLOR[verdict["agreement"]]
    return (
        '<div class="rc"><div class="rc-top">'
        f'<div class="ring" style="--c:{ring_color};--target:{avg}"><div class="num"><b>{avg:.0f}%</b><span>avg Monkeypox</span></div></div>'
        '<div class="verdict"><div class="k">What my models think</div>'
        f'<div class="badge" style="--c:{color};--bg:{GLOW[color]}">{escape(verdict["headline"])}</div><br>'
        f'<span class="chipx" style="--c:{agree_color}">{AGREE_TEXT[verdict["agreement"]]}</span>'
        f'<div class="votes">{dots}<span class="vt">{escape(sentence)}</span></div></div></div>'
        f'<div class="meaning">{escape(_meaning(verdict))}</div>'
        f"{rows}{warn}"
        f'<div class="legend"><span style="--c:{AMBER}">Monkeypox</span><span style="--c:{BLUE}">Others</span></div>'
        f'<div class="next"><b>What to do next</b><ul>{steps}</ul></div>'
        '<div class="foot">This is what three AI models think of the photo, not a diagnosis. The ring is a simple average '
        "and the percentages are not calibrated. Only a healthcare professional can tell if someone has monkeypox.</div></div>"
    )


def report_text(result):
    """Plain-text summary for the download button."""
    verdict = _verdict_info(result)
    lines = [
        "MPOX BUDDY: IMAGE CHECK SUMMARY",
        "Educational tool. This is NOT a medical diagnosis.",
        "",
        f"What the models think: {verdict['headline']}",
        f"Agreement between models: {AGREE_TEXT[verdict['agreement']]}. {verdict.get('note', '')}".strip(),
        f"Majority vote: {result['final']} ({result['agreement']} models agree)",
        f"Average Monkeypox probability: {result['avg_monkeypox_pct']}%",
        "",
        _meaning(verdict),
        "",
        "Individual models:",
    ]
    for m in result["models"]:
        lines.append(f"- {m['name']}: {m['label']} (Monkeypox {m['monkeypox_pct']}%, Others {m['others_pct']}%)")
    lines += ["", "What to do next:"] + [f"- {t}" for t in _next_steps(verdict)]
    lines += [
        "",
        "Notes:",
        "- Probabilities are not calibrated and can be overconfident.",
        "- A photo cannot confirm monkeypox. Only a healthcare professional can, usually with an exam and tests.",
    ]
    return "\n".join(lines)
