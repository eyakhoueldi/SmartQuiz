import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
load_dotenv()
from rag.loader import load_document
from rag.chunking import split_text
from rag.embeddings import add_chunks, clear_collection
from quiz.generator import generate_quiz
from quiz.evaluator import evaluate_answers
from quiz.flashcards import generate_flashcards
from agent.agent import run_agent

st.set_page_config(
    page_title="SmartQuiz AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE  (only write missing keys once)
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "chat_history": [],
    "quiz_questions": [], "quiz_answers": {}, "quiz_submitted": False, "quiz_type": "MCQ",
    "uploaded_files": [],
    "fc_cards": [], "fc_index": 0, "fc_known": set(), "fc_unknown": set(),
    "fc_flipped": False, "fc_topic": "",
    "_css_injected": False,
    "_js_injected": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM  — injected once per session via a flag
# ══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?:wght@400;500;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

:root {
  --bg:#07070f; --s1:#0d0d1a; --s2:#12121f; --s3:#191927; --s4:#1e1e30;
  --border:rgba(255,255,255,0.06); --border2:rgba(255,255,255,0.11);
  --a1:#7c6fff; --a2:#c26fff; --a3:#ff6fbe;
  --text:#e4e4f4; --muted:#5c5c78; --muted2:#7878a0;
  --green:#34d399; --red:#fb7185; --amber:#fbbf24; --cyan:#06b6d4; --blue:#3b82f6;
  --font-h:'Syne',sans-serif; --font-b:'DM Sans',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,.stApp{font-family:var(--font-b)!important;color:var(--text)!important;}
.stApp{
  background:var(--bg)!important;
  background-image:
    radial-gradient(ellipse 90% 55% at 15% -5%,rgba(124,111,255,.10) 0%,transparent 58%),
    radial-gradient(ellipse 70% 45% at 85% 105%,rgba(194,111,255,.07) 0%,transparent 55%),
    radial-gradient(ellipse 50% 35% at 50% 50%,rgba(255,111,190,.04) 0%,transparent 60%)!important;
  background-attachment:fixed!important;
}
.btn-row{display:flex;justify-content:center;gap:12px;width:100%;flex-wrap:wrap;margin:10px 0;}
#MainMenu,footer,[data-testid="stDecoration"],[data-testid="stToolbar"],[data-testid="stMainMenu"]{visibility:hidden!important;display:none!important;}
header[data-testid="stHeader"]{background:transparent!important;box-shadow:none!important;border:none!important;}
[data-testid="stSidebarCollapsedControl"]{display:flex!important;visibility:visible!important;opacity:1!important;pointer-events:auto!important;z-index:999999!important;}
.block-container{padding:2.75rem 2rem 5rem 2rem!important;max-width:1100px!important;margin:0 auto!important;}

/* fc-pills — hidden until CSS is ready to prevent FOUC */
.fc-pills{
  display:flex;justify-content:center;align-items:center;gap:14px;flex-wrap:wrap;
  width:100%;margin:20px 0;
  /* start invisible; the keyframe below reveals after first paint */
  animation:fcReveal .01s forwards;
}
@keyframes fcReveal{to{opacity:1;}}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:var(--s1)!important;border-right:1px solid var(--border)!important;box-shadow:6px 0 48px rgba(0,0,0,.55)!important;}
[data-testid="stSidebar"]>div:first-child{padding:0!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] hr{border-color:var(--border)!important;margin:6px 0!important;}
.sb-logo{background:linear-gradient(135deg,#0f0f20 0%,#17102e 100%);border-bottom:1px solid var(--border);padding:26px 24px 22px;position:relative;overflow:hidden;}
.sb-logo::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 120% 80% at 10% 50%,rgba(124,111,255,.18) 0%,transparent 65%);pointer-events:none;}
.sb-logo-icon{font-size:32px;display:block;margin-bottom:8px;position:relative;filter:drop-shadow(0 0 12px rgba(124,111,255,.7));}
.sb-logo-title{font-family:var(--font-h)!important;font-size:18px!important;font-weight:800!important;letter-spacing:-.3px!important;background:linear-gradient(90deg,#fff 30%,var(--a2) 100%);-webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;background-clip:text!important;position:relative;z-index:2;}
.sb-logo-sub{font-size:11px!important;color:var(--muted)!important;letter-spacing:.5px!important;margin-top:3px!important;position:relative;z-index:2;}
[data-testid="stRadio"]>label{display:none!important;}
[data-testid="stRadio"]>div{gap:3px!important;flex-direction:column!important;padding:0 14px!important;}
[data-testid="stRadio"]>div>label{background:transparent!important;border:1px solid transparent!important;border-radius:9px!important;padding:10px 13px!important;font-size:13.5px!important;font-weight:500!important;color:var(--muted2)!important;cursor:pointer!important;transition:all .16s ease!important;}
[data-testid="stRadio"]>div>label:hover{background:rgba(124,111,255,.07)!important;color:var(--text)!important;}
[data-testid="stRadio"]>div>label:has(input:checked){background:linear-gradient(135deg,rgba(124,111,255,.18),rgba(194,111,255,.12))!important;color:#fff!important;border-color:rgba(124,111,255,.28)!important;box-shadow:0 2px 12px rgba(124,111,255,.15)!important;}
.sb-section{font-size:9.5px!important;font-weight:700!important;letter-spacing:2.8px!important;text-transform:uppercase!important;color:var(--muted)!important;padding:18px 22px 5px!important;}
.file-pill{display:flex;align-items:center;gap:8px;background:var(--s2);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:11.5px;color:var(--text);margin:3px 14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.file-pill-icon{color:var(--a1)!important;font-size:12px;flex-shrink:0;}

/* ── Page header ── */
.page-header{margin-bottom:32px;padding-bottom:22px;border-bottom:1px solid var(--border);}
.page-title{font-family:var(--font-h)!important;font-size:28px;font-weight:800;letter-spacing:-.5px;background:linear-gradient(90deg,#ffffff 0%,var(--a2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.2;margin:0 0 4px;}
.page-sub{font-size:13.5px;color:var(--muted2);margin:0;font-weight:400;line-height:1.4;}

/* ── Control card ── */
.ctrl-card{background:linear-gradient(135deg,rgba(124,111,255,.05) 0%,rgba(194,111,255,.03) 100%);border:1px solid rgba(124,111,255,.15);border-radius:18px;padding:22px 24px;margin-bottom:24px;position:relative;overflow:hidden;backdrop-filter:blur(8px);}
.ctrl-card::before{content:'';position:absolute;top:-40px;right:-40px;width:140px;height:140px;border-radius:50%;background:radial-gradient(circle,rgba(124,111,255,.08) 0%,transparent 70%);pointer-events:none;}

/* ── Inputs ── */
[data-baseweb="input"]{background:var(--s3)!important;border-radius:11px!important;border:1px solid rgba(124,111,255,.2)!important;}
[data-baseweb="select"]>div{background:var(--s3)!important;border:1px solid rgba(124,111,255,.2)!important;border-radius:11px!important;color:#fff!important;}
[data-baseweb="popover"]{background:var(--s2)!important;border:1px solid rgba(124,111,255,.2)!important;border-radius:12px!important;}
[data-baseweb="menu"]{background:var(--s2)!important;}
[data-baseweb="option"]:hover{background:rgba(124,111,255,.12)!important;}
label{color:var(--text)!important;font-size:13px!important;font-weight:500!important;letter-spacing:.1px!important;font-family:var(--font-b)!important;}
[data-testid="stFileUploader"]{background:linear-gradient(135deg,rgba(124,111,255,.04) 0%,rgba(194,111,255,.02) 100%)!important;border:1.5px dashed rgba(124,111,255,.4)!important;border-radius:14px!important;transition:all .3s ease!important;padding:18px!important;}
[data-testid="stFileUploader"]:hover{border-color:rgba(124,111,255,.7)!important;background:linear-gradient(135deg,rgba(124,111,255,.07) 0%,rgba(194,111,255,.04) 100%)!important;}
[data-testid="stFileUploaderDropzoneInstructions"]{color:var(--muted)!important;font-size:12.5px!important;line-height:1.5!important;}

/* ── Buttons ── */
button[kind="primary"]{background:linear-gradient(135deg,var(--a1) 0%,var(--a2) 100%)!important;border:none!important;border-radius:10px!important;color:#fff!important;font-weight:600!important;font-size:13.5px!important;letter-spacing:.1px!important;font-family:var(--font-b)!important;box-shadow:0 4px 22px rgba(124,111,255,.38),inset 0 1px 0 rgba(255,255,255,.16)!important;transition:all .18s cubic-bezier(.34,1.56,.64,1)!important;padding:10px 18px!important;position:relative;overflow:hidden;}
button[kind="primary"]:hover{transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(124,111,255,.55),inset 0 1px 0 rgba(255,255,255,.22)!important;}
button[kind="primary"]:active{transform:translateY(0)!important;}
button[kind="secondary"]{background:linear-gradient(135deg,rgba(124,111,255,.12),rgba(194,111,255,.08))!important;border:1px solid rgba(124,111,255,.25)!important;border-radius:10px!important;color:var(--text)!important;font-weight:500!important;font-size:13.5px!important;font-family:var(--font-b)!important;transition:all .18s ease!important;padding:10px 18px!important;}
button[kind="secondary"]:hover{background:linear-gradient(135deg,rgba(124,111,255,.18),rgba(194,111,255,.14))!important;border-color:rgba(124,111,255,.4)!important;color:#fff!important;transform:translateY(-1px)!important;box-shadow:0 4px 16px rgba(124,111,255,.2)!important;}
button[kind="secondary"]:active{transform:translateY(0)!important;}
button:disabled{opacity:.32!important;cursor:not-allowed!important;}

/* ── Quiz radio ── */
.quiz-radio [data-testid="stRadio"]>div{flex-direction:column!important;gap:7px!important;padding:0!important;}
.quiz-radio [data-testid="stRadio"]>div>label{background:linear-gradient(135deg,rgba(30,30,48,.8),rgba(25,25,39,.8))!important;border:1px solid rgba(124,111,255,.15)!important;border-radius:11px!important;padding:12px 18px!important;font-size:14px!important;font-weight:400!important;color:var(--text)!important;cursor:pointer!important;transition:all .15s ease!important;line-height:1.4!important;}
.quiz-radio [data-testid="stRadio"]>div>label:hover{border-color:rgba(124,111,255,.4)!important;background:linear-gradient(135deg,rgba(124,111,255,.09),rgba(194,111,255,.06))!important;color:#fff!important;}
.quiz-radio [data-testid="stRadio"]>div>label:has(input:checked){border-color:var(--a1)!important;background:linear-gradient(135deg,rgba(124,111,255,.18),rgba(194,111,255,.12))!important;color:#fff!important;box-shadow:0 0 16px rgba(124,111,255,.2)!important;}

/* ── Question card ── */
.q-card{background:linear-gradient(135deg,rgba(30,30,48,.6),rgba(25,25,39,.6));border:1px solid rgba(124,111,255,.15);border-radius:16px;padding:20px 24px;margin-bottom:16px;transition:all .2s ease;backdrop-filter:blur(4px);}
.q-card:hover{border-color:rgba(124,111,255,.3);background:linear-gradient(135deg,rgba(30,30,48,.8),rgba(25,25,39,.8));}
.q-num{display:inline-block;background:linear-gradient(135deg,var(--a1),var(--a2));color:#fff;font-size:11px;font-weight:700;letter-spacing:1px;padding:4px 10px;border-radius:50px;margin-bottom:12px;font-family:var(--font-h);box-shadow:0 2px 8px rgba(124,111,255,.3);}
.q-text{font-family:var(--font-h)!important;font-size:15.5px;font-weight:600;color:#fff;line-height:1.5;margin:0;word-wrap:break-word;overflow-wrap:break-word;}

/* ── Score cards ── */
.score-row{display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;}
.score-card{flex:1;min-width:120px;background:linear-gradient(135deg,rgba(30,30,48,.7),rgba(25,25,39,.7));border:1px solid rgba(124,111,255,.15);border-radius:16px;padding:22px 16px;text-align:center;transition:all .3s ease;overflow:hidden;backdrop-filter:blur(4px);}
.score-card:hover{transform:translateY(-4px);border-color:rgba(124,111,255,.3);}
.score-card.green{border-color:rgba(52,211,153,.28);background:linear-gradient(135deg,rgba(52,211,153,.08),rgba(52,211,153,.04));}
.score-card.blue{border-color:rgba(124,111,255,.28);background:linear-gradient(135deg,rgba(124,111,255,.08),rgba(124,111,255,.04));}
.score-card.red{border-color:rgba(251,113,133,.28);background:linear-gradient(135deg,rgba(251,113,133,.08),rgba(251,113,133,.04));}
.score-num{font-family:var(--font-h);font-size:38px;font-weight:900;color:#fff;line-height:1;letter-spacing:-1px;word-break:break-word;}
.score-lbl{font-size:10px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:var(--muted);margin-top:8px;line-height:1.2;}

/* ── Expanders ── */
[data-testid="stExpander"]{background:linear-gradient(135deg,rgba(30,30,48,.6),rgba(25,25,39,.6))!important;border:1px solid rgba(124,111,255,.15)!important;border-radius:13px!important;margin-bottom:8px!important;overflow:hidden!important;backdrop-filter:blur(4px)!important;}
[data-testid="stExpander"]:hover{border-color:rgba(124,111,255,.3)!important;}
[data-testid="stExpanderDetails"]{background:transparent!important;padding:16px 24px!important;overflow:hidden;}
[data-testid="stExpanderHeader"]{padding:14px 18px!important;}
summary{color:var(--text)!important;font-weight:500!important;font-size:14px!important;word-break:break-word;}
.result-col{padding:12px;border-radius:10px;background:rgba(124,111,255,.06);word-wrap:break-word;overflow-wrap:break-word;border:1px solid rgba(124,111,255,.1);}
.result-label{font-size:12px;color:var(--muted);margin-bottom:6px;line-height:1.2;font-weight:600;}
.result-text{font-weight:600;font-size:14px;line-height:1.4;word-wrap:break-word;overflow-wrap:break-word;}

/* ── Progress ── */
[data-testid="stProgressBar"]>div{background:linear-gradient(90deg,var(--a1),var(--a2),var(--a3))!important;border-radius:6px!important;transition:width .5s cubic-bezier(.34,1.56,.64,1)!important;box-shadow:0 0 16px rgba(124,111,255,.4)!important;}
[data-testid="stProgressBar"]{background:rgba(124,111,255,.08)!important;border-radius:6px!important;height:7px!important;overflow:hidden!important;border:1px solid rgba(124,111,255,.12)!important;}

/* ── Chat ── */
[data-testid="stChatMessage"]{background:linear-gradient(135deg,rgba(30,30,48,.6),rgba(25,25,39,.6))!important;border:1px solid rgba(124,111,255,.15)!important;border-radius:16px!important;padding:14px 18px!important;margin-bottom:10px!important;overflow:hidden!important;backdrop-filter:blur(4px)!important;}
[data-testid="stChatMessage"][data-testid*="user"]{background:linear-gradient(135deg,rgba(124,111,255,.12),rgba(194,111,255,.08))!important;border-color:rgba(124,111,255,.25)!important;}
[data-testid="stChatInput"]{background:linear-gradient(135deg,rgba(30,30,48,.7),rgba(25,25,39,.7))!important;border:1px solid rgba(124,111,255,.2)!important;border-radius:14px!important;overflow:hidden!important;backdrop-filter:blur(4px)!important;}
[data-testid="stChatInput"]:focus-within{border-color:var(--a1)!important;box-shadow:0 0 0 3px rgba(124,111,255,.12)!important;}
[data-testid="stChatInputTextArea"]{background:transparent!important;color:#fff!important;font-family:var(--font-b)!important;}
[data-testid="stChatInputTextArea"]::placeholder{color:transparent!important;}

/* ── Alerts ── */
[data-testid="stAlert"]{border-radius:12px!important;border:1px solid!important;font-size:13.5px!important;line-height:1.5!important;overflow:hidden!important;backdrop-filter:blur(4px)!important;}
div[data-testid="stAlertContentWarning"]{background:rgba(251,191,36,.07)!important;border-color:rgba(251,191,36,.22)!important;color:var(--amber)!important;}
div[data-testid="stAlertContentSuccess"]{background:rgba(52,211,153,.07)!important;border-color:rgba(52,211,153,.22)!important;color:var(--green)!important;}
div[data-testid="stAlertContentInfo"]{background:rgba(124,111,255,.07)!important;border-color:rgba(124,111,255,.22)!important;color:#c4b8ff!important;line-height:1.5!important;}

/* ── Misc ── */
hr{border:none!important;border-top:1px solid rgba(124,111,255,.12)!important;margin:24px 0!important;}
[data-testid="stSpinner"]>div{border-top-color:var(--a1)!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(124,111,255,.2);border-radius:10px;}
::-webkit-scrollbar-thumb:hover{background:rgba(124,111,255,.4);}

/* ── Flashcard pills — fixed size, no FOUC ── */
.fc-pill{
  border-radius:50px;padding:8px 16px;font-size:12px;font-weight:600;letter-spacing:.1px;
  display:inline-flex;align-items:center;gap:6px;white-space:nowrap;line-height:1.2;
  border:1.5px solid;backdrop-filter:blur(6px);transition:all .3s ease;
  /* hard sizes so the row never jumps */
  min-width:108px;justify-content:center;
}
.fc-pill.g{background:linear-gradient(135deg,rgba(52,211,153,.15),rgba(52,211,153,.08));border-color:rgba(52,211,153,.5);color:#34d399;box-shadow:0 2px 12px rgba(52,211,153,.15);}
.fc-pill.g:hover{background:linear-gradient(135deg,rgba(52,211,153,.22),rgba(52,211,153,.14));box-shadow:0 4px 16px rgba(52,211,153,.25);transform:translateY(-2px);}
.fc-pill.b{background:linear-gradient(135deg,rgba(124,111,255,.15),rgba(124,111,255,.08));border-color:rgba(124,111,255,.4);color:#a8a8ff;box-shadow:0 2px 12px rgba(124,111,255,.15);}
.fc-pill.b:hover{background:linear-gradient(135deg,rgba(124,111,255,.22),rgba(124,111,255,.14));box-shadow:0 4px 16px rgba(124,111,255,.25);transform:translateY(-2px);}
.fc-pill.r{background:linear-gradient(135deg,rgba(251,113,133,.15),rgba(251,113,133,.08));border-color:rgba(251,113,133,.5);color:#fb7185;box-shadow:0 2px 12px rgba(251,113,133,.15);}
.fc-pill.r:hover{background:linear-gradient(135deg,rgba(251,113,133,.22),rgba(251,113,133,.14));box-shadow:0 4px 16px rgba(251,113,133,.25);transform:translateY(-2px);}

.fc-dots{display:flex;justify-content:center;align-items:center;gap:8px;width:100%;margin:16px 0;padding:6px 0;flex-wrap:wrap;}
.card-ctr{font-size:11px;color:var(--muted);letter-spacing:2px;font-weight:700;margin-bottom:20px;text-transform:uppercase;line-height:1.2;text-align:center;width:100%;}
.empty-state{padding:64px 0;opacity:.35;text-align:center;}
.empty-state .icon{font-size:52px;margin-bottom:14px;}
.empty-state .msg{font-size:15px;color:var(--text);line-height:1.5;}
.review-title{font-size:14px;font-weight:700;color:#fff;margin-bottom:16px;line-height:1.2;text-transform:uppercase;letter-spacing:.5px;}
.review-card{background:linear-gradient(135deg,rgba(30,30,48,.6),rgba(25,25,39,.6));border:1px solid rgba(124,111,255,.15);border-radius:12px;padding:16px 18px;margin-bottom:8px;word-break:break-word;overflow-wrap:break-word;backdrop-filter:blur(4px);transition:all .2s ease;}
.review-card:hover{border-color:rgba(124,111,255,.3);background:linear-gradient(135deg,rgba(30,30,48,.8),rgba(25,25,39,.8));}
.review-card-inner{background:rgba(124,111,255,.06);border-radius:8px;padding:12px;margin-top:10px;line-height:1.5;word-wrap:break-word;overflow-wrap:break-word;border:1px solid rgba(124,111,255,.1);}
p{font-size:13.5px;line-height:1.6;}
</style>
"""

# Inject CSS only once per session (stored in a hidden component slot, not re-executed)
st.markdown(_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR TOGGLE JS — injected once via session flag
# ══════════════════════════════════════════════════════════════════════════════
_SIDEBAR_JS = """
<script>
(function(){
  var doc=window.parent.document,win=window.parent,BTN_ID='sq-sidebar-btn';
  if(doc.getElementById(BTN_ID))return; // guard: already injected
  var IO='<svg width="18" height="14" viewBox="0 0 18 14" fill="none"><rect width="18" height="2" rx="1" fill="white"/><rect y="6" width="12" height="2" rx="1" fill="white"/><rect y="12" width="18" height="2" rx="1" fill="white"/></svg>';
  var IC='<svg width="13" height="13" viewBox="0 0 13 13" fill="none"><line x1="1" y1="1" x2="12" y2="12" stroke="white" stroke-width="2" stroke-linecap="round"/><line x1="12" y1="1" x2="1" y2="12" stroke="white" stroke-width="2" stroke-linecap="round"/></svg>';
  function vis(){var sb=doc.querySelector('[data-testid="stSidebar"]');if(!sb)return false;var cs=win.getComputedStyle(sb);if(cs.display==='none'||cs.visibility==='hidden')return false;var r=sb.getBoundingClientRect();return r.width>80&&r.left>-10;}
  function clickNative(){var sb=doc.querySelector('[data-testid="stSidebar"]');if(sb){var btns=sb.querySelectorAll('button');for(var i=0;i<btns.length;i++){var l=(btns[i].getAttribute('aria-label')||'').toLowerCase();if(l.indexOf('close')>-1||l.indexOf('collapse')>-1||l.indexOf('hide')>-1||l.indexOf('sidebar')>-1){btns[i].click();return true;}}}var ctrl=doc.querySelector('[data-testid="stSidebarCollapsedControl"]');if(ctrl){(ctrl.querySelector('button')||ctrl).click();return true;}var all=doc.querySelectorAll('button');for(var j=0;j<all.length;j++){var lj=(all[j].getAttribute('aria-label')||'').toLowerCase();if(lj.indexOf('sidebar')>-1||lj.indexOf('navigation')>-1){all[j].click();return true;}}return false;}
  function forceToggle(open){var sb=doc.querySelector('[data-testid="stSidebar"]');if(!sb)return;if(open){sb.style.setProperty('transform','none','important');sb.style.setProperty('display','flex','important');sb.style.setProperty('visibility','visible','important');sb.style.setProperty('width','21rem','important');sb.style.setProperty('min-width','21rem','important');}else{sb.style.setProperty('transform','translateX(-110%)','important');setTimeout(function(){sb.style.removeProperty('transform');},4000);}}
  function toggle(){var open=vis();if(!clickNative())forceToggle(!open);}
  var btn=doc.createElement('button');
  btn.id=BTN_ID;btn.innerHTML=IO;
  btn.style.cssText='position:fixed;top:12px;left:12px;z-index:2147483647;width:42px;height:42px;border-radius:10px;background:linear-gradient(135deg,#7c6fff,#c26fff);border:none;outline:none;color:#fff;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 4px 20px rgba(124,111,255,.5);transition:transform .16s,box-shadow .16s,left .3s;';
  btn.onmouseover=function(){btn.style.transform='scale(1.08)';btn.style.boxShadow='0 6px 28px rgba(124,111,255,.72)';};
  btn.onmouseout=function(){btn.style.transform='';btn.style.boxShadow='0 4px 20px rgba(124,111,255,.5)';};
  btn.addEventListener('click',function(e){e.preventDefault();e.stopPropagation();toggle();});
  doc.body.appendChild(btn);
  var _last=null;
  setInterval(function(){
    var b=doc.getElementById(BTN_ID);if(!b)return;
    var open=vis();
    if(open!==_last){b.innerHTML=open?IC:IO;b.title=open?'Close sidebar':'Open sidebar';b.style.left=open?'calc(21rem - 54px)':'12px';_last=open;}
  },350);
})();
</script>
"""

# Use a stable key so Streamlit skips re-rendering this component on reruns
components.html(_SIDEBAR_JS, height=0)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <span class="sb-logo-icon">🧠</span>
        <div class="sb-logo-title">SmartQuiz AI</div>
        <div class="sb-logo-sub">Intelligent study assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("nav", ["💬  Chat", "📝  Quiz", "🃏  Flashcards"], label_visibility="collapsed")
    st.divider()

    st.markdown('<div class="sb-section">Documents</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")
    with col2:
        st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)

    if uploaded:
        for file in uploaded:
            if file.name not in st.session_state.uploaded_files:
                os.makedirs("uploads", exist_ok=True)
                path = os.path.join("uploads", file.name)
                with open(path, "wb") as f:
                    f.write(file.read())
                with st.spinner(f"Indexing {file.name}…"):
                    text   = load_document(path)
                    chunks = split_text(text)
                    n      = add_chunks(chunks, file.name)
                st.session_state.uploaded_files.append(file.name)
                st.success(f"✓ {file.name} — {n} chunks")

    if st.session_state.uploaded_files:
        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        for fname in st.session_state.uploaded_files:
            short = fname if len(fname) < 26 else fname[:23] + "…"
            st.markdown(f'<div class="file-pill"><span class="file-pill-icon">📄</span><span>{short}</span></div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        if st.button("Clear all", type="secondary", use_container_width=True):
            clear_collection()
            st.session_state.uploaded_files = []
            st.rerun()
    else:
        st.markdown("""
        <div style='padding:10px 20px 14px;font-size:12px;color:#5c5c78;line-height:1.6;'>
            Upload PDF or TXT files to get started. Documents are used to generate quizzes, flashcards and answers.
        </div>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def page_header(title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">{title}</div>
        <p class="page-sub">{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def no_docs_warning():
    st.warning("Upload at least one document from the sidebar to get started.")

def esc(s):
    """Escape a string for safe embedding in HTML attributes / inline JS."""
    return s.replace("&", "&amp;").replace("'", "\\'").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

# ══════════════════════════════════════════════════════════════════════════════
# FLASHCARD HTML — built once and reused via @st.cache_data
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _build_flashcard_html(front: str, back: str, hint: str, is_flipped: bool) -> str:
    front_s = esc(front)
    back_s  = esc(back)
    hint_s  = esc(hint)
    hint_block = f'<div class="hint">💡 {hint_s}</div>' if hint_s else ''
    flip_cls   = "flipped" if is_flipped else ""
    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:opsz,wght@9..40,400;9..40,500&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;padding:4px 2px;}}
.scene{{width:100%;height:252px;perspective:1400px;cursor:pointer;}}
.card{{width:100%;height:100%;position:relative;transform-style:preserve-3d;transition:transform .68s cubic-bezier(.4,0,.2,1);}}
.card.flipped{{transform:rotateY(180deg);}}
.face{{position:absolute;inset:0;border-radius:20px;backface-visibility:hidden;-webkit-backface-visibility:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:34px 44px;text-align:center;overflow:hidden;}}
.front{{background:#0d0d1a;border:1px solid rgba(124,111,255,.38);box-shadow:0 0 0 1px rgba(124,111,255,.07),0 22px 56px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,255,255,.06);}}
.front::before{{content:'';position:absolute;top:-55px;right:-55px;width:190px;height:190px;border-radius:50%;background:radial-gradient(circle,rgba(124,111,255,.17) 0%,transparent 68%);pointer-events:none;}}
.front::after{{content:'';position:absolute;bottom:-65px;left:-25px;width:155px;height:155px;border-radius:50%;background:radial-gradient(circle,rgba(194,111,255,.11) 0%,transparent 68%);pointer-events:none;}}
.back{{background:linear-gradient(148deg,#130a2c 0%,#0c0820 42%,#1a0e3c 100%);border:1px solid rgba(194,111,255,.34);box-shadow:0 0 0 1px rgba(194,111,255,.07),0 22px 56px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,255,255,.06);transform:rotateY(180deg);}}
.back::before{{content:'';position:absolute;top:-28px;left:50%;transform:translateX(-50%);width:210px;height:110px;border-radius:50%;background:radial-gradient(ellipse,rgba(194,111,255,.2) 0%,transparent 68%);pointer-events:none;}}
.pill{{font-size:9px;font-weight:700;letter-spacing:3.2px;text-transform:uppercase;padding:4px 13px;border-radius:50px;margin-bottom:18px;display:inline-block;position:relative;z-index:1;font-family:'DM Sans',sans-serif;}}
.front .pill{{background:rgba(124,111,255,.14);color:rgba(160,148,255,.95);border:1px solid rgba(124,111,255,.26);}}
.back  .pill{{background:rgba(194,111,255,.14);color:rgba(210,160,255,.95);border:1px solid rgba(194,111,255,.26);}}
.ft{{font-size:19px;font-weight:700;color:#fff;line-height:1.45;letter-spacing:-.2px;position:relative;z-index:1;word-break:break-word;overflow-wrap:break-word;font-family:'Syne',sans-serif;}}
.bt{{font-size:15px;font-weight:400;color:#ddddf0;line-height:1.68;position:relative;z-index:1;word-break:break-word;overflow-wrap:break-word;font-family:'DM Sans',sans-serif;}}
.hint{{margin-top:15px;font-size:11px;color:rgba(255,255,255,.27);font-style:italic;position:relative;z-index:1;word-break:break-word;}}
.tap{{position:absolute;bottom:14px;font-size:9px;letter-spacing:1.8px;color:rgba(255,255,255,.17);text-transform:uppercase;font-family:'DM Sans',sans-serif;}}
.scene:hover .card:not(.flipped){{transform:rotateY(5deg) rotateX(1.5deg);}}
.scene:hover .card.flipped{{transform:rotateY(175deg) rotateX(1.5deg);}}
.card.pop{{transform:scale(0.975)!important;transition:transform .09s!important;}}
</style></head><body>
<div class="scene" onclick="flip()">
  <div class="card {flip_cls}" id="c">
    <div class="face front">
      <span class="pill">Question</span>
      <div class="ft">{front_s}</div>
      {hint_block}
      <span class="tap">tap to reveal</span>
    </div>
    <div class="face back">
      <span class="pill">Answer</span>
      <div class="bt">{back_s}</div>
      <span class="tap">tap to go back</span>
    </div>
  </div>
</div>
<script>
function flip(){{var c=document.getElementById('c');c.classList.add('pop');setTimeout(function(){{c.classList.remove('pop');}},90);c.classList.toggle('flipped');}}
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════════════════
if page == "💬  Chat":
    page_header("Chat", "Ask questions, request summaries, or say 'quiz me on X'")

    if not st.session_state.uploaded_files:
        no_docs_warning()
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = run_agent(user_input, st.session_state.chat_history[:-1])
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.session_state.chat_history:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear conversation", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝  Quiz":
    page_header("Quiz", "Test your knowledge with AI-generated questions from your documents")

    if not st.session_state.uploaded_files:
        no_docs_warning()
    else:
        st.markdown('<div class="ctrl-card">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            topic = st.text_input("Topic / Chapter", placeholder="e.g. Binary Trees, Photosynthesis…")
        with c2:
            num_q = st.selectbox("Questions", [3, 5, 10], index=1)
        with c3:
            quiz_type = st.selectbox("Type", ["MCQ", "True/False"])
        with c4:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Generate Quiz", type="primary"):
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Crafting your quiz…"):
                    st.session_state.quiz_questions = generate_quiz(topic, num_q, quiz_type, difficulty)
                st.session_state.quiz_type      = quiz_type
                st.session_state.quiz_answers   = {}
                st.session_state.quiz_submitted = False

        if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            st.divider()
            st.markdown('<div class="quiz-radio">', unsafe_allow_html=True)
            for i, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"""
                <div class="q-card">
                    <span class="q-num">Q {i+1}</span>
                    <p class="q-text">{q['question']}</p>
                </div>""", unsafe_allow_html=True)
                opts = ["True", "False"] if st.session_state.quiz_type == "True/False" else q.get("options", [])
                choice = st.radio(f"_q{i}", opts, key=f"q_{i}", label_visibility="collapsed")
                st.session_state.quiz_answers[i] = choice
                st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("Submit Quiz", type="primary"):
                st.session_state.quiz_submitted = True
                st.rerun()

        if st.session_state.quiz_submitted and st.session_state.quiz_questions:
            user_answers = [st.session_state.quiz_answers.get(i, "") for i in range(len(st.session_state.quiz_questions))]
            ev    = evaluate_answers(st.session_state.quiz_questions, user_answers, quiz_type=st.session_state.get("quiz_type", "MCQ"))
            pct   = ev["percentage"]
            wrong = ev["total"] - ev["score"]
            emoji = "🏆" if pct >= 80 else "👍" if pct >= 50 else "📚"

            st.markdown(f"""
            <div class="score-row">
                <div class="score-card green"><div class="score-num">{ev["score"]}</div><div class="score-lbl">Correct</div></div>
                <div class="score-card blue"><div class="score-num">{pct}</div><div class="score-lbl" style="font-size:9px;margin-top:2px;">% Score</div></div>
                <div class="score-card red"><div class="score-num">{wrong}</div><div class="score-lbl">Incorrect</div></div>
            </div>""", unsafe_allow_html=True)
            st.progress(pct / 100)

            verdict = "Outstanding!" if pct == 100 else "Great work!" if pct >= 80 else "Almost there!" if pct >= 50 else "Keep studying — you've got this!"
            st.markdown(f"<h3 style='margin-top:14px;line-height:1.4;word-break:break-word;font-family:var(--font-h)'>{emoji} {verdict}</h3>", unsafe_allow_html=True)
            st.divider()

            for i, r in enumerate(ev["results"]):
                icon    = "✅" if r["is_correct"] else "❌"
                preview = r['question'][:60] + "…" if len(r['question']) > 60 else r['question']
                with st.expander(f"{icon}  Q{i+1} — {preview}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""<div class="result-col">
                            <div class="result-label">Your Answer</div>
                            <div class="result-text" style="color:{'#34d399' if r['is_correct'] else '#fb7185'};">{r['user_answer']}</div>
                        </div>""", unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"""<div class="result-col">
                            <div class="result-label">Correct Answer</div>
                            <div class="result-text" style="color:#34d399;">{r['correct_answer']}</div>
                        </div>""", unsafe_allow_html=True)
                    if r["explanation"]:
                        st.info(r["explanation"])

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("New Quiz", type="secondary"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted  = False
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# FLASHCARDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🃏  Flashcards":
    page_header("Flashcards", "Tap a card to flip · Mark what you know · Retry the rest")

    if not st.session_state.uploaded_files:
        no_docs_warning()
    else:
        st.markdown('<div class="ctrl-card">', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([3, 1, 1])
        with fc1:
            fc_topic = st.text_input("Topic", key="fc_topic_input", placeholder="e.g. Git commands, Neural Networks…")
        with fc2:
            num_cards = st.selectbox("Cards", [5, 10, 15, 20], index=1)
        with fc3:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_btn = st.button("Generate", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if gen_btn:
            if not fc_topic.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating flashcards…"):
                    cards = generate_flashcards(fc_topic, num_cards)
                st.session_state.fc_cards   = cards
                st.session_state.fc_index   = 0
                st.session_state.fc_known   = set()
                st.session_state.fc_unknown = set()
                st.session_state.fc_flipped = False
                st.session_state.fc_topic   = fc_topic
                st.rerun()

        cards = st.session_state.fc_cards

        if not cards:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">🃏</div>
                <div class="msg">Enter a topic above and hit Generate</div>
            </div>""", unsafe_allow_html=True)
        else:
            idx     = st.session_state.fc_index
            total   = len(cards)
            card    = cards[idx]
            known   = st.session_state.fc_known
            unknown = st.session_state.fc_unknown
            is_flip = st.session_state.fc_flipped

            remaining = total - len(known) - len(unknown)

            # Pills — stable size via min-width in CSS, no FOUC
            st.markdown(f"""
            <div class="fc-pills">
                <span class="fc-pill g">✓ {len(known)}&nbsp;Know it</span>
                <span class="fc-pill b">◌ {remaining}&nbsp;Remaining</span>
                <span class="fc-pill r">✗ {len(unknown)}&nbsp;Review</span>
            </div>""", unsafe_allow_html=True)

            # Progress dots
            dots = ""
            for i in range(total):
                if i == idx:       c, s = "#7c6fff", "13px"
                elif i in known:   c, s = "#34d399", "9px"
                elif i in unknown: c, s = "#fb7185", "9px"
                else:              c, s = "rgba(255,255,255,.13)", "9px"
                dots += f'<div style="width:{s};height:{s};border-radius:50%;background:{c};transition:all .3s;flex-shrink:0;"></div>'
            st.markdown(f'<div class="fc-dots">{dots}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-ctr">{idx+1} of {total}</div>', unsafe_allow_html=True)

            # ── Flip card (cached HTML) ────────────────────────────────────
            components.html(
                _build_flashcard_html(card["front"], card["back"], card.get("hint", ""), is_flip),
                height=260,
                scrolling=False,
            )

            # Reveal / flip back
            st.markdown("<br>", unsafe_allow_html=True)
            rb1, _ = st.columns([1, 4])
            with rb1:
                if st.button("Reveal" if not is_flip else "Flip back", use_container_width=True, type="secondary"):
                    st.session_state.fc_flipped = not st.session_state.fc_flipped
                    st.rerun()

            # Navigation row
            st.markdown("<br>", unsafe_allow_html=True)
            nb1, nb2, nb3, nb4, nb5 = st.columns(5)

            with nb1:
                if st.button("Prev", use_container_width=True, type="secondary", disabled=(idx == 0)):
                    st.session_state.fc_index  -= 1
                    st.session_state.fc_flipped = False
                    st.rerun()

            with nb2:
                if st.button("✓ Know it", use_container_width=True, disabled=(idx in known)):
                    st.session_state.fc_known.add(idx)
                    st.session_state.fc_unknown.discard(idx)
                    if idx + 1 < total:
                        st.session_state.fc_index  += 1
                        st.session_state.fc_flipped = False
                    st.rerun()

            with nb3:
                if st.button("◌ Learning", use_container_width=True, disabled=(idx in unknown)):
                    st.session_state.fc_unknown.add(idx)
                    st.session_state.fc_known.discard(idx)
                    if idx + 1 < total:
                        st.session_state.fc_index  += 1
                        st.session_state.fc_flipped = False
                    st.rerun()

            with nb4:
                if st.button("Next", use_container_width=True, type="secondary", disabled=(idx == total - 1)):
                    st.session_state.fc_index  += 1
                    st.session_state.fc_flipped = False
                    st.rerun()

            with nb5:
                if st.button("🔀 Shuffle", use_container_width=True, type="secondary"):
                    random.shuffle(st.session_state.fc_cards)
                    st.session_state.fc_index   = 0
                    st.session_state.fc_flipped = False
                    st.session_state.fc_known   = set()
                    st.session_state.fc_unknown = set()
                    st.rerun()

            # ── Round complete ─────────────────────────────────────────────
            if len(known) + len(unknown) == total:
                st.divider()
                pct_k = round(len(known) / total * 100)
                em    = "🏆" if pct_k >= 80 else "👍" if pct_k >= 50 else "📚"
                st.markdown(f"""
                <div class="score-row">
                    <div class="score-card green"><div class="score-num">{len(known)}</div><div class="score-lbl">Knew</div></div>
                    <div class="score-card blue"><div class="score-num">{pct_k}</div><div class="score-lbl" style="font-size:9px;margin-top:2px;">% Score</div></div>
                    <div class="score-card red"><div class="score-num">{len(unknown)}</div><div class="score-lbl">Review</div></div>
                </div>""", unsafe_allow_html=True)
                st.progress(pct_k / 100)
                st.markdown(f"<h3 style='margin-top:14px;line-height:1.4;word-break:break-word;font-family:var(--font-h)'>{em} Round complete!</h3>", unsafe_allow_html=True)

                if unknown:
                    st.divider()
                    st.markdown('<div class="review-title">Cards to review:</div>', unsafe_allow_html=True)
                    for i in sorted(unknown):
                        preview = cards[i]['front'][:50] + ('…' if len(cards[i]['front']) > 50 else '')
                        with st.expander(f"Q: {preview}"):
                            st.markdown(f'<div class="review-card-inner">{cards[i]["back"]}</div>', unsafe_allow_html=True)

                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("Retry missed", use_container_width=True, type="primary"):
                        missed = [cards[i] for i in sorted(unknown)]
                        st.session_state.fc_cards   = missed
                        st.session_state.fc_index   = 0
                        st.session_state.fc_known   = set()
                        st.session_state.fc_unknown = set()
                        st.session_state.fc_flipped = False
                        st.rerun()
                with rc2:
                    if st.button("New deck", use_container_width=True, type="secondary"):
                        st.session_state.fc_cards   = []
                        st.session_state.fc_index   = 0
                        st.session_state.fc_known   = set()
                        st.session_state.fc_unknown = set()
                        st.session_state.fc_flipped = False
                        st.rerun()