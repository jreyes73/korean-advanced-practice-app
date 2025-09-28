import io
import re
import random
import unicodedata as ud
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="JR's Korean Practice",
    page_icon="🇰🇷",
    layout="centered",
)

# -----------------------------
# Utilities
# -----------------------------
def to_col_idx(col: str) -> int:
    """
    Convert 'C' or '3' (1-based) to 0-based column index.
    """
    if not col:
        return 0
    s = str(col).strip()
    if s.isdigit():
        return max(0, int(s) - 1)
    acc = 0
    for ch in s.upper():
        if 'A' <= ch <= 'Z':
            acc = acc * 26 + (ord(ch) - 64)
    return max(0, acc - 1)

def collapse_spaces(s: str) -> str:
    return re.sub(r"[\s\u00A0\u200B-\u200D\uFEFF]+", " ", s).strip()

_PUNCT_RE = re.compile(r"""[.,!?;:~'"“”‘’()\[\]{}<>•·\-–—_/\\|＠@#%^&*+=`]+""")

def strip_punct(s: str) -> str:
    return _PUNCT_RE.sub("", s)

def normalize_answer(s: str, strict: bool = False, ignore_paren: bool = True) -> str:
    """
    Normalize Hangul to NFC, optionally strip parentheses, punctuation, and collapse spaces.
    """
    if not s:
        return ""
    s = ud.normalize("NFC", str(s))
    if ignore_paren:
        s = re.sub(r"\([^)]*\)", "", s)
        s = re.sub(r"（[^）]*）", "", s)  # fullwidth parentheses
    if not strict:
        s = strip_punct(s)
    return collapse_spaces(s)

def same_answer(user: str, gold_list: List[str], strict: bool, ignore_paren: bool) -> bool:
    u = normalize_answer(user, strict=strict, ignore_paren=ignore_paren)
    for gold in gold_list:
        g = normalize_answer(gold, strict=strict, ignore_paren=ignore_paren)
        if u == g:
            return True
    return False

def parse_variants(cell: Any) -> List[str]:
    """
    Split a Korean cell into multiple acceptable answers by ; / | ｜ or comma.
    """
    s = str(cell or "").strip()
    if not s:
        return []
    parts = re.split(r"\s*[,;/｜|]\s*", s)
    return [p for p in (p.strip() for p in parts) if p]

def df_from_upload(file_bytes: bytes, filename: str, sheet_name: Optional[str], has_header: bool) -> pd.DataFrame:
    """
    Read CSV or Excel into a DataFrame. Returns all columns as strings.
    """
    fn = filename.lower()
    header = 0 if has_header else None
    if fn.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), header=header, dtype=str, encoding="utf-8")
    elif fn.endswith(".xlsx") or fn.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        target = sheet_name or xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=target, header=header, dtype=str)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

def get_sheet_names(file_bytes: bytes, filename: str) -> List[str]:
    fn = filename.lower()
    if fn.endswith(".xlsx") or fn.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        return xls.sheet_names
    return []

# -----------------------------
# State Helpers
# -----------------------------
def init_state():
    defaults = {
        "data": [],            # list[dict]: {en, ko, variants}
        "queue": [],           # list[int] indices into data
        "idx": -1,             # current position in queue
        "last": None,          # current item
        "stats": {"correct": 0, "wrong": 0, "attempted": 0, "total": 0, "streak": 0},
        "history": [],         # list[dict]: {ok, en, your, gold}
        "wrong_log": [],       # list[dict]: {en, correct, your}
        "file_bytes": None,
        "filename": "",
        "sheet_names": [],
        "chosen_sheet": None,
        "col_en": "C",
        "col_ko": "D",
        "has_header": True,
        "strict": False,
        "ignore_paren": True,
        "requeue_wrong": True,
        # Default mode = Flashcards
        "mode": "Flashcards",
        "flashcard_show": False,
        # Shuffle setting (checkbox) used by Start Again / Load Data / Load Demo
        "shuffle_on_start": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_stats():
    st.session_state["stats"] = {
        "correct": 0, "wrong": 0, "attempted": 0,
        "total": len(st.session_state["data"]), "streak": 0
    }

def start_run(shuffle: Optional[bool] = None):
    """
    Start/reset a run. If shuffle is None, use the Settings checkbox (shuffle_on_start).
    """
    if not st.session_state["data"]:
        return
    if shuffle is None:
        shuffle = st.session_state.get("shuffle_on_start", True)

    st.session_state["queue"] = list(range(len(st.session_state["data"])))
    if shuffle:
        random.shuffle(st.session_state["queue"])
    st.session_state["idx"] = 0
    st.session_state["history"].clear()
    st.session_state["wrong_log"].clear()
    st.session_state["flashcard_show"] = False
    reset_stats()
    update_last()

def update_last():
    q, i = st.session_state["queue"], st.session_state["idx"]
    if 0 <= i < len(q):
        st.session_state["last"] = st.session_state["data"][q[i]]
    else:
        st.session_state["last"] = None

def push_history(ok: bool, en: str, your: str, gold_list: List[str]):
    st.session_state["history"].insert(0, {
        "ok": ok, "en": en, "your": your or "(blank)", "gold": " ; ".join(gold_list)
    })

# -----------------------------
# Quiz + Flashcard Logic
# -----------------------------
def evaluate_answer(user_input: str):
    """
    Quiz mode evaluation: compare typed answer with variants.
    """
    last = st.session_state["last"]
    if not last:
        return
    strict = st.session_state["strict"]
    ignore_paren = st.session_state["ignore_paren"]
    ok = same_answer(user_input, last["variants"], strict=strict, ignore_paren=ignore_paren)

    st.session_state["stats"]["attempted"] += 1
    if ok:
        st.session_state["stats"]["correct"] += 1
        st.session_state["stats"]["streak"] += 1
    else:
        st.session_state["stats"]["wrong"] += 1
        st.session_state["stats"]["streak"] = 0
        st.session_state["wrong_log"].append({
            "en": last["en"], "correct": " ; ".join(last["variants"]), "your": user_input or ""
        })

    push_history(ok, last["en"], user_input, last["variants"])

    cur = st.session_state["idx"]
    if (not ok) and st.session_state["requeue_wrong"]:
        rest = max(cur + 1, 1)
        insert_at = random.randint(rest, len(st.session_state["queue"])) if rest < len(st.session_state["queue"]) else len(st.session_state["queue"])
        st.session_state["queue"].insert(insert_at, st.session_state["queue"][cur])

    st.session_state["idx"] += 1
    st.session_state["flashcard_show"] = False
    update_last()

def mark_flashcard(ok: bool):
    """
    Flashcard mode: user self-evaluates after seeing the answer.
    """
    last = st.session_state["last"]
    if not last:
        return
    st.session_state["stats"]["attempted"] += 1
    if ok:
        st.session_state["stats"]["correct"] += 1
        st.session_state["stats"]["streak"] += 1
        your = "✓ (flashcard)"
    else:
        st.session_state["stats"]["wrong"] += 1
        st.session_state["stats"]["streak"] = 0
        your = "✗ (flashcard)"
        st.session_state["wrong_log"].append({
            "en": last["en"], "correct": " ; ".join(last["variants"]), "your": ""
        })

    push_history(ok, last["en"], your, last["variants"])

    cur = st.session_state["idx"]
    if (not ok) and st.session_state["requeue_wrong"]:
        rest = max(cur + 1, 1)
        insert_at = random.randint(rest, len(st.session_state["queue"])) if rest < len(st.session_state["queue"]) else len(st.session_state["queue"])
        st.session_state["queue"].insert(insert_at, st.session_state["queue"][cur])

    st.session_state["idx"] += 1
    st.session_state["flashcard_show"] = False
    update_last()

def show_answer_text() -> str:
    last = st.session_state["last"]
    if not last:
        return ""
    return " ; ".join(last["variants"])

def load_parsed_rows(df: pd.DataFrame, col_en_idx: int, col_ko_idx: int, has_header: bool) -> List[Dict[str, Any]]:
    """
    Extract (en, ko) rows from df using position-based indices (iloc).
    Skip rows with missing English or Korean.
    """
    rows = []
    for _, row in df.iterrows():
        try:
            en = str(row.iloc[col_en_idx]).strip()
            ko = str(row.iloc[col_ko_idx]).strip()
        except Exception:
            continue
        if not en or not ko or en.lower() == "nan" or ko.lower() == "nan":
            continue
        variants = parse_variants(ko)
        if not variants:
            variants = [ko]
        rows.append({"en": en, "ko": ko, "variants": variants})
    return rows

def export_wrong_csv() -> bytes:
    wrong = st.session_state["wrong_log"]
    if not wrong:
        return b""
    df = pd.DataFrame(wrong)
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------
# App State Init
# -----------------------------
init_state()

# -----------------------------
# Sidebar Controls (Data + Settings)
# -----------------------------
with st.sidebar:
    st.header("Mode")
    st.session_state["mode"] = st.selectbox(
        "Choose a mode",
        ["Quiz (Type Korean)", "Flashcards"],
        index=0 if st.session_state["mode"] == "Quiz (Type Korean)" else 1
    )
    st.caption("You can switch modes anytime. Stats apply to the current run.")

    st.divider()
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV or Excel (.xlsx/.xls)", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        st.session_state["file_bytes"] = uploaded.getvalue()
        st.session_state["filename"] = uploaded.name
        st.session_state["sheet_names"] = get_sheet_names(st.session_state["file_bytes"], st.session_state["filename"])
        if st.session_state["sheet_names"]:
            st.session_state["chosen_sheet"] = st.selectbox("Sheet (Excel only)", st.session_state["sheet_names"], index=0)
        else:
            st.session_state["chosen_sheet"] = None

    st.session_state["col_en"] = st.text_input("English Column (letter or 1-based index)", value=st.session_state["col_en"])
    st.session_state["col_ko"] = st.text_input("Korean Column (letter or 1-based index)", value=st.session_state["col_ko"])
    st.session_state["has_header"] = st.checkbox("First row is header", value=st.session_state["has_header"])

    # -------------------
    # Settings
    # -------------------
    st.divider()
    st.header("Settings")
    st.session_state["strict"] = st.checkbox("Strict exact match (quiz mode)", value=st.session_state["strict"])
    st.session_state["ignore_paren"] = st.checkbox("Ignore parentheses (quiz mode)", value=st.session_state["ignore_paren"])
    st.session_state["requeue_wrong"] = st.checkbox("Re-queue wrong answers", value=st.session_state["requeue_wrong"])

    # Shuffle checkbox (controls Start Again / Load Data / Load Demo)
    st.session_state["shuffle_on_start"] = st.checkbox(
        "Shuffle",
        value=st.session_state["shuffle_on_start"]
    )

    st.divider()
    # Start Again (uses shuffle setting)
    if st.button("Start Again 🔁"):
        start_run(shuffle=None)

    st.divider()
    # Load controls
    load_cols = st.columns(2)
    with load_cols[0]:
        if st.button("Load Data", type="secondary"):
            try:
                if not st.session_state["file_bytes"]:
                    st.error("Please upload a CSV or Excel file first.")
                else:
                    df = df_from_upload(
                        st.session_state["file_bytes"],
                        st.session_state["filename"],
                        st.session_state["chosen_sheet"],
                        st.session_state["has_header"],
                    )
                    en_idx = to_col_idx(st.session_state["col_en"])
                    ko_idx = to_col_idx(st.session_state["col_ko"])
                    parsed = load_parsed_rows(df, en_idx, ko_idx, st.session_state["has_header"])
                    if not parsed:
                        st.warning("No usable rows found. Check your columns (English=C, Korean=D) and header option.")
                    else:
                        st.session_state["data"] = parsed
                        start_run(shuffle=None)  # respect Shuffle checkbox
                        st.success(f"Loaded {len(parsed)} items. Quiz started.")
            except Exception as e:
                st.exception(e)
    with load_cols[1]:
        if st.button("Load Demo", type="secondary"):
            st.session_state["data"] = [
                {"en": "Hello", "ko": "안녕하세요", "variants": ["안녕하세요", "안녕"]},
                {"en": "Thank you", "ko": "감사합니다", "variants": ["감사합니다", "고마워요"]},
                {"en": "Good morning", "ko": "좋은 아침", "variants": ["좋은 아침"]},
            ]
            start_run(shuffle=None)  # respect Shuffle checkbox
            st.success("Demo items loaded.")

# -----------------------------
# Main Panel
# -----------------------------
st.title("JR's Korean Practice")
st.caption(
    "Choose Flashcards or Quiz (Type Korean) in the sidebar. Multiple correct answers are allowed via ';' or '/' in the Korean cell."
)

# Progress + stats
stats = st.session_state["stats"]
progress = 0
if stats["total"]:
    progress = min(100, round(100 * stats["attempted"] / stats["total"]))
st.progress(progress / 100.0)
st.caption(f"Progress: {stats['attempted']} / {stats['total']}")

stat_cols = st.columns(4)
stat_cols[0].metric("Correct", stats["correct"])
stat_cols[1].metric("Wrong", stats["wrong"])
acc = round(100 * stats["correct"] / stats["attempted"]) if stats["attempted"] else 0
stat_cols[2].metric("Accuracy", f"{acc}%")
stat_cols[3].metric("Streak", stats["streak"])

st.divider()

# Current prompt (shared)
last = st.session_state["last"]
if last is None and st.session_state["data"]:
    st.success("🎉 Done! You reached the end of the queue.")
elif not st.session_state["data"]:
    st.info("Upload data (or click Load Demo in the sidebar) to begin.")
else:
    st.subheader("Prompt")
    st.markdown(f"**EN:** {last['en']}")

    # -------------------------
    # Mode: Quiz (Type Korean)
    # -------------------------
    if st.session_state["mode"] == "Quiz (Type Korean)":
        # Form so Enter submits; only Submit is inside (clears the input)
        with st.form("answer_form", clear_on_submit=True):
            answer = st.text_input("Type Korean here…", key="answer_box")
            submitted = st.form_submit_button("Submit ✅")
        if submitted:
            evaluate_answer(answer)
            st.rerun()

        # Actions outside form so they don't clear the input
        c1, c2 = st.columns([1, 1])
        if c1.button("Show Answer 💡"):
            ans = show_answer_text()
            if ans:
                st.info(f"**Answer:** {ans}")
            else:
                st.info("No current item.")
        if c2.button("Skip ⏭️"):
            cur = st.session_state["idx"]
            if 0 <= cur < len(st.session_state["queue"]):
                rest = max(cur + 1, 1)
                insert_at = random.randint(rest, len(st.session_state["queue"])) if rest < len(st.session_state["queue"]) else len(st.session_state["queue"])
                st.session_state["queue"].insert(insert_at, st.session_state["queue"][cur])
                st.session_state["idx"] += 1
                st.session_state["flashcard_show"] = False
                update_last()
                st.rerun()

    # -------------------------
    # Mode: Flashcards (default)
    # -------------------------
    else:
        if not st.session_state["flashcard_show"]:
            if st.button("Show Answer 💡"):
                st.session_state["flashcard_show"] = True
                st.rerun()
            st.caption("Tip: After revealing, mark yourself Correct or Wrong.")
        else:
            st.markdown(f"**KO:** {show_answer_text()}")

            c1, c2, c3 = st.columns([1, 1, 1])
            if c1.button("I was correct ✅"):
                mark_flashcard(True)
                st.rerun()
            if c2.button("I was wrong ❌"):
                mark_flashcard(False)
                st.rerun()
            if c3.button("Skip ⏭️"):
                cur = st.session_state["idx"]
                if 0 <= cur < len(st.session_state["queue"]):
                    rest = max(cur + 1, 1)
                    insert_at = random.randint(rest, len(st.session_state["queue"])) if rest < len(st.session_state["queue"]) else len(st.session_state["queue"])
                    st.session_state["queue"].insert(insert_at, st.session_state["queue"][cur])
                    st.session_state["idx"] += 1
                    st.session_state["flashcard_show"] = False
                    update_last()
                    st.rerun()

# History
st.divider()
st.subheader("History")
if not st.session_state["history"]:
    st.caption("Your previous answers will appear here.")
else:
    for item in st.session_state["history"]:
        color = "#bbf7d0" if item["ok"] else "#fecaca"
        st.markdown(f"**EN:** {item['en']}")
        st.markdown(
            f"<span style='color:{color}'><b>Your:</b> {item['your']}</span>",
            unsafe_allow_html=True
        )
        if not item["ok"]:
            st.markdown(
                f"<span style='color:#fcd34d'><b>Expected:</b> {item['gold']}</span>",
                unsafe_allow_html=True
            )
        st.markdown("---")

# Export wrong answers
st.divider()
btn_cols = st.columns([1, 2])
with btn_cols[0]:
    csv_bytes = export_wrong_csv()
    st.download_button(
        label="Export Wrong Answers CSV",
        data=csv_bytes if csv_bytes else b"",
        file_name="wrong-answers.csv",
        mime="text/csv",
        disabled=(len(st.session_state["wrong_log"]) == 0),
    )
