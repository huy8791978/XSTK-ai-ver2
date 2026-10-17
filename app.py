"""
XSTK Tutor - trợ lí AI học Xác suất thống kê (Streamlit).
Chạy thử : streamlit run app.py
Triển khai: Streamlit Community Cloud, Main file path = app.py
"""
import hashlib
import io
import json
import re
import textwrap
import datetime as dt
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors, types
from PIL import Image, ImageOps

# ───────────────────────── CẤU HÌNH ─────────────────────────
MODEL = "gemini-3.5-flash-lite"  # đổi sang "gemini-2.5-pro" nếu cần chấm kĩ hơn (chậm và đắt hơn)
# Cách 1 (khuyên dùng): đặt GEMINI_API_KEY trong Streamlit Cloud > Settings > Secrets.
# Cách 2: dán khóa (dạng AIza...) vào dòng dưới. Chỉ làm vậy khi repo GitHub ở chế độ PRIVATE:
# khóa nằm trong repo public sẽ bị lộ và bị thu hồi.
EMBEDDED_KEY = "DAN_API_KEY_VAO_DAY"
PROFILE_DIR = Path("profiles")  # shortcut: đĩa của Streamlit Cloud bị xoá khi app khởi động lại, nên có nút tải/khôi phục hồ sơ

TOPICS = [
    "Biến cố và xác suất cổ điển",
    "Xác suất có điều kiện, công thức nhân",
    "Công thức xác suất toàn phần và Bayes",
    "Biến ngẫu nhiên rời rạc, kì vọng, phương sai",
    "Phân phối nhị thức",
    "Phân phối Poisson",
    "Biến ngẫu nhiên liên tục, phân phối chuẩn",
    "Số đặc trưng của mẫu số liệu ghép nhóm",
    "Ước lượng khoảng",
    "Kiểm định giả thuyết",
    "Tương quan và hồi quy tuyến tính",
]
ERR = [
    "Hiểu sai khái niệm", "Áp dụng sai công thức", "Sai tính toán",
    "Thiếu hoặc sai lập luận", "Đọc sai đề hoặc dữ kiện",
    "Trình bày thiếu chặt chẽ", "Khác",
]

# ───────────────────────── PROMPT ─────────────────────────
BASE = (
    "Bạn là gia sư Xác suất thống kê (XSTK) cho học sinh, sinh viên Việt Nam. "
    "Luôn trả lời bằng tiếng Việt, ngắn gọn, chính xác, thân thiện, không dùng emoji. "
    "Công thức toán viết bằng LaTeX: $...$ trong dòng, $$...$$ cho dòng riêng. "
)
JSON_RULE = (
    r' Chỉ trả về MỘT đối tượng JSON hợp lệ, không có chữ nào bên ngoài, không dùng khối mã. '
    r'Trong chuỗi JSON, mọi dấu gạch chéo ngược của LaTeX phải viết đôi (ví dụ "\\frac{1}{2}"). '
    r'Xuống dòng trong chuỗi dùng \n.'
)
SOLVE_FLOW = (
    "Quy trình giải toán 4 bước: (1) Tìm hiểu đề: phân biệt cái đã cho, cái phải tìm, kí hiệu hoá; "
    "(2) Xây dựng chương trình giải: quy về bài toán đơn giản quen thuộc, chọn kiến thức gần nhất; "
    "(3) Trình bày lời giải; (4) Kiểm tra, nhìn lại, rút phương pháp, mở rộng. "
)
SYS_HINT = BASE + SOLVE_FLOW + (
    "Nhiệm vụ: đưa CHIẾN LƯỢC giải bài theo các bước 1, 2 và gợi ý bước 4 (cách tự kiểm tra). "
    "Nêu: cái đã cho, cái cần tìm, dạng bài, kiến thức/công thức cần dùng, cách chia nhỏ bài toán. "
    "TUYỆT ĐỐI không đưa đáp án, không tính ra kết quả số, không thay số hoàn chỉnh vào công thức, "
    "không viết lời giải từng bước. Kết thúc bằng 1-2 câu hỏi gợi mở để học sinh tự làm."
)
SYS_SOL = BASE + SOLVE_FLOW + (
    "Nhiệm vụ: viết BÀI GIẢI CHI TIẾT theo 4 bước, rõ ràng từng phép biến đổi, in đậm kết quả cuối, "
    "cuối cùng nêu 1-2 sai lầm thường gặp của dạng bài này."
)
SYS_GRADE = BASE + (
    "Bạn chấm bài làm XSTK của học sinh (có thể là ảnh chụp chữ viết tay hoặc PDF: hãy đọc/OCR cẩn thận). "
    "Tự giải đề trước để biết kết quả đúng, nhưng CHỈ chấm điểm và chỉ ra chỗ sai: KHÔNG viết lại lời giải đúng, "
    "KHÔNG nêu kết quả cuối cùng đúng. Với mỗi lỗi, nói sai ở bước nào, vì sao, và gợi hướng sửa bằng một câu hỏi "
    "hoặc nhắc kiến thức. Chấm thang 10, ghi nhận phần đúng. Nếu không đọc được bài làm thì diem = 0 và nói rõ. "
    'Trường "loai" của loi_sai phải là một trong: ' + "; ".join(ERR) + "." + JSON_RULE +
    ' Cấu trúc: {"van_ban_chep_lai":"nội dung bài làm đọc được", "diem":số, "nhan_xet":"2-4 câu", '
    '"buoc":[{"ten":"...","dung":true,"ghi_chu":"..."}], "loi_sai":[{"loai":"...","mo_ta":"..."}], '
    '"lo_hong":["kiến thức còn thiếu"], "chu_de":"tên chủ đề ngắn"}'
)
_PRACTICE = (
    "Soạn bài tập XSTK bám sát chương trình, số liệu cụ thể, đủ dữ kiện, giải được, có lời giải đúng. "
)
SYS_MC = BASE + _PRACTICE + (
    "Soạn đúng 5 câu trắc nghiệm 4 phương án; trộn nhận diện khái niệm và vận dụng. Phương án nhiễu phản ánh sai lầm "
    "phổ biến. Vị trí đáp án đúng phân bố ngẫu nhiên giữa A-D. 'chien_luoc' là gợi ý theo bước 1-2 của quy trình giải, "
    "TUYỆT ĐỐI không tiết lộ đáp án hay kết quả số. 'sai_lam' chỉ gồm các phương án sai." + JSON_RULE +
    ' Cấu trúc: {"cau":[{"de":"...","lua_chon":{"A":"..","B":"..","C":"..","D":".."},"dap_an":"B",'
    '"chien_luoc":"...","loi_giai":"lời giải chi tiết nêu rõ đáp án","sai_lam":{"A":{"loai":"(một trong: '
    + "; ".join(ERR) + ')","mo_ta":"vì sao học sinh hay chọn nhầm"}}}]}'
)
SYS_ES = BASE + _PRACTICE + (
    "Soạn đúng 5 câu tự luận: 2 cơ bản, 2 trung bình, 1 gắn thực tiễn. 'chien_luoc' theo bước 1-2 của quy trình giải, "
    "TUYỆT ĐỐI không tiết lộ kết quả. 'tieu_chi' là thang điểm 10 tham khảo." + JSON_RULE +
    ' Cấu trúc: {"cau":[{"de":"...","chien_luoc":"...","loi_giai":"lời giải chi tiết 4 bước","tieu_chi":"..."}]}'
)
ORAL_CONCEPT = (
    "Thứ tự 5 câu hỏi (học khái niệm): (1) phát biểu định nghĩa và dấu hiệu bản chất; (2) thuật ngữ, kí hiệu then chốt "
    "hoặc phân biệt với khái niệm dễ nhầm; (3) nhận diện qua ví dụ và phản ví dụ đơn giản; (4) mối liên hệ logic với khái "
    "niệm khác; (5) vận dụng trong một tình huống thực tiễn."
)
ORAL_THEOREM = (
    "Thứ tự 5 câu hỏi (học định lí, công thức): (1) phát biểu định lí/công thức; (2) chỉ rõ giả thiết, kết luận, điều kiện "
    "áp dụng; (3) luận cứ và ý tưởng chứng minh; (4) cấu trúc logic (trường hợp đặc biệt, mệnh đề đảo); (5) vận dụng "
    "vào bài toán hoặc thực tiễn."
)
SYS_ORAL_GRADE = BASE + (
    "Chấm phần vấn đáp lí thuyết XSTK dựa trên bản ghi. Câu chưa trả lời hoặc trả lời sai hoàn toàn tính 0. "
    "Chỉ ra lỗ hổng kiến thức và sai lầm cụ thể của học sinh. " 'Trường "loai" của loi_sai là một trong: '
    + "; ".join(ERR) + "." + JSON_RULE +
    ' Cấu trúc: {"diem":số 0-10,"nhan_xet":"2-4 câu","tung_cau":[{"cau":"tóm tắt câu hỏi","diem":số,"nhan_xet":"..."}],'
    '"loi_sai":[{"loai":"...","mo_ta":"..."}],"lo_hong":["..."],"goi_y_on_tap":["..."],"chu_de":"tên chủ đề"}'
)
SYS_MAP = BASE + (
    "Tạo bản đồ tư duy cho một chủ đề XSTK. Gốc là tên chủ đề. 5-6 nhánh cấp 1 (ví dụ: Khái niệm; Công thức và tính chất; "
    "Dạng bài và phương pháp; Sai lầm thường gặp; Ví dụ và ứng dụng; Liên hệ chủ đề khác), mỗi nhánh 2-4 nút con, "
    "có thể thêm cấp 3. Tên nút tối đa 8 từ, văn bản thuần, không LaTeX, không dấu nháy." + JSON_RULE +
    ' Cấu trúc: {"ten":"gốc","con":[{"ten":"...","con":[{"ten":"..."}]}]}'
)
SYS_PLAN = BASE + (
    "Bạn nhận thống kê hồ sơ học XSTK của một học sinh (JSON). Chỉ dựa vào số liệu đó, không bịa. Trả lời dưới 350 từ, gồm: "
    "**Nhận định** (2-4 gạch đầu dòng nêu quy luật lỗi sai xuyên suốt); **Ưu tiên ôn** (tối đa 3 chủ đề, vì sao); "
    "**Kế hoạch 7 ngày** (mỗi ngày một việc cụ thể, dùng các chức năng: Chấm bài, Luyện tập, Vấn đáp, Bản đồ tư duy); "
    "**Cách tự kiểm tra tiến bộ**. Nếu dữ liệu còn ít, nói rõ và đề xuất làm thêm."
)

S = st.session_state


# ───────────────────────── GỌI AI (GEMINI) ─────────────────────────
def _key():
    try:
        k = st.secrets["GEMINI_API_KEY"]
    except Exception:
        k = ""
    return str(k or EMBEDDED_KEY)


def _has_key():
    return _key().startswith("AIza")


@st.cache_resource
def _client(key):
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180_000))


def ask(system, content=None, msgs=None, max_tokens=3000, json_mode=False):
    """Trả về text, hoặc None (đã hiện lỗi thân thiện, không lộ khóa)."""
    if not _has_key():
        st.error("Chưa cấu hình API key. Quản trị viên cần thêm khóa vào Secrets hoặc vào mã nguồn.")
        return None
    if msgs:
        content = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                 parts=[types.Part(text=m["content"])]) for m in msgs]
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=min(max_tokens * 3, 60000),  # shortcut: Gemini 2.5 tính cả token "suy nghĩ" vào giới hạn, nên nhân 3
        response_mime_type="application/json" if json_mode else None)
    try:
        r = _client(_key()).models.generate_content(model=MODEL, contents=content, config=cfg)
    except errors.APIError as e:
        code = getattr(e, "code", None)
        if code == 429:
            st.error("Hệ thống đang quá tải hoặc hết hạn mức. Hãy đợi ít giây rồi thử lại.")
        elif code in (400, 401, 403):
            st.error(f"Yêu cầu bị từ chối (mã {code}). Kiểm tra API key hoặc tệp tải lên.")
        else:
            st.error(f"Lỗi kết nối AI (mã {code}). Hãy thử lại.")
    except Exception as e:
        st.error(f"Lỗi kết nối AI ({type(e).__name__}). Hãy thử lại.")
    else:
        if r.text:
            return r.text.strip()
        st.error("AI không trả lời (có thể do bộ lọc an toàn hoặc bị cắt). Hãy diễn đạt lại và thử lại.")
    return None


def _fix(o):
    """JSON nuốt \\f \\t \\r \\b của LaTeX (\\frac, \\text, \\right, \\bar): khôi phục lại."""
    if isinstance(o, str):
        return o.replace("\x0c", "\\f").replace("\x08", "\\b").replace("\t", "\\t").replace("\r", "\\r")
    if isinstance(o, list):
        return [_fix(i) for i in o]
    if isinstance(o, dict):
        return {k: _fix(v) for k, v in o.items()}
    return o


def ask_json(system, content, max_tokens=3000):
    for _ in range(2):
        t = ask(system, content, max_tokens=max_tokens, json_mode=True)
        if t is None:
            return None
        try:
            return _fix(json.loads(t[t.index("{"): t.rindex("}") + 1], strict=False))
        except ValueError:
            continue
    st.error("AI trả về dữ liệu không đọc được. Hãy thử lại.")
    return None


# ───────────────────────── FILE / OCR ─────────────────────────
def _T(s):
    return types.Part(text=s)


def fblocks(files):
    """Ảnh -> JPEG thu nhỏ (ảnh điện thoại rất nặng); PDF gửi nguyên. Gemini đọc chữ viết tay (OCR)."""
    out = []
    for f in files or []:
        data = f.getvalue()
        if f.name.lower().endswith(".pdf"):
            if len(data) > 20_000_000:
                st.warning(f"{f.name} lớn hơn 20 MB, bỏ qua.")
                continue
            out.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
            continue
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception:
            st.warning(f"Không đọc được ảnh {f.name}.")
            continue
        im.thumbnail((1800, 1800))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=85)
        out.append(types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg"))
    return out


def build(prob_txt="", prob_files=None, work_txt="", work_files=None, extra=""):
    c = []
    pf = fblocks(prob_files)
    if prob_txt.strip() or pf:
        c.append(_T("ĐỀ BÀI:"))
        if prob_txt.strip():
            c.append(_T(prob_txt.strip()))
        c += pf
    wf = fblocks(work_files)
    if work_txt.strip() or wf:
        c.append(_T("BÀI LÀM CỦA HỌC SINH:"))
        if work_txt.strip():
            c.append(_T(work_txt.strip()))
        c += wf
    if extra:
        c.append(_T(extra))
    return c


# ───────────────────────── HỒ SƠ ─────────────────────────
def _ppath(name):
    return PROFILE_DIR / (hashlib.sha1(name.strip().lower().encode()).hexdigest()[:12] + ".json")


def load_profile(name):
    try:
        p = json.loads(_ppath(name).read_text("utf-8"))
        if isinstance(p.get("events"), list):
            return p
    except (OSError, ValueError, AttributeError):
        pass
    return {"name": name, "created": dt.date.today().isoformat(), "events": []}


def save_profile(p):
    try:
        PROFILE_DIR.mkdir(exist_ok=True)
        _ppath(p["name"]).write_text(json.dumps(p, ensure_ascii=False), "utf-8")
    except OSError:
        pass  # shortcut: lưu đĩa thất bại thì vẫn giữ trong phiên và cho tải về


def num(x):
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def errs(x):
    out = []
    for e in x if isinstance(x, list) else []:
        if isinstance(e, str):
            e = {"mo_ta": e}
        if isinstance(e, dict):
            out.append({"loai": e.get("loai") if e.get("loai") in ERR else "Khác", "mo_ta": str(e.get("mo_ta", ""))[:300]})
    return out


def log(mode, topic, score, errors, gaps):
    P = S.profile
    gaps = [str(g)[:200] for g in gaps if g] if isinstance(gaps, list) else []
    P["events"].append({"t": dt.datetime.now().isoformat(timespec="seconds"), "mode": mode,
                        "chu_de": str(topic)[:80] or "Chưa phân loại", "diem": round(score, 1),
                        "loi_sai": errs(errors), "lo_hong": gaps})
    P["events"] = P["events"][-500:]
    save_profile(P)


def log_grade(mode, g):
    log(mode, g.get("chu_de") or "Chưa phân loại", num(g.get("diem")), g.get("loi_sai"), g.get("lo_hong"))


@st.dialog("Hồ sơ học XSTK", width="large")
def profile_dialog():
    P = S.profile
    ev = [e for e in P.get("events", []) if isinstance(e, dict)]
    st.markdown(f"**{P['name']}** · {len(ev)} lượt học")
    if not ev:
        st.info("Chưa có dữ liệu. Hãy chấm một bài, làm bài tập hoặc vấn đáp để hồ sơ bắt đầu ghi nhận.")
        return
    sc = [num(e.get("diem")) for e in ev]
    avg, rec, prev = sum(sc) / len(sc), sc[-5:], sc[-10:-5]
    ra = sum(rec) / len(rec)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số lượt", len(ev))
    c2.metric("Điểm trung bình", f"{avg:.1f}")
    c3.metric("5 lượt gần nhất", f"{ra:.1f}", f"{ra - sum(prev) / len(prev):+.1f}" if prev else None)
    st.markdown("**Diễn tiến điểm**")
    st.line_chart(pd.DataFrame({"Điểm": sc}), color="#0F5C5A")

    all_err = [x for e in ev for x in (e.get("loi_sai") or []) if isinstance(x, dict)]
    by_type = Counter(x.get("loai", "Khác") for x in all_err)
    if by_type:
        st.markdown("**Lỗi sai theo nhóm**")
        st.bar_chart(pd.Series(by_type, name="Số lần").sort_values(ascending=False), color="#B4531F")

    df = pd.DataFrame({"Chủ đề": [str(e.get("chu_de", "?")) for e in ev], "Điểm": sc})
    topic = (df.groupby("Chủ đề")["Điểm"].agg(["mean", "count"]).round(1).reset_index()
             .rename(columns={"mean": "Điểm TB", "count": "Lượt"}).sort_values("Điểm TB"))
    st.markdown("**Điểm theo chủ đề** (thấp nhất ở trên)")
    st.dataframe(topic, hide_index=True)

    gaps = Counter(str(g).strip() for e in ev for g in (e.get("lo_hong") or []) if g)
    if gaps:
        st.markdown("**Lỗ hổng kiến thức hay gặp**")
        for g, n in gaps.most_common(8):
            st.markdown(f"- {g}" + (f" (×{n})" if n > 1 else ""))
    if all_err:
        st.markdown("**Sai lầm gần đây**")
        for x in all_err[-5:][::-1]:
            st.markdown(f"- *{x.get('loai', 'Khác')}*: {x.get('mo_ta', '')}")

    st.divider()
    if st.button("Đề xuất phương án học cải thiện", icon=":material/route:", type="primary", key="b_plan"):
        data = {"so_luot": len(ev), "diem_tb": round(avg, 1), "diem_5_gan_nhat": round(ra, 1),
                "loi_theo_nhom": dict(by_type), "chu_de_yeu_nhat": topic.head(3).to_dict("records"),
                "lo_hong_hay_gap": dict(gaps.most_common(8)),
                "sai_lam_gan_day": [x.get("mo_ta", "") for x in all_err[-6:]]}
        with st.spinner("Đang phân tích hồ sơ..."):
            S.r_plan = ask(SYS_PLAN, json.dumps(data, ensure_ascii=False), max_tokens=1500)
    if S.get("r_plan"):
        st.markdown(S.r_plan)


# ───────────────────────── GIAO DIỆN CHUNG ─────────────────────────
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root{--paper:#F5F1E8;--card:#FBF9F4;--ink:#1F2A27;--muted:#5E6B66;--line:#D9D2C3;--teal:#0F5C5A;--clay:#B4531F;--amber:#8A6A12}
.stApp{background:var(--paper);color:var(--ink);font-family:'Be Vietnam Pro',system-ui,-apple-system,'Segoe UI',sans-serif}
.block-container{max-width:1100px;padding:2.2rem 1.1rem 5rem}
#MainMenu,footer{visibility:hidden}
h1,h2,h3{font-family:'Newsreader',Georgia,serif!important;color:var(--ink);letter-spacing:-.015em}
h1{font-size:clamp(2rem,5vw,3.2rem)!important;line-height:1.08!important;font-weight:600!important;margin:.2rem 0 .7rem!important;padding:0!important}
h2{font-size:clamp(1.5rem,3.4vw,2rem)!important;font-weight:600!important;padding-top:.4rem!important}
.eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:600}
.lead{color:var(--muted);font-size:1.05rem;line-height:1.65;max-width:34rem}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{width:100%;min-height:46px;border-radius:12px;font-weight:500;transition:transform .15s ease,border-color .15s ease}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);border-color:var(--teal)}
.stButton>button:active{transform:scale(.98)}
:focus-visible{outline:2px solid var(--teal)!important;outline-offset:2px}
.score{display:flex;align-items:baseline;gap:.3rem;font-family:'Newsreader',Georgia,serif;margin:.1rem 0 .5rem}
.score b{font-size:3.6rem;line-height:1;font-weight:600}.score span{color:var(--muted);font-size:1.2rem}
.score.teal b{color:var(--teal)}.score.amber b{color:var(--amber)}.score.clay b{color:var(--clay)}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.block-container{animation:rise .45s ease both}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>"""


def head(title, sub):
    st.markdown(f"## {title}")
    st.caption(sub)


def topic_pick(k):
    sel = st.selectbox("Chủ đề", TOPICS, key=f"{k}_sel")
    own = st.text_input("Hoặc nhập chủ đề riêng", key=f"{k}_own", placeholder="Ví dụ: Phân phối chuẩn tắc")
    return own.strip() or sel


def show_score(d):
    col = "teal" if d >= 7 else "amber" if d >= 5 else "clay"
    st.markdown(f'<div class="score {col}"><b>{d:g}</b><span>/10</span></div>', unsafe_allow_html=True)


def show_issues(g):
    if g.get("loi_sai"):
        st.markdown("**Sai lầm phát hiện**")
        for e in errs(g["loi_sai"]):
            st.markdown(f"- *{e['loai']}*: {e['mo_ta']}")
    if g.get("lo_hong"):
        st.markdown("**Lỗ hổng kiến thức**")
        for x in g["lo_hong"]:
            st.markdown(f"- {x}")


def show_grade(g):
    show_score(num(g.get("diem")))
    st.markdown(str(g.get("nhan_xet", "")))
    for s in g.get("buoc") or []:
        if isinstance(s, dict):
            tag = ":green[Đúng]" if s.get("dung") else ":red[Cần sửa]"
            st.markdown(f"{tag} · **{s.get('ten', '')}**: {s.get('ghi_chu', '')}")
    show_issues(g)
    if g.get("van_ban_chep_lai"):
        with st.expander("Nội dung AI đọc được từ bài làm (OCR)"):
            st.markdown(str(g["van_ban_chep_lai"]))


# ───────────────────────── TRANG: CHẤM BÀI ─────────────────────────
def page_tutor():
    head("Gia sư chấm bài", "Tải đề và bài làm. AI đọc chữ viết tay (OCR), chấm điểm và gợi ý hướng giải, không tiết lộ đáp án.")
    types = ["png", "jpg", "jpeg", "pdf"]
    with st.container(border=True):
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**1. Đề bài**")
            prob = st.text_area("Nhập đề bài", key="t_prob", height=150)
            pf = st.file_uploader("Hoặc tải ảnh/PDF đề bài", type=types, accept_multiple_files=True, key="t_pf")
        with b:
            st.markdown("**2. Bài làm của bạn**")
            wt = st.text_area("Gõ bài làm (không bắt buộc)", key="t_wt", height=150)
            wf = st.file_uploader("Tải ảnh chụp/PDF bài làm", type=types, accept_multiple_files=True, key="t_wf")
        has_p, has_w = bool(prob.strip() or pf), bool(wt.strip() or wf)
        c1, c2, c3 = st.columns(3)
        if c1.button("Gợi ý chiến lược", icon=":material/lightbulb:", key="b_t_hint"):
            if not has_p:
                st.warning("Hãy nhập hoặc tải đề bài trước.")
            else:
                with st.spinner("Đang phân tích đề..."):
                    S.r_hint = ask(SYS_HINT, build(prob, pf), max_tokens=1800)
        if c2.button("Chấm bài", icon=":material/task_alt:", type="primary", key="b_t_grade"):
            if not (has_p and has_w):
                st.warning("Cần có cả đề bài và bài làm.")
            else:
                with st.spinner("Đang đọc bài làm và chấm điểm..."):
                    g = ask_json(SYS_GRADE, build(prob, pf, wt, wf), max_tokens=3500)
                if g:
                    S.r_grade = g
                    log_grade("tutor", g)
        if c3.button("Hiện bài giải chi tiết", icon=":material/menu_book:", key="b_t_sol"):
            if not has_p:
                st.warning("Hãy nhập hoặc tải đề bài trước.")
            else:
                with st.spinner("Đang viết bài giải..."):
                    S.r_sol = ask(SYS_SOL, build(prob, pf), max_tokens=3500)
    shown = False
    if S.get("r_hint"):
        shown = True
        with st.container(border=True):
            st.markdown("#### Chiến lược giải")
            st.markdown(S.r_hint)
    if S.get("r_grade"):
        shown = True
        with st.container(border=True):
            st.markdown("#### Kết quả chấm")
            show_grade(S.r_grade)
    if S.get("r_sol"):
        shown = True
        with st.container(border=True):
            st.markdown("#### Bài giải chi tiết")
            st.markdown(S.r_sol)
    if shown and st.button("Xoá kết quả", key="b_t_clear"):
        for k in ("r_hint", "r_grade", "r_sol"):
            S.pop(k, None)
        st.rerun()


# ───────────────────────── TRANG: LUYỆN TẬP ─────────────────────────
def ok_mc(q):
    return (isinstance(q, dict) and q.get("de") and isinstance(q.get("lua_chon"), dict)
            and all(x in q["lua_chon"] for x in "ABCD") and q.get("dap_an") in list("ABCD"))


def toggle(k):
    S[k] = not S.get(k)


def mc_card(i, q, topic):
    k = f"pq_mc{i}"
    with st.container(border=True):
        st.markdown(f"**Câu {i + 1}.** {q['de']}")
        pick = st.radio("Chọn đáp án", list("ABCD"), index=None, key=k, label_visibility="collapsed",
                        format_func=lambda x: f"{x}. {q['lua_chon'][x]}")
        a, b, c = st.columns(3)
        a.button("Gợi ý chiến lược", icon=":material/lightbulb:", key=f"b_{k}_h", on_click=toggle, args=(f"{k}_h",))
        if b.button("Chấm bài", icon=":material/task_alt:", type="primary", key=f"b_{k}_g"):
            if pick is None:
                st.warning("Hãy chọn một đáp án.")
            else:
                S[f"{k}_g"] = pick
                if not S.get(f"{k}_l"):  # chỉ ghi hồ sơ một lần cho mỗi câu
                    S[f"{k}_l"] = 1
                    m = (q.get("sai_lam") or {}).get(pick)
                    log("practice", topic, 10.0 if pick == q["dap_an"] else 0.0,
                        [] if pick == q["dap_an"] else [m if m else {"loai": "Khác", "mo_ta": "Chọn sai đáp án"}], [])
        c.button("Hiện lời giải", icon=":material/menu_book:", key=f"b_{k}_s", on_click=toggle, args=(f"{k}_s",))
        gp = S.get(f"{k}_g")
        if gp:
            if gp == q["dap_an"]:
                st.success(f"Chính xác. Bạn đã chọn {gp}.")
            else:
                m = (q.get("sai_lam") or {}).get(gp)
                why = m.get("mo_ta", "") if isinstance(m, dict) else str(m or "")
                st.error(f"Chưa đúng (bạn chọn {gp}). {why}")
        if S.get(f"{k}_h"):
            st.info(q.get("chien_luoc", ""))
        if S.get(f"{k}_s"):
            st.success(f"**Đáp án: {q['dap_an']}**\n\n{q.get('loi_giai', '')}")


def es_card(i, q, topic):
    k = f"pq_es{i}"
    with st.container(border=True):
        st.markdown(f"**Câu {i + 1}.** {q['de']}")
        ans = st.text_area("Bài làm của bạn", key=f"{k}_a", height=130)
        with st.expander("Hoặc tải ảnh/PDF bài làm"):
            fl = st.file_uploader("Ảnh/PDF", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True,
                                  key=f"{k}_f", label_visibility="collapsed")
        a, b, c = st.columns(3)
        a.button("Gợi ý chiến lược", icon=":material/lightbulb:", key=f"b_{k}_h", on_click=toggle, args=(f"{k}_h",))
        if b.button("Chấm bài", icon=":material/task_alt:", type="primary", key=f"b_{k}_g"):
            if not (ans.strip() or fl):
                st.warning("Hãy nhập hoặc tải bài làm.")
            else:
                ref = f"ĐÁP ÁN THAM KHẢO VÀ THANG ĐIỂM (chỉ để chấm, không tiết lộ):\n{q.get('loi_giai', '')}\n{q.get('tieu_chi', '')}"
                with st.spinner("Đang chấm..."):
                    g = ask_json(SYS_GRADE, build(q["de"], None, ans, fl, ref), max_tokens=3500)
                if g:
                    g.setdefault("chu_de", topic)
                    S[f"{k}_g"] = g
                    log_grade("practice", g)
        c.button("Hiện lời giải", icon=":material/menu_book:", key=f"b_{k}_s", on_click=toggle, args=(f"{k}_s",))
        if S.get(f"{k}_g"):
            show_grade(S[f"{k}_g"])
        if S.get(f"{k}_h"):
            st.info(q.get("chien_luoc", ""))
        if S.get(f"{k}_s"):
            st.success(q.get("loi_giai", ""))


def page_practice():
    head("Luyện tập", "5 câu trắc nghiệm và 5 câu tự luận theo chủ đề bạn chọn. Chiến lược gợi ý không chứa đáp án.")
    with st.container(border=True):
        a, b = st.columns([2, 1], gap="large")
        with a:
            topic = topic_pick("p")
        with b:
            level = st.radio("Mức độ", ["Cơ bản", "Trung bình", "Nâng cao"], index=1, key="p_lv")
        if st.button("Tạo bộ 10 bài tập", icon=":material/auto_awesome:", type="primary", key="b_p_gen"):
            for kk in [x for x in S if x.startswith("pq_")]:
                del S[kk]
            req = f"Chủ đề: {topic}. Mức độ: {level}."
            with st.spinner("Đang soạn 5 câu trắc nghiệm..."):
                mc = ask_json(SYS_MC, req, max_tokens=7000)
            with st.spinner("Đang soạn 5 câu tự luận..."):
                es = ask_json(SYS_ES, req, max_tokens=7000)
            mcq = [q for q in ((mc or {}).get("cau") or []) if ok_mc(q)][:5]
            esq = [q for q in ((es or {}).get("cau") or []) if isinstance(q, dict) and q.get("de")][:5]
            if mcq or esq:
                S.r_pq = {"topic": topic, "mc": mcq, "es": esq}
            else:
                st.error("Không tạo được bài tập. Hãy thử lại.")
    pq = S.get("r_pq")
    if not pq:
        st.info("Chọn chủ đề rồi bấm “Tạo bộ 10 bài tập”.")
        return
    t1, t2 = st.tabs([f"Trắc nghiệm ({len(pq['mc'])})", f"Tự luận ({len(pq['es'])})"])
    with t1:
        for i, q in enumerate(pq["mc"]):
            mc_card(i, q, pq["topic"])
    with t2:
        for i, q in enumerate(pq["es"]):
            es_card(i, q, pq["topic"])


# ───────────────────────── TRANG: VẤN ĐÁP ─────────────────────────
def oral_sys(O):
    n = sum(m["role"] == "assistant" for m in O["msgs"])
    plan = ORAL_CONCEPT if O["kind"].startswith("Khái") else ORAL_THEOREM
    return BASE + (
        f"Bạn là giám khảo vấn đáp lí thuyết XSTK, chủ đề: {O['topic']}. {plan} "
        "Mỗi lượt chỉ đặt MỘT câu hỏi ngắn, tổng cộng 5 câu theo thứ tự trên. Sau mỗi câu trả lời của học sinh, nhận xét "
        "1-2 câu (đúng, chưa đủ hay sai ở đâu, có gợi mở), KHÔNG giảng đáp án đầy đủ, rồi hỏi câu tiếp theo. "
        f"Đã hỏi {n} câu. Nếu học sinh vừa trả lời câu 5, chỉ nhận xét và nhắc bấm nút “Kết thúc và chấm điểm”. "
        "Không cho điểm số trong lúc hỏi."
    )


def page_oral():
    head("Vấn đáp lí thuyết", "Giám khảo AI hỏi từng câu về khái niệm hoặc định lí. Cuối buổi AI chấm điểm và chỉ ra lỗ hổng kiến thức.")
    with st.container(border=True):
        a, b = st.columns([2, 1], gap="large")
        with a:
            topic = topic_pick("o")
        with b:
            kind = st.radio("Dạng vấn đáp", ["Khái niệm", "Định lí, công thức"], key="o_kind")
        if st.button("Bắt đầu vấn đáp mới", icon=":material/record_voice_over:", type="primary", key="b_o_start"):
            O = {"topic": topic, "kind": kind, "msgs": [{"role": "user", "content": "Hãy bắt đầu vấn đáp."}], "res": None}
            with st.spinner("Giám khảo đang chuẩn bị câu hỏi..."):
                r = ask(oral_sys(O), msgs=O["msgs"], max_tokens=700)
            S.oral = {**O, "msgs": O["msgs"] + [{"role": "assistant", "content": r}]} if r else None
    O = S.get("oral")
    if not O:
        st.info("Chọn chủ đề và dạng vấn đáp, rồi bấm “Bắt đầu vấn đáp mới”.")
        return
    st.caption(f"Chủ đề: {O['topic']} · {O['kind']}")
    for m in O["msgs"][1:]:
        st.chat_message(m["role"]).markdown(m["content"])
    done = bool(O.get("res"))
    p = st.chat_input("Nhập câu trả lời của bạn...", disabled=done)
    if p:
        O["msgs"].append({"role": "user", "content": p})
        st.chat_message("user").markdown(p)
        with st.chat_message("assistant"):
            with st.spinner("Giám khảo đang nhận xét..."):
                r = ask(oral_sys(O), msgs=O["msgs"], max_tokens=700)
            if r:
                st.markdown(r)
                O["msgs"].append({"role": "assistant", "content": r})
            else:
                O["msgs"].pop()
    answered = sum(m["role"] == "user" for m in O["msgs"]) - 1
    if not done and answered >= 1 and st.button("Kết thúc và chấm điểm", icon=":material/fact_check:",
                                                  type="primary", key="b_o_end"):
        tr = "\n".join(f"{'Giám khảo' if m['role'] == 'assistant' else 'Học sinh'}: {m['content']}" for m in O["msgs"][1:])
        with st.spinner("Đang chấm phần vấn đáp..."):
            res = ask_json(SYS_ORAL_GRADE, f"Chủ đề: {O['topic']}\nDạng: {O['kind']}\nBản ghi:\n{tr}", max_tokens=3000)
        if res:
            O["res"] = res
            res.setdefault("chu_de", O["topic"])
            log_grade("oral", res)
            st.rerun()
    if done:
        res = O["res"]
        with st.container(border=True):
            st.markdown("#### Kết quả vấn đáp")
            show_score(num(res.get("diem")))
            st.markdown(str(res.get("nhan_xet", "")))
            for c in res.get("tung_cau") or []:
                if isinstance(c, dict):
                    st.markdown(f"- **{c.get('cau', '')}** ({num(c.get('diem')):g}/10): {c.get('nhan_xet', '')}")
            show_issues(res)
            if res.get("goi_y_on_tap"):
                st.markdown("**Gợi ý ôn tập**")
                for x in res["goi_y_on_tap"]:
                    st.markdown(f"- {x}")


# ───────────────────────── TRANG: BẢN ĐỒ TƯ DUY ─────────────────────────
def to_dot(tree):
    pal = ["#0F5C5A", "#B4531F", "#4F6D3A", "#8A6A12", "#2F5D8C", "#8A3B4E"]
    cnt = [0]
    out = ['digraph G{rankdir=LR;bgcolor="transparent";ranksep=0.5;nodesep=0.18;'
           'node[shape=box,style="rounded,filled",fontname="Helvetica",fontsize=13,margin="0.16,0.08",penwidth=0];'
           'edge[penwidth=1.5,arrowhead=none];']

    def lab(s):
        return "\\n".join(textwrap.wrap(re.sub(r'["\\]', "", str(s)), 22)) or "?"

    def walk(n, parent, color, depth):
        if cnt[0] >= 70 or not isinstance(n, dict):
            return
        cnt[0] += 1
        me = f"n{cnt[0]}"
        if depth == 0:
            out.append(f'{me}[label="{lab(n.get("ten"))}",fillcolor="#1F2A27",fontcolor="white",fontsize=16];')
        elif depth == 1:
            out.append(f'{me}[label="{lab(n.get("ten"))}",fillcolor="{color}",fontcolor="white"];')
        else:
            out.append(f'{me}[label="{lab(n.get("ten"))}",fillcolor="{color}26",fontcolor="#1F2A27"];')
        if parent:
            out.append(f'{parent}->{me}[color="{color}"];')
        if depth < 3:
            for j, ch in enumerate(n.get("con") or []):
                walk(ch, me, pal[j % len(pal)] if depth == 0 else color, depth + 1)

    walk(tree, None, pal[0], 0)
    return "".join(out) + "}"


def outline(n, d=0):
    if not isinstance(n, dict):
        return ""
    s = "  " * d + f"- {n.get('ten', '')}\n"
    return s + "".join(outline(c, d + 1) for c in (n.get("con") or []) if d < 3)


def page_map():
    head("Bản đồ tư duy", "AI hệ thống hoá một chủ đề thành sơ đồ: khái niệm, công thức, dạng bài, sai lầm thường gặp.")
    with st.container(border=True):
        topic = topic_pick("m")
        if st.button("Tạo bản đồ tư duy", icon=":material/account_tree:", type="primary", key="b_m"):
            with st.spinner("Đang vẽ bản đồ..."):
                t = ask_json(SYS_MAP, f"Chủ đề: {topic}", max_tokens=2500)
            if isinstance(t, dict) and t.get("ten"):
                S.r_map = t
            elif t is not None:
                st.error("Không tạo được bản đồ. Hãy thử lại.")
    t = S.get("r_map")
    if t:
        st.graphviz_chart(to_dot(t))
        with st.expander("Dạng danh sách (dễ đọc trên điện thoại)"):
            st.markdown(outline(t))


# ───────────────────────── ĐĂNG NHẬP & KHUNG CHÍNH ─────────────────────────
def login():
    l, r = st.columns([1.15, 1], gap="large")
    with l:
        st.markdown(
            '<div class="eyebrow">Xác suất thống kê · Trợ lí học tập</div>'
            '<h1>Học từng bước,<br>hiểu đến cùng.</h1>'
            '<p class="lead">Chấm bài bằng ảnh chụp, luyện tập, vấn đáp và bản đồ tư duy. '
            'Hồ sơ cá nhân ghi lại các lỗi sai của bạn để đề xuất cách học cải thiện.</p>',
            unsafe_allow_html=True)
    with r, st.container(border=True):
        with st.form("login"):
            name = st.text_input("Tên của bạn", max_chars=40)
            go = st.form_submit_button("Vào học", type="primary")
        if go:
            if name.strip():
                S.user, S.profile = name.strip(), load_profile(name.strip())  # shortcut: chỉ dùng tên, chưa có mật khẩu
                st.rerun()
            st.warning("Hãy nhập tên của bạn.")
        with st.expander("Khôi phục hồ sơ đã tải về"):
            up = st.file_uploader("Tệp .json", type=["json"], key="restore", label_visibility="collapsed")
            if up:
                try:
                    p = json.load(up)
                    assert isinstance(p.get("events"), list) and str(p.get("name", "")).strip()
                except (ValueError, AssertionError, AttributeError):
                    st.error("Tệp hồ sơ không hợp lệ.")
                else:
                    p["name"] = str(p["name"]).strip()[:40]
                    save_profile(p)
                    S.user, S.profile = p["name"], p
                    st.rerun()


PAGES = {"Chấm bài": page_tutor, "Luyện tập": page_practice, "Vấn đáp": page_oral, "Sơ đồ tư duy": page_map}


def main():
    st.set_page_config(page_title="XSTK Tutor", page_icon=":material/functions:", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    if not S.get("user"):
        login()
        return
    a, b, c = st.columns([5, 1.5, 1.5], vertical_alignment="center")
    a.markdown('<div class="eyebrow">Xác suất thống kê · Trợ lí học tập</div><h1>Học từng bước,<br>hiểu đến cùng.</h1>',
               unsafe_allow_html=True)
    if b.button("Hồ sơ học tập", icon=":material/person:", type="primary", key="b_prof"):
        profile_dialog()
    with c.popover(S.user, icon=":material/account_circle:"):
        st.download_button("Tải hồ sơ (.json)", json.dumps(S.profile, ensure_ascii=False),
                           file_name="ho_so_xstk.json", mime="application/json")
        if st.button("Đổi người dùng", key="b_out"):
            for k in list(S.keys()):
                del S[k]
            st.rerun()
    if not _has_key():
        st.warning("Chưa cấu hình API key: các tính năng AI tạm thời chưa hoạt động.")
    nav = st.segmented_control("Chức năng", list(PAGES), default="Chấm bài", key="nav", label_visibility="collapsed")
    if nav:
        S.last_nav = nav
    PAGES[S.get("last_nav", "Chấm bài")]()


main()
