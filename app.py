"""
XSTK Tutor - trợ giảng AI học Xác suất thống kê theo 3 quy trình 4 bước (Streamlit).
Chạy thử : streamlit run app.py
Triển khai: Streamlit Community Cloud, Main file path = app.py

API key (GEMINI_API_KEY) được tìm theo thứ tự:
  1. Biến môi trường GEMINI_API_KEY (GitHub Secrets: Codespaces / Actions / máy chủ tự dựng đưa vào dưới dạng biến môi trường)
  2. Streamlit Community Cloud > App settings > Secrets
"""
import copy
import hashlib
import io
import json
import os
import random
import re
import threading
import time
import datetime as dt
from collections import Counter, OrderedDict, deque
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import errors, types
from PIL import Image, ImageOps

# ───────────────────────── CẤU HÌNH ─────────────────────────
MODEL = "gemini-3.5-flash-lite"
KEY_NAME = "GEMINI_API_KEY"

# ── Hạn mức gói free của AI Studio cho Gemini 3.5 Flash Lite: RPM 15 · TPM 250k · RPD 500.
# Ngưỡng dưới đây đặt thấp hơn hạn mức thật để chừa biên an toàn (đồng hồ Google lệch, app khác dùng chung key).
RPM_CAP = 12            # số lượt gọi tối đa trong 60 giây, toàn hệ thống
TPM_CAP = 200_000       # token (đầu vào + đầu ra ước lượng) tối đa trong 60 giây
RPD_CAP = 450           # số lượt gọi tối đa mỗi ngày (reset 0h giờ Thái Bình Dương, khoảng 14-15h giờ Việt Nam)
USER_DAILY = 30         # lượt AI tối đa mỗi người mỗi ngày
IP_DAILY = 90           # lượt AI tối đa mỗi địa chỉ mạng mỗi ngày (cả lớp có thể dùng chung wifi)
QUEUE_MAX = 40          # giây tối đa một yêu cầu chịu xếp hàng trước khi báo bận
CACHE_TTL = 24 * 3600   # bộ nhớ chung giữ kết quả 24 giờ
POOL_SIZE = 3           # mỗi (loại, quy trình, chủ đề, mức độ) chỉ sinh tối đa 3 bộ đề, người sau dùng lại
HEAD = {"minimal": 0, "medium": 6000, "high": 12000}  # chỗ chừa cho token "suy nghĩ" (tính vào max_output_tokens)
PROFILE_DIR = Path("profiles")  # shortcut: đĩa của Streamlit Cloud bị xoá khi app khởi động lại, nên có nút tải/khôi phục hồ sơ

CONCEPTS = [
    "Phép thử, không gian mẫu và biến cố",
    "Xác suất của biến cố (cổ điển, thống kê)",
    "Biến cố xung khắc, biến cố đối",
    "Xác suất có điều kiện",
    "Biến cố độc lập",
    "Biến ngẫu nhiên rời rạc, bảng phân phối",
    "Kì vọng (giá trị trung bình)",
    "Phương sai và độ lệch chuẩn",
    "Biến ngẫu nhiên liên tục, hàm mật độ",
    "Phân phối chuẩn",
    "Tổng thể, mẫu và số đặc trưng mẫu",
    "Khoảng tin cậy",
    "Giả thuyết thống kê, mức ý nghĩa, p-giá trị",
    "Hệ số tương quan",
]
THEOREMS = [
    "Quy tắc cộng xác suất",
    "Quy tắc nhân xác suất",
    "Công thức xác suất toàn phần",
    "Công thức Bayes",
    "Công thức Bernoulli (phân phối nhị thức)",
    "Tính chất của kì vọng và phương sai",
    "Phân phối Poisson và xấp xỉ nhị thức",
    "Chuẩn hoá phân phối chuẩn, quy tắc 3 sigma",
    "Luật số lớn",
    "Định lí giới hạn trung tâm",
    "Khoảng tin cậy cho trung bình và tỉ lệ",
    "Kiểm định giả thuyết về trung bình",
    "Phương trình hồi quy tuyến tính (bình phương tối thiểu)",
]
ERR = [
    "Hiểu sai khái niệm", "Áp dụng sai công thức", "Sai tính toán",
    "Thiếu hoặc sai lập luận", "Đọc sai đề hoặc dữ kiện",
    "Trình bày thiếu chặt chẽ", "Khác",
]

# Ba quy trình, mỗi quy trình 4 bước: (tên bước, mục tiêu, việc học sinh cần làm)
WF = {
    "concept": {
        "name": "Học khái niệm", "icon": ":material/lightbulb:", "kind": "Khái niệm", "topics": CONCEPTS,
        "pick": "Khái niệm cần học",
        "desc": "Từ tình huống quen thuộc đến định nghĩa chính xác, củng cố bằng vấn đáp rồi vận dụng vào bài tập.",
        "steps": [
            ("Tiếp cận", "Gặp khái niệm qua tình huống thực tế, tự nhận ra dấu hiệu chung.",
             "Đọc tình huống, trả lời các câu hỏi dự đoán trước khi xem định nghĩa."),
            ("Hình thành", "Nắm định nghĩa, dấu hiệu bản chất, kí hiệu và phân biệt với khái niệm dễ nhầm.",
             "Đọc thẻ khái niệm, sau đó tự phát biểu lại bằng lời của bạn để AI nhận xét."),
            ("Củng cố", "Vấn đáp 5 câu: định nghĩa, phân biệt, ví dụ và phản ví dụ, liên hệ, vận dụng.",
             "Trả lời từng câu của giám khảo AI, cuối buổi nhận điểm và lỗ hổng kiến thức."),
            ("Vận dụng", "Luyện bài tập trắc nghiệm, tự luận và hệ thống hoá bằng sơ đồ tư duy.",
             "Tạo bài tập theo mức độ, tự làm rồi chấm; vẽ sơ đồ tư duy để tổng kết."),
        ],
    },
    "theorem": {
        "name": "Học định lí", "icon": ":material/function:", "kind": "Định lí, công thức", "topics": THEOREMS,
        "pick": "Định lí, công thức cần học",
        "desc": "Tính thử, dự đoán, phát biểu chính xác, hiểu chứng minh rồi củng cố và vận dụng.",
        "steps": [
            ("Tiếp cận, phát biểu", "Tính thử vài trường hợp cụ thể, dự đoán quy luật rồi phát biểu chính xác.",
             "Làm phần tính thử, đọc phát biểu, chỉ ra giả thiết, kết luận và tự phát biểu lại."),
            ("Chứng minh", "Hiểu ý tưởng và các luận cứ của chứng minh.",
             "Đi theo các câu hỏi gợi mở để tự chứng minh, chỉ xem chứng minh đầy đủ khi cần."),
            ("Củng cố", "Vấn đáp 5 câu: phát biểu, điều kiện, ý tưởng chứng minh, cấu trúc logic, vận dụng.",
             "Trả lời từng câu của giám khảo AI, cuối buổi nhận điểm và lỗ hổng kiến thức."),
            ("Vận dụng", "Áp dụng định lí vào bài tập và hệ thống hoá bằng sơ đồ tư duy.",
             "Tạo bài tập theo mức độ, tự làm rồi chấm; vẽ sơ đồ tư duy để tổng kết."),
        ],
    },
    "solve": {
        "name": "Giải toán", "icon": ":material/calculate:", "kind": "Giải toán", "topics": [],
        "desc": "Quy trình 4 bước của Pólya: tìm hiểu đề, lập chương trình giải, trình bày, kiểm tra và mở rộng.",
        "steps": [
            ("Tìm hiểu đề", "Phân biệt cái đã cho, cái phải tìm, kí hiệu hoá các đại lượng.",
             "Tự đọc kĩ đề, gạch chân dữ kiện; dùng AI để đối chiếu phần phân tích của bạn."),
            ("Lập chương trình giải", "Quy về bài toán quen thuộc, chọn kiến thức và chia nhỏ bài toán.",
             "Xem gợi ý chiến lược (không có đáp án), tự lập dàn ý lời giải."),
            ("Trình bày lời giải", "Viết lời giải chặt chẽ rồi để AI chấm điểm, chỉ ra chỗ sai.",
             "Gõ hoặc chụp ảnh bài làm, bấm Chấm bài; sửa theo nhận xét rồi chấm lại."),
            ("Kiểm tra, mở rộng", "Kiểm tra kết quả, rút ra phương pháp, luyện thêm bài tương tự.",
             "Đối chiếu bài giải chi tiết, ghi lại phương pháp, tạo bài tương tự để luyện tiếp."),
        ],
    },
}
WF_LOG = {"concept": "Học khái niệm", "theorem": "Học định lí", "solve": "Giải toán",
          "tutor": "Giải toán", "practice": "Luyện tập", "oral": "Vấn đáp"}  # tên cũ để đọc hồ sơ đã lưu

# ───────────────────────── PROMPT ─────────────────────────
BASE = (
    "Bạn là trợ giảng Xác suất thống kê (XSTK) cho học sinh, sinh viên Việt Nam. "
    "Luôn trả lời bằng tiếng Việt, ngắn gọn, chính xác, thân thiện, không dùng emoji. "
    "Công thức toán viết bằng LaTeX: $...$ trong dòng, $$...$$ cho dòng riêng. "
)
JSON_RULE = (
    r' Chỉ trả về MỘT đối tượng JSON hợp lệ, không có chữ nào bên ngoài, không dùng khối mã. '
    r'Trong chuỗi JSON, mọi dấu gạch chéo ngược của LaTeX phải viết đôi (ví dụ "\\frac{1}{2}"). '
    r'Xuống dòng trong chuỗi dùng \n.'
)
ERR_RULE = 'Trường "loai" của loi_sai phải là một trong: ' + "; ".join(ERR) + "."

# Học khái niệm
SYS_C1 = BASE + (
    "Bước 1 của quy trình học khái niệm: TIẾP CẬN. Giúp học sinh tự đi đến khái niệm, CHƯA nêu định nghĩa chính thức. "
    "Trình bày dưới 320 từ, gồm các mục in đậm: **Tình huống** (một tình huống thực tế hoặc trò chơi gần gũi, có số liệu nhỏ, "
    "làm nảy sinh nhu cầu cần khái niệm); **Ví dụ dẫn dắt** (2-3 ví dụ cụ thể có chung dấu hiệu bản chất); "
    "**Câu hỏi dự đoán** (2-3 câu để học sinh tự khái quát dấu hiệu chung); **Vì sao cần khái niệm này** (1-2 câu)."
)
SYS_C2 = BASE + (
    "Bước 2 của quy trình học khái niệm: HÌNH THÀNH. Viết thẻ khái niệm dưới 450 từ, gồm các mục in đậm: "
    "**Định nghĩa** (chính xác, đúng chương trình); **Dấu hiệu bản chất** (gạch đầu dòng); **Kí hiệu và thuật ngữ**; "
    "**Ví dụ** (2 ví dụ, giải thích ngắn vì sao thoả định nghĩa); **Phản ví dụ** (2 trường hợp, chỉ rõ thiếu dấu hiệu nào); "
    "**Dễ nhầm với** (khái niệm gần giống và điểm khác biệt); **Liên hệ** (khái niệm, định lí liên quan)."
)
# Học định lí
SYS_T1 = BASE + (
    "Bước tiếp cận của quy trình học định lí, công thức. CHƯA phát biểu định lí. Dưới 320 từ, các mục in đậm: "
    "**Tình huống** (bài toán thực tế làm nảy sinh nhu cầu); **Tính thử** (2-3 trường hợp cụ thể với số liệu nhỏ, "
    "để học sinh tự tính, KHÔNG ghi kết quả); **Dự đoán** (câu hỏi gợi ý học sinh phát hiện quy luật chung); "
    "**Cách kiểm tra dự đoán** (1-2 câu)."
)
SYS_T2 = BASE + (
    "Bước phát biểu của quy trình học định lí, công thức. Dưới 420 từ, các mục in đậm: **Phát biểu** (chính xác, có công thức); "
    "**Giả thiết**; **Kết luận**; **Điều kiện áp dụng** (khi nào KHÔNG được dùng); **Ý nghĩa các đại lượng**; "
    "**Trường hợp đặc biệt, hệ quả**; **Mệnh đề đảo** (đúng hay sai, vì sao); **Sai lầm hay gặp khi áp dụng**."
)
SYS_T3_HINT = BASE + (
    "Bước chứng minh của quy trình học định lí, dạy theo kiểu gợi mở. Nêu **Ý tưởng chính** (1-2 câu), "
    "**Luận cứ cần dùng** (định nghĩa, tính chất, định lí đã biết), rồi **Các bước** (3-5 bước nhỏ, mỗi bước là một câu hỏi "
    "gợi ý để học sinh tự làm). TUYỆT ĐỐI không viết chứng minh hoàn chỉnh. Nếu chứng minh chặt chẽ vượt chương trình, "
    "hãy nói rõ và hướng dẫn một lập luận trực quan hoặc kiểm chứng bằng ví dụ."
)
SYS_T3_FULL = BASE + (
    "Viết chứng minh đầy đủ, chặt chẽ cho định lí hoặc công thức XSTK được hỏi, đánh số từng bước và ghi rõ luận cứ của mỗi bước. "
    "Nếu chứng minh chặt chẽ vượt chương trình, trình bày chứng minh cho trường hợp riêng hoặc lập luận trực quan và nói rõ điều đó. "
    "Cuối cùng nêu **Điều rút ra** về phương pháp chứng minh."
)
SYS_SELF = BASE + (
    "Học sinh tự phát biểu lại một định nghĩa hoặc định lí XSTK bằng lời của mình. Đánh giá độ chính xác, độ đầy đủ "
    "(có thiếu dấu hiệu bản chất, giả thiết hay điều kiện nào không), cách dùng thuật ngữ. Ghi nhận phần đúng. "
    "Không chép lại nguyên văn định nghĩa chuẩn, chỉ chỉ ra phần cần sửa. " + ERR_RULE + JSON_RULE +
    ' Cấu trúc: {"diem":số 0-10,"nhan_xet":"2-4 câu","loi_sai":[{"loai":"...","mo_ta":"..."}],'
    '"lo_hong":["..."],"chu_de":"tên chủ đề ngắn"}'
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
    "Chỉ ra lỗ hổng kiến thức và sai lầm cụ thể của học sinh. " + ERR_RULE + JSON_RULE +
    ' Cấu trúc: {"diem":số 0-10,"nhan_xet":"2-4 câu","tung_cau":[{"cau":"tóm tắt câu hỏi","diem":số,"nhan_xet":"..."}],'
    '"loi_sai":[{"loai":"...","mo_ta":"..."}],"lo_hong":["..."],"goi_y_on_tap":["..."],"chu_de":"tên chủ đề"}'
)
FOCUS = {
    "concept": "Bài tập phục vụ củng cố và vận dụng KHÁI NIỆM: nhận dạng qua ví dụ và phản ví dụ, thể hiện khái niệm, "
               "phân biệt với khái niệm dễ nhầm, vận dụng vào tình huống thực tế. ",
    "theorem": "Bài tập phục vụ vận dụng ĐỊNH LÍ, CÔNG THỨC: kiểm tra điều kiện áp dụng, áp dụng trực tiếp, "
               "kết hợp với kiến thức khác, tình huống thực tế. ",
}
_PRACTICE = "Soạn bài tập XSTK bám sát chương trình, số liệu cụ thể, đủ dữ kiện, giải được, có lời giải đúng. "
SYS_MC = BASE + _PRACTICE + (
    "Soạn đúng 5 câu trắc nghiệm, 4 phương án, phương án nhiễu phản ánh sai lầm phổ biến, vị trí đáp án đúng phân bố ngẫu nhiên "
    "giữa A-D, 'sai_lam' chỉ gồm các phương án sai. Mọi công thức, phân số, kí hiệu trong đề và trong từng phương án "
    "PHẢI đặt giữa hai dấu $ (ví dụ \"$\\\\frac{7}{15}$\"). 'chien_luoc' theo bước 1-2 của quy trình giải, TUYỆT ĐỐI không tiết lộ "
    "đáp án hay kết quả số. Kiểm tra kĩ đáp án trước khi trả lời." + JSON_RULE +
    ' Cấu trúc: {"cau":[{"de":"...","lua_chon":{"A":"..","B":"..","C":"..","D":".."},"dap_an":"B",'
    '"chien_luoc":"...","loi_giai":"lời giải chi tiết nêu rõ đáp án","sai_lam":{"A":{"loai":"(một trong: '
    + "; ".join(ERR) + ')","mo_ta":"vì sao học sinh hay chọn nhầm"}}}]}'
)
SYS_ES = BASE + _PRACTICE + (
    "Soạn đúng 5 câu tự luận: 2 câu cơ bản, 2 câu trung bình, 1 câu gắn thực tiễn (độ khó tổng thể theo mức độ được yêu cầu). "
    "'tieu_chi' là thang điểm 10 tham khảo. 'chien_luoc' theo bước 1-2 của quy trình giải, TUYỆT ĐỐI không tiết lộ đáp án "
    "hay kết quả số. Kiểm tra kĩ lời giải." + JSON_RULE +
    ' Cấu trúc: {"cau":[{"de":"...","chien_luoc":"...","loi_giai":"lời giải chi tiết 4 bước","tieu_chi":"..."}]}'
)
SYS_MAP = BASE + (
    "Tạo bản đồ tư duy cho một chủ đề XSTK. Gốc là tên chủ đề. 5-6 nhánh cấp 1 (ví dụ: Khái niệm; Công thức và tính chất; "
    "Dạng bài và phương pháp; Sai lầm thường gặp; Ví dụ và ứng dụng; Liên hệ chủ đề khác), mỗi nhánh 2-4 nút con, "
    "có thể thêm cấp 3. Tên nút tối đa 8 từ, văn bản thuần, không LaTeX, không dấu nháy." + JSON_RULE +
    ' Cấu trúc: {"ten":"gốc","con":[{"ten":"...","con":[{"ten":"..."}]}]}'
)
# Giải toán
SOLVE_FLOW = (
    "Quy trình giải toán 4 bước: (1) Tìm hiểu đề: phân biệt cái đã cho, cái phải tìm, kí hiệu hoá; "
    "(2) Xây dựng chương trình giải: quy về bài toán đơn giản quen thuộc, chọn kiến thức gần nhất; "
    "(3) Trình bày lời giải; (4) Kiểm tra, nhìn lại, rút phương pháp, mở rộng. "
)
SYS_S1 = BASE + SOLVE_FLOW + (
    "Chỉ làm BƯỚC 1 - Tìm hiểu đề. Các mục in đậm: **Tóm tắt đề** (1-2 câu); **Cái đã cho** (kí hiệu hoá: biến cố, biến ngẫu nhiên, "
    "tham số); **Cái phải tìm** (viết dưới dạng kí hiệu, ví dụ $P(A|B)$); **Điều kiện, ràng buộc ngầm**; **Dạng bài**; "
    "**Gợi ý biểu diễn** (sơ đồ cây, bảng, trục số... nếu phù hợp). TUYỆT ĐỐI không giải, không tính kết quả."
)
SYS_HINT = BASE + SOLVE_FLOW + (
    "Chỉ làm BƯỚC 2 - Xây dựng chương trình giải. Nêu: bài toán quen thuộc có thể quy về, kiến thức/công thức cần dùng "
    "và vì sao chọn nó, cách chia bài toán thành các bài toán nhỏ theo thứ tự, điểm dễ sai. "
    "TUYỆT ĐỐI không đưa đáp án, không tính ra kết quả số, không thay số hoàn chỉnh vào công thức, "
    "không viết lời giải từng bước. Kết thúc bằng 1-2 câu hỏi gợi mở để học sinh tự làm."
)
SYS_GRADE = BASE + (
    "Bạn chấm bài làm XSTK của học sinh (có thể là ảnh chụp chữ viết tay hoặc PDF: hãy đọc/OCR cẩn thận). "
    "Tự giải đề trước để biết kết quả đúng, nhưng CHỈ chấm điểm và chỉ ra chỗ sai: KHÔNG viết lại lời giải đúng, "
    "KHÔNG nêu kết quả cuối cùng đúng. Với mỗi lỗi, nói sai ở bước nào, vì sao, và gợi hướng sửa bằng một câu hỏi "
    "hoặc nhắc kiến thức. Chấm thang 10, ghi nhận phần đúng. Nếu không đọc được bài làm thì diem = 0 và nói rõ. "
    + ERR_RULE + JSON_RULE +
    ' Cấu trúc: {"van_ban_chep_lai":"nội dung bài làm đọc được", "diem":số, "nhan_xet":"2-4 câu", '
    '"buoc":[{"ten":"...","dung":true,"ghi_chu":"..."}], "loi_sai":[{"loai":"...","mo_ta":"..."}], '
    '"lo_hong":["kiến thức còn thiếu"], "chu_de":"tên chủ đề ngắn"}'
)
SYS_SOL = BASE + SOLVE_FLOW + (
    "Viết BÀI GIẢI CHI TIẾT theo 4 bước, rõ ràng từng phép biến đổi, in đậm kết quả cuối. Sau đó thêm mục **Nhìn lại**: "
    "cách tự kiểm tra kết quả (ước lượng, trường hợp đặc biệt, điều kiện $0 \\le P \\le 1$...), **Phương pháp rút ra** cho dạng bài, "
    "**Mở rộng** (một hướng thay đổi đề), và 1-2 **Sai lầm thường gặp**."
)
SYS_SIMILAR = BASE + SOLVE_FLOW + (
    "Từ đề bài gốc, soạn đúng 3 bài tập tương tự để học sinh luyện thêm: bài 1 cùng dạng, đổi số liệu và bối cảnh; "
    "bài 2 cùng phương pháp nhưng đổi cái phải tìm; bài 3 nâng cao hoặc kết hợp thêm một kiến thức. "
    "Mỗi bài đủ dữ kiện, giải được, kèm chiến lược (không lộ đáp án) và lời giải đúng." + JSON_RULE +
    ' Cấu trúc: {"chu_de":"tên dạng bài","bai":[{"muc_do":"Tương tự","de":"...","chien_luoc":"...","loi_giai":"..."}]}'
)
CASIO_GUIDE = r"""
Tên các mục trên máy hiển thị bằng tiếng Anh. Số thứ tự mục có thể khác đôi chút giữa các đời máy: hãy chọn theo **tên** hiện trên màn hình.

**Chuẩn bị**
- Khôi phục cài đặt gốc khi máy chạy lạ: `SHIFT` `9` (RESET) → `3` (Initialize All) → `=` → `AC`.
- Đổi kết quả giữa phân số và số thập phân: phím `S⇔D`.
- Lưu kết quả vào biến A: `SHIFT` `RCL` (STO) → phím `(−)` (biến A). Gọi lại: `ALPHA` `(−)`.

**Giai thừa, tổ hợp, chỉnh hợp**
- $n!$: nhập $n$ → `SHIFT` `x⁻¹` (x!) → `=`.
- $C_n^k$: nhập $n$ → `SHIFT` `÷` (nCr) → nhập $k$ → `=`. Ví dụ $C_{10}^{2}$: `10` `SHIFT` `÷` `2` `=` → 45.
- $A_n^k$: nhập $n$ → `SHIFT` `×` (nPr) → nhập $k$ → `=`.

**Thống kê một biến** (trung bình, phương sai, độ lệch chuẩn)
1. `MENU` → **Statistics** → **1-Variable**.
2. Nếu dữ liệu có tần số: `SHIFT` `MENU` (SETUP) → `▼` → **Statistics** → **Frequency? On**, rồi vào lại bước 1.
3. Nhập từng giá trị cột x (và cột Freq), mỗi số bấm `=`. Xong bấm `AC`.
4. `OPTN` → **1-Variable Calc**: đọc $\bar{x}$, $\sigma_x$ (chia cho $n$), $s_x$ (độ lệch chuẩn mẫu hiệu chỉnh, chia cho $n-1$), $n$.

**Hồi quy tuyến tính, hệ số tương quan**
1. `MENU` → **Statistics** → **y=a+bx**, nhập cặp số (x, y), xong bấm `AC`.
2. `OPTN` → **Regression Calc**: đọc $a$, $b$ (đường $y=a+bx$) và hệ số tương quan $r$.

**Phân phối xác suất** (`MENU` → **Distribution**, chọn **Variable** để tính một giá trị)
- **Binomial PD**: $P(X = x)$ với $X \sim B(N, p)$; nhập x, N, p.
- **Binomial CD**: $P(X \le x)$. Muốn $P(X \ge k)$ thì lấy $1 - P(X \le k-1)$.
- **Poisson PD / Poisson CD**: $P(X = x)$ và $P(X \le x)$; nhập x, λ.
- **Normal CD**: $P(a \le X \le b)$ với $X \sim N(\mu, \sigma^2)$; nhập Lower $= a$, Upper $= b$, σ, μ.
  Tính $P(X \le b)$ thì nhập Lower $= -10^{99}$ (`(−)` `1` `×10ˣ` `99`); tính $P(X \ge a)$ thì nhập Upper $= 10^{99}$.
- **Inverse Normal**: biết diện tích bên trái (Area), σ, μ thì tìm x. Tra $z_{\alpha/2}$: Area $= 1 - \alpha/2$, σ $= 1$, μ $= 0$
  (ví dụ $\alpha = 0{,}05$ cho $z \approx 1{,}96$).

**Mẹo kiểm tra**: xác suất luôn nằm trong $[0, 1]$; tổng xác suất trong bảng phân phối bằng 1; với phân phối chuẩn,
$P(\mu - \sigma \le X \le \mu + \sigma) \approx 0{,}6827$.
"""
SYS_CASIO = BASE + (
    "Hướng dẫn học sinh dùng máy tính Casio fx-580VN X để tính các đại lượng trong bài toán XSTK được cho. "
    "Chỉ dùng các thao tác có trong bảng tra dưới đây, ghi phím trong dấu `...`, đánh số từng bước, nêu màn hình kết quả "
    "nên hiện gì (ví dụ đại lượng nào), KHÔNG nêu đáp số cuối cùng của bài. Nếu một phần bài không cần máy tính, nói rõ. "
    "Kết thúc bằng một cách kiểm tra nhanh kết quả trên máy. Bảng tra thao tác:\n" + CASIO_GUIDE
)
SYS_PLAN = BASE + (
    "Bạn nhận thống kê hồ sơ học XSTK của một học sinh (JSON). Chỉ dựa vào số liệu đó, không bịa. Trả lời dưới 350 từ, gồm: "
    "**Nhận định** (2-4 gạch đầu dòng nêu quy luật lỗi sai xuyên suốt); **Ưu tiên ôn** (tối đa 3 chủ đề, vì sao); "
    "**Kế hoạch 7 ngày** (mỗi ngày một việc cụ thể, dùng 3 quy trình của ứng dụng: Học khái niệm (tiếp cận, hình thành, "
    "vấn đáp, luyện tập, sơ đồ tư duy), Học định lí (tính thử, phát biểu, chứng minh, vấn đáp, luyện tập), Giải toán "
    "(phân tích đề, chiến lược, chấm bài, bài tập tương tự)); **Cách tự kiểm tra tiến bộ**. "
    "Nếu dữ liệu còn ít, nói rõ và đề xuất làm thêm."
)

S = st.session_state


# ───────────────────────── BỘ ĐIỀU TIẾT HẠN MỨC (DÙNG CHUNG MỌI NGƯỜI) ─────────────────────────
def _day():
    try:
        tz = ZoneInfo("America/Los_Angeles")
    except Exception:
        tz = dt.timezone(dt.timedelta(hours=-8))
    return dt.datetime.now(tz).date().isoformat()


class Shared:
    """Một đối tượng duy nhất cho cả tiến trình (st.cache_resource): đếm lượt gọi, hàng đợi, bộ nhớ đệm, kho đề.
    shortcut: nằm trong RAM, mất khi app khởi động lại (khi đó bộ đếm về 0, nên ngưỡng mới chừa biên an toàn)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls, self.toks = deque(), deque()
        self.day, self.rpd, self.block_until, self.used = _day(), 0, 0.0, {}
        self.cache, self.pool, self.flight = OrderedDict(), OrderedDict(), {}

    def _roll(self, now):
        d = _day()
        if d != self.day:
            self.day, self.rpd, self.used = d, 0, {}
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        while self.toks and now - self.toks[0][0] > 60:
            self.toks.popleft()

    def reserve(self, who, est):
        """-> ("ok", mục_token) | ("wait", giây) | ("day", 0) | ("user", loại). Chỉ tính lượt khi trả về ok."""
        now = time.time()
        with self.lock:
            self._roll(now)
            if self.rpd >= RPD_CAP:
                return "day", 0
            for k, cap in who:
                if self.used.get(k, 0) >= cap:
                    return "user", k[0]
            if now < self.block_until:
                return "wait", self.block_until - now
            if len(self.calls) >= RPM_CAP:
                return "wait", self.calls[0] + 60 - now
            if self.toks and sum(t for _, t in self.toks) + est > TPM_CAP:
                return "wait", self.toks[0][0] + 60 - now
            self.calls.append(now)
            entry = [now, est]
            self.toks.append(entry)
            self.rpd += 1
            for k, _ in who:
                self.used[k] = self.used.get(k, 0) + 1
            return "ok", entry

    def settle(self, entry, actual):
        with self.lock:
            entry[1] = actual  # thay ước lượng bằng số token thật Google báo về

    def penalize(self, seconds, day=False):
        with self.lock:
            self.block_until = time.time() + seconds
            if day:
                self.rpd = RPD_CAP

    def used_of(self, k):
        with self.lock:
            self._roll(time.time())
            return self.used.get(k, 0)

    def cget(self, k):
        with self.lock:
            v = self.cache.get(k)
            return copy.deepcopy(v[1]) if v and time.time() - v[0] < CACHE_TTL else None

    def cput(self, k, val):
        with self.lock:
            self.cache[k] = (time.time(), copy.deepcopy(val))
            self.cache.move_to_end(k)
            while len(self.cache) > 400:
                self.cache.popitem(last=False)

    def flight_lock(self, k):
        with self.lock:
            return self.flight.setdefault(k, threading.Lock())

    def pool_get(self, k):
        with self.lock:
            return list(self.pool.get(k, []))

    def pool_add(self, k, item):
        with self.lock:
            self.pool.setdefault(k, []).append(item)
            while len(self.pool) > 200:
                self.pool.popitem(last=False)


@st.cache_resource
def shared():
    return Shared()


def _ip():
    try:
        return st.context.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    except Exception:
        return ""


def est_tokens(system, content, msgs):
    n = len(system) / 2.5  # tiếng Việt tốn token hơn tiếng Anh nên chia 2.5 (ước lượng nhiều lên cho an toàn)
    items = [m["content"] for m in msgs] if msgs else (content if isinstance(content, list) else [content])
    for p in items:
        if isinstance(p, str):
            n += len(p) / 2.5
        elif getattr(p, "text", None):
            n += len(p.text) / 2.5
        elif getattr(p, "inline_data", None):
            n += 6000 if p.inline_data.mime_type == "application/pdf" else 1500
    return int(n)


def _digest(system, content, tag):
    h = hashlib.sha256()
    for x in (MODEL, system, tag):
        h.update(x.encode())
    for p in content if isinstance(content, list) else [content]:
        if isinstance(p, str):
            h.update(p.encode())
        elif getattr(p, "text", None) is not None:
            h.update(p.text.encode())
        else:
            h.update(p.inline_data.data)
    return h.hexdigest()


def memo(key, fn):
    """Có kết quả cũ thì dùng lại (không tốn lượt). Nhiều người hỏi cùng lúc một thứ thì chỉ gọi AI một lần."""
    sh = shared()
    v = sh.cget(key)
    if v is not None:
        return v
    with sh.flight_lock(key):
        v = sh.cget(key)
        if v is None:
            v = fn()
            if v is not None:
                sh.cput(key, v)
        sh.flight.pop(key, None)
        return v


# ───────────────────────── GỌI AI (GEMINI) ─────────────────────────
def _key_and_source():
    """Ưu tiên biến môi trường (GitHub Secrets), sau đó Secrets của Streamlit Community Cloud."""
    k = os.environ.get(KEY_NAME, "").strip()
    if k:
        return k, "GitHub Secrets / biến môi trường"
    try:
        k = str(st.secrets.get(KEY_NAME, "") or "").strip()
    except Exception:  # chưa có tệp secrets.toml
        k = ""
    return (k, "Streamlit Secrets") if k else ("", "")


def _key():
    return _key_and_source()[0]


def _has_key():
    return len(_key()) >= 20


@st.cache_resource
def _client(key):
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180_000))


def ask(system, content=None, msgs=None, max_tokens=3000, think="minimal", json_mode=False):
    """Một lượt gọi thật tới Gemini, đi qua bộ điều tiết. Trả về text hoặc None (đã báo lỗi, không lộ khóa)."""
    if not _has_key():
        st.error("Chưa cấu hình GEMINI_API_KEY. Quản trị viên cần thêm khóa vào GitHub Secrets hoặc Streamlit Secrets.")
        return None
    sh, ip = shared(), _ip()
    who = [(("u", (S.get("user") or "?").lower()), USER_DAILY)] + ([(("ip", ip), IP_DAILY)] if ip else [])
    est = est_tokens(system, content, msgs) + max_tokens + HEAD[think]
    state, h = sh.reserve(who, est)
    if state == "wait":
        with st.spinner("Hệ thống đang đông, bạn đang được xếp hàng..."):
            end = time.time() + QUEUE_MAX
            while state == "wait" and time.time() < end:
                time.sleep(min(max(h, 0.3), 1.5))
                state, h = sh.reserve(who, est)
    if state == "wait":
        st.error("Hệ thống đang rất đông. Hãy thử lại sau ít phút.")
        return None
    if state == "day":
        st.error("Hệ thống đã dùng hết hạn mức AI của hôm nay. Hạn mức làm mới khoảng 14-15h (giờ Việt Nam).")
        return None
    if state == "user":
        st.error(f"Bạn đã dùng hết {USER_DAILY} lượt AI hôm nay. Hãy quay lại vào ngày mai." if h == "u"
                 else "Mạng bạn đang dùng đã hết lượt AI hôm nay. Hãy quay lại vào ngày mai.")
        return None
    if msgs:
        content = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                 parts=[types.Part(text=m["content"])]) for m in msgs]  # lượt cuối luôn là "user"
    cfg = types.GenerateContentConfig(
        system_instruction=system, max_output_tokens=max_tokens + HEAD[think],
        thinking_config=types.ThinkingConfig(thinking_level=think),
        response_mime_type="application/json" if json_mode else None)
    r = None
    for attempt in range(2):  # lỗi mạng hoặc máy chủ 5xx thoáng qua: thử lại một lần
        try:
            r = _client(_key()).models.generate_content(model=MODEL, contents=content, config=cfg)
            break
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code == 429:
                daily = "perday" in str(e).lower().replace(" ", "")
                sh.penalize(3600 if daily else 45, day=daily)
                st.error("AI đã hết hạn mức hôm nay." if daily else "AI đang quá tải. Hãy đợi khoảng 1 phút rồi thử lại.")
                return None
            if code in (400, 401, 403):
                st.error(f"Yêu cầu bị từ chối (mã {code}). Kiểm tra API key hoặc tệp tải lên.")
                return None
            if attempt:
                st.error(f"Lỗi kết nối AI (mã {code}). Hãy thử lại.")
                return None
        except Exception as e:
            if attempt:
                st.error(f"Lỗi kết nối AI ({type(e).__name__}). Hãy thử lại.")
                return None
        time.sleep(1.5)
    used = getattr(getattr(r, "usage_metadata", None), "total_token_count", None)
    if used:
        sh.settle(h, used)
    try:
        text = r.text
    except Exception:
        text = None
    if text:
        return text.strip()
    st.error("AI không trả lời (có thể do bộ lọc an toàn hoặc bị cắt). Hãy diễn đạt lại và thử lại.")
    return None


def ask_cached(system, content, max_tokens=3000, think="minimal"):
    """Văn bản trả lời, dùng lại kết quả của người khác nếu đề y hệt."""
    return memo(_digest(system, content, f"t{think}{max_tokens}"),
                lambda: ask(system, content, max_tokens=max_tokens, think=think))


def _fix(o):
    """JSON nuốt \\f \\t \\r \\b của LaTeX (\\frac, \\text, \\right, \\bar): khôi phục lại."""
    if isinstance(o, str):
        return o.replace("\x0c", "\\f").replace("\x08", "\\b").replace("\t", "\\t").replace("\r", "\\r")
    if isinstance(o, list):
        return [_fix(i) for i in o]
    if isinstance(o, dict):
        return {k: _fix(v) for k, v in o.items()}
    return o


def ask_json(system, content, max_tokens=3000, think="minimal", cache=True):
    def run():
        for _ in range(2):
            t = ask(system, content, max_tokens=max_tokens, think=think, json_mode=True)
            if t is None:
                return None
            try:
                v = _fix(json.loads(t[t.index("{"): t.rindex("}") + 1], strict=False))
                if isinstance(v, dict):
                    return v
            except ValueError:
                continue
        st.error("AI trả về dữ liệu không đọc được. Hãy thử lại.")
        return None

    return memo(_digest(system, content, f"j{think}{max_tokens}"), run) if cache else run()


# ───────────────────────── FILE / OCR ─────────────────────────
def _T(s):
    return types.Part(text=s)


def fblocks(files):
    """Ảnh -> JPEG thu nhỏ (ảnh điện thoại rất nặng); PDF gửi nguyên. Gemini đọc chữ viết tay (OCR)."""
    out = []
    if files and len(files) > 3:
        st.warning("Chỉ dùng 3 tệp đầu tiên để tiết kiệm lượt AI.")
    for f in (files or [])[:3]:
        data = f.getvalue()
        if f.name.lower().endswith(".pdf"):
            if len(data) > 10_000_000:
                st.warning(f"{f.name} lớn hơn 10 MB, bỏ qua.")
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
        st.info("Chưa có dữ liệu. Hãy làm bài, luyện tập hoặc vấn đáp trong một quy trình để hồ sơ bắt đầu ghi nhận.")
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

    by_wf = Counter(WF_LOG.get(e.get("mode"), "Khác") for e in ev)
    st.markdown("**Số lượt theo quy trình**")
    st.bar_chart(pd.Series(by_wf, name="Số lượt"), color="#2F5D8C", horizontal=True)

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
                "luot_theo_quy_trinh": dict(by_wf), "loi_theo_nhom": dict(by_type),
                "chu_de_yeu_nhat": topic.head(3).to_dict("records"),
                "lo_hong_hay_gap": dict(gaps.most_common(8)),
                "sai_lam_gan_day": [x.get("mo_ta", "") for x in all_err[-6:]]}
        with st.spinner("Đang phân tích hồ sơ..."):
            S.r_plan = ask_cached(SYS_PLAN, json.dumps(data, ensure_ascii=False), max_tokens=1500)
    if S.get("r_plan"):
        st.markdown(S.r_plan)


@st.dialog("Bấm máy Casio fx-580VN X cho Xác suất thống kê", width="large")
def casio_dialog():
    st.markdown(CASIO_GUIDE)
    st.caption("Muốn hướng dẫn bấm máy cho đúng bài bạn đang làm: vào quy trình Giải toán, bước 4, bấm “Bấm máy fx-580VN X”.")
    disclaimer()


# ───────────────────────── GIAO DIỆN CHUNG ─────────────────────────
def disclaimer():
    st.markdown('<div class="disclaimer" role="note"><b>Lưu ý:</b><span>Nội dung do AI tạo ra, xin hãy kiểm chứng '
                'với đề cương và giáo trình môn học.</span></div>', unsafe_allow_html=True)


BELL = (  # đường cong phân phối chuẩn, vùng giữa được tô
    '<svg class="bell" viewBox="0 0 64 36" role="img" aria-label="đường cong phân phối chuẩn">'
    '<path d="M22 33 L22 11 C25 6 28 4 32 4 C36 4 39 6 42 11 L42 33 Z" fill="currentColor" opacity=".18"/>'
    '<path d="M2 33 C14 33 20 4 32 4 C44 4 50 33 62 33" fill="none" stroke="currentColor" stroke-width="3" '
    'stroke-linecap="round"/><line x1="32" y1="5" x2="32" y2="33" stroke="currentColor" stroke-width="1.6" '
    'stroke-dasharray="2.5 3"/><line x1="1" y1="34" x2="63" y2="34" stroke="currentColor" stroke-width="1.6"/></svg>'
)


def _art():
    """Hình trang trí: biểu đồ tần số và đường cong chuẩn, kèm vài kí hiệu XSTK."""
    hs = [14, 26, 46, 74, 104, 128, 136, 122, 98, 70, 44, 24, 12]
    bars = "".join(f'<rect x="{34 + i * 23}" y="{190 - h}" width="19" height="{h}" rx="3" '
                   f'fill="{"#0F5C5A" if 4 <= i <= 8 else "#0F5C5A55"}"/>' for i, h in enumerate(hs))
    return (
        '<svg viewBox="0 0 360 230" aria-hidden="true" focusable="false">'
        f'{bars}'
        '<path d="M22 190 C110 190 120 46 183 46 C246 46 256 190 344 190" fill="none" stroke="#B4531F" '
        'stroke-width="3" stroke-linecap="round"/>'
        '<line x1="183" y1="40" x2="183" y2="198" stroke="#1F2A27" stroke-width="1.2" stroke-dasharray="3 4"/>'
        '<line x1="16" y1="191" x2="350" y2="191" stroke="#1F2A27" stroke-width="1.5"/>'
        '<text x="183" y="216" text-anchor="middle" class="t">μ</text>'
        '<text x="117" y="216" text-anchor="middle" class="t">μ − σ</text>'
        '<text x="249" y="216" text-anchor="middle" class="t">μ + σ</text>'
        '<text x="18" y="34" class="f">P(A | B) = P(AB) / P(B)</text>'
        '<text x="250" y="96" class="f">σ<tspan dy="-6" font-size="10">2</tspan><tspan dy="6"> = E(X</tspan>'
        '<tspan dy="-6" font-size="10">2</tspan><tspan dy="6">) − μ</tspan><tspan dy="-6" font-size="10">2</tspan></text>'
        '<text x="18" y="70" class="f">x</text><line x1="18.5" y1="58.5" x2="26" y2="58.5" stroke="#1F2A27" '
        'stroke-width="1.2" opacity=".72"/><text x="30" y="70" class="f">± z · s / √n</text>'
        '</svg>'
    )


CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root{--paper:#F5F1E8;--card:#FBF9F4;--ink:#1F2A27;--muted:#5E6B66;--line:#D9D2C3;--teal:#0F5C5A;--teal-2:#E3EEEC;--clay:#B4531F;--amber:#8A6A12}
.stApp{background:radial-gradient(circle at 1px 1px,rgba(15,92,90,.07) 1px,transparent 0) 0 0/22px 22px,var(--paper);color:var(--ink);
  font-family:'Be Vietnam Pro',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
/* thanh header của Streamlit nằm đè lên nội dung: làm trong suốt và chừa khoảng trống phía trên */
header[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:1120px;padding:4.2rem 1.1rem 5rem!important}
#MainMenu,footer{visibility:hidden}
[data-testid="stHeaderActionElements"]{display:none}
h1,h2,h3,h4{font-family:'Newsreader',Georgia,serif!important;color:var(--ink);letter-spacing:-.01em}
h4{font-size:1.25rem!important;padding:.1rem 0 .4rem!important}
.eyebrow{display:inline-block;font-size:.74rem;line-height:1.7;letter-spacing:.14em;text-transform:uppercase;color:var(--teal);
  font-weight:600;padding:.05rem 0}
.brand{display:flex;align-items:center;gap:.6rem;font-weight:700;font-size:1.05rem;color:var(--ink);line-height:1.4;padding:.2rem 0}
.brand .logo{display:grid;place-items:center;width:38px;height:38px;border-radius:11px;background:var(--teal);color:#fff}
.brand .logo .bell{width:24px;height:auto;margin:0}
.brand small{display:block;font-weight:500;font-size:.75rem;color:var(--muted)}
.hero{display:grid;grid-template-columns:1.35fr 1fr;gap:2rem;align-items:center;margin:.6rem 0 1.4rem;padding:1.6rem 1.8rem;
  background:linear-gradient(135deg,var(--card) 0%,#F1ECE0 100%);border:1px solid var(--line);border-radius:22px;
  box-shadow:0 1px 0 rgba(255,255,255,.6) inset,0 18px 40px -28px rgba(31,42,39,.35)}
.hero .title{font-family:'Newsreader',Georgia,serif;font-weight:600;font-size:clamp(1.9rem,4.6vw,3rem);line-height:1.18;
  letter-spacing:-.015em;color:var(--ink);margin:.35rem 0 .7rem;padding-top:.15em}
.hero .title .bell{width:1.3em;height:.75em;color:var(--teal);vertical-align:-.04em;margin:0 .08em}
.hero .title em{font-style:normal;color:var(--teal)}
.lead{color:var(--muted);font-size:1.04rem;line-height:1.65;max-width:36rem;margin:0}
.chips{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem}
.chips span{font-size:.8rem;font-weight:500;padding:.3rem .7rem;border-radius:999px;background:var(--teal-2);color:var(--teal)}
.hero-art svg{width:100%;height:auto;display:block}
.hero-art .t{font:500 13px 'Be Vietnam Pro',sans-serif;fill:var(--muted)}
.hero-art .f{font-style:italic;font-weight:500;font-size:14px;fill:var(--ink);opacity:.72;
  font-family:'Cambria Math','STIX Two Math','Latin Modern Math','Times New Roman','Newsreader',serif}
.flow{display:flex;gap:1rem;align-items:flex-start;padding:1rem 1.2rem;border-radius:16px;background:var(--card);
  border:1px solid var(--line);margin:.2rem 0 .8rem}
.flow .ico{flex:none;width:42px;height:42px;border-radius:12px;background:var(--teal);color:#fff;display:grid;place-items:center;
  font:600 1.15rem 'Newsreader',serif}
.flow b{font-size:1.02rem}.flow p{margin:.15rem 0 0;color:var(--muted);font-size:.93rem;line-height:1.5}
.chain{list-style:none;display:flex;flex-wrap:wrap;gap:.35rem .5rem;padding:0;margin:.65rem 0 0}
.chain li{display:flex;align-items:center;gap:.4rem;font-size:.82rem;color:var(--muted);margin:0}
.chain li i{font-style:normal;display:grid;place-items:center;width:22px;height:22px;border-radius:50%;
  border:1.5px solid var(--line);font-weight:600;font-size:.75rem;background:#fff}
.chain li.on{color:var(--ink);font-weight:600}.chain li.on i{background:var(--teal);border-color:var(--teal);color:#fff}
.chain li.done i{background:var(--teal-2);border-color:var(--teal);color:var(--teal)}
.chain li+li::before{content:"";width:14px;height:1.5px;background:var(--line);margin-right:.1rem}
.stepcard{border-left:4px solid var(--teal);background:var(--card);border-radius:4px 14px 14px 4px;padding:.85rem 1.1rem;
  margin:.4rem 0 1rem;border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.stepcard .no{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);font-weight:600}
.stepcard .nm{font-family:'Newsreader',Georgia,serif;font-size:1.45rem;font-weight:600;line-height:1.3;margin:.1rem 0 .2rem}
.stepcard p{margin:.1rem 0;color:var(--ink);font-size:.95rem;line-height:1.55}
.stepcard p.todo{color:var(--muted)}
.bar{height:5px;border-radius:9px;background:var(--line);overflow:hidden;margin:.55rem 0 0}
.bar span{display:block;height:100%;background:var(--teal);border-radius:9px;transition:width .3s ease}
.sec{font-family:'Newsreader',Georgia,serif;font-size:1.25rem;font-weight:600;margin:.2rem 0 .3rem}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px}
/* mọi nút giãn đều theo cột chứa nó: các hàng nút thẳng hàng, cùng kích thước */
[data-testid="stElementContainer"]:has(.stButton),[data-testid="stElementContainer"]:has(.stDownloadButton),
[data-testid="stElementContainer"]:has(.stFormSubmitButton),[data-testid="stElementContainer"]:has([data-testid="stPopover"]){width:100%!important}
.stButton,.stDownloadButton,.stFormSubmitButton,[data-testid="stPopover"]{width:100%!important}
.stButton button,.stDownloadButton button,.stFormSubmitButton button,[data-testid="stPopover"] button{width:100%!important;
  min-height:46px;border-radius:12px;font-weight:500;justify-content:center;
  transition:transform .15s ease,border-color .15s ease,box-shadow .15s ease}
.stButton button p,.stDownloadButton button p{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stButton button:hover,.stDownloadButton button:hover{transform:translateY(-1px);border-color:var(--teal);
  box-shadow:0 6px 16px -10px rgba(15,92,90,.6)}
.stButton button:active{transform:scale(.98)}
[data-testid="stColumn"]{min-width:0}
[data-testid="stPills"] button{min-height:34px}
.disclaimer{display:flex;gap:.5rem;align-items:flex-start;margin-top:.8rem;padding:.7rem .9rem;border-radius:12px;
  background:#F6EBDD;border:1px solid #E7D3B8;color:#6B4A1F;font-size:.86rem;line-height:1.5}
[data-testid="stButtonGroup"] button{min-height:42px}
:focus-visible{outline:2px solid var(--teal)!important;outline-offset:2px}
.score{display:flex;align-items:baseline;gap:.3rem;font-family:'Newsreader',Georgia,serif;margin:.1rem 0 .5rem}
.score b{font-size:3.4rem;line-height:1.1;font-weight:600}.score span{color:var(--muted);font-size:1.2rem}
.score.teal b{color:var(--teal)}.score.amber b{color:var(--amber)}.score.clay b{color:var(--clay)}
.foot{margin-top:3rem;padding-top:1.2rem;border-top:1px solid var(--line);color:var(--muted);font-size:.82rem;
  display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;justify-content:space-between}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.hero,.flow,.stepcard{animation:rise .45s ease both}
@media (max-width:760px){
  .block-container{padding:3.6rem .8rem 4rem!important}
  .hero{grid-template-columns:1fr;padding:1.2rem 1.1rem;gap:.5rem}
  .hero-art{display:none}
  .flow{padding:.8rem .9rem}
  .chain li+li::before{display:none}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>"""


def hero():
    st.markdown(
        '<section class="hero"><div>'
        '<div class="eyebrow">Xác suất thống kê · Trợ giảng AI</div>'
        f'<div class="title" role="heading" aria-level="1"><em>Hiểu bản chất</em> {BELL}<br>giải đúng từng bài toán</div>'
        '<p class="lead">Học khái niệm, học định lí và giải toán theo quy trình 4 bước, có trợ giảng AI đồng hành.</p>'
        '<div class="chips"><span>Học khái niệm</span><span>Học định lí</span><span>Giải toán 4 bước</span>'
        '<span>Hồ sơ lỗi sai cá nhân</span></div>'
        f'</div><div class="hero-art">{_art()}</div></section>',
        unsafe_allow_html=True)


def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


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
        for x in g["lo_hong"] if isinstance(g["lo_hong"], list) else [g["lo_hong"]]:
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


def toggle(k):
    S[k] = not S.get(k)


_TEX = re.compile(r"\\[a-zA-Z]+|[\^_]")


def mathify(s):
    """AI đôi khi quên đặt công thức trong $...$ (ví dụ phương án "\\frac{7}{15}"): bọc các cụm công thức lại để KaTeX hiển thị."""
    s = str(s or "")
    if "$" in s or not _TEX.search(s):
        return s
    out, run = [], []
    for tok in s.split(" "):
        if _TEX.search(tok) or (run and re.fullmatch(r"[=<>+\-*/()\d.,]+|\\?[a-zA-Z]", tok)):
            run.append(tok)
            continue
        if run:
            out.append("$" + " ".join(run) + "$")
            run = []
        out.append(tok)
    if run:
        out.append("$" + " ".join(run) + "$")
    return " ".join(out)


def half():
    """Một nút đứng riêng chiếm nửa hàng, để các nút trên trang có cùng bề rộng."""
    return st.columns(2)[0]


# ── Bàn phím kí hiệu toán cho ô gõ ──
SYMBOLS = ["∩", "∪", "Ā", "∅", "Ω", "⊂", "∈", "|", "≤", "≥", "≠", "≈", "±", "√", "π", "μ", "σ", "σ²", "λ", "α",
           "χ²", "∑", "∞", "→", "⇒", "×", "÷", "·", "²", "³", "⁻¹", "x̄", "p̂", "n!"]
TEX_TEMPLATES = {
    "Phân số": r"$\frac{a}{b}$", "Tổ hợp": r"$C_{n}^{k}$", "Chỉnh hợp": r"$A_{n}^{k}$", "Căn": r"$\sqrt{x}$",
    "Lũy thừa": r"$x^{2}$", "Chỉ số": r"$x_{i}$", "Biến cố đối": r"$\overline{A}$", "XS có điều kiện": r"$P(A \mid B)$",
    "Trung bình mẫu": r"$\bar{x}$", "Tổng": r"$\sum_{i=1}^{n} x_i$", "Kì vọng": r"$E(X)$", "Phương sai": r"$D(X)$",
    "Tích phân": r"$\int_{a}^{b} f(x)\,dx$", "Hệ": r"$\begin{cases} a \\ b \end{cases}$",
}


def _insert(target, pk, templates, limit):
    v = S.get(pk)
    if v:
        add = TEX_TEMPLATES[v] if templates else v
        cur = S.get(target) or ""
        sep = " " if templates and cur and not cur.endswith((" ", "\n")) else ""
        S[target] = (cur + sep + add)[:limit]
    S[pk] = None  # bỏ chọn để bấm lại cùng kí hiệu được


def math_area(label, key, height=130, max_chars=4000, placeholder=None, collapsed=False):
    """Ô gõ có bàn phím kí hiệu toán và xem trước công thức (LaTeX trong $...$)."""
    v = st.text_area(label, key=key, height=height, max_chars=max_chars, placeholder=placeholder,
                     label_visibility="collapsed" if collapsed else "visible")
    with st.expander("Bàn phím toán · xem trước công thức", icon=":material/keyboard:"):
        st.pills("Kí hiệu", SYMBOLS, key=f"{key}__sym", on_change=_insert, args=(key, f"{key}__sym", False, max_chars))
        st.pills("Mẫu công thức", list(TEX_TEMPLATES), key=f"{key}__tex", on_change=_insert,
                 args=(key, f"{key}__tex", True, max_chars))
        st.caption("Bấm để chèn vào cuối ô. Mẫu công thức viết bằng LaTeX giữa hai dấu $, sửa a, b, n, k... thành số của bạn.")
        if (v or "").strip():
            with st.container(border=True):
                st.markdown(v)
        else:
            st.caption("Phần xem trước hiện ở đây khi bạn gõ.")
    return v


def _token():
    return f"{random.getrandbits(40):x}"


def clear_prefix(p):
    for k in [x for x in S if isinstance(x, str) and x.startswith(p)]:
        del S[k]


def result_box(title, body):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(body)


def ai_button(R, slot, label, icon, run, primary=True, solo=True):
    """Nút gọi AI; kết quả lưu ở R[slot] để không mất khi chuyển bước. solo: nút đứng riêng thì chiếm nửa hàng."""
    where = half() if solo else st
    if where.button(label, icon=icon, type="primary" if primary and not R.get(slot) else "secondary",
                    key=f"b_{R['id']}_{slot}"):
        with st.spinner("Trợ giảng AI đang soạn..."):
            v = run()
        if v:
            R[slot] = v


# ───────────────────────── KHUNG QUY TRÌNH 4 BƯỚC ─────────────────────────
def _go(wf, i):
    S[f"{wf}_step"] = i


def flow_banner(wf, cur):
    W = WF[wf]
    chain = "".join(
        f'<li class="{"on" if i + 1 == cur else "done" if i + 1 < cur else ""}"><i>{i + 1}</i>{n}</li>'
        for i, (n, _, _) in enumerate(W["steps"]))
    st.markdown(
        f'<div class="flow"><div class="ico">{list(WF).index(wf) + 1}</div><div><b>Quy trình {W["name"].lower()}</b>'
        f'<p>{W["desc"]}</p><ol class="chain">{chain}</ol></div></div>', unsafe_allow_html=True)


def step_nav(wf):
    """Thanh chọn bước. Trả về bước hiện tại (1-4)."""
    W, k = WF[wf], f"{wf}_step"
    if S.get(k) not in (1, 2, 3, 4):
        S[k] = S.get(f"{k}_last", 1)  # lần đầu hoặc khi bấm bỏ chọn
    cur = S[k]
    flow_banner(wf, cur)
    st.segmented_control("Bước", [1, 2, 3, 4], key=k, label_visibility="collapsed",
                         format_func=lambda i: f"Bước {i} · {W['steps'][i - 1][0]}")
    cur = S.get(k) or cur
    S[f"{k}_last"] = cur
    name, goal, todo = W["steps"][cur - 1]
    st.markdown(
        f'<div class="stepcard"><div class="no">Bước {cur}/4</div><div class="nm">{name}</div>'
        f'<p>{goal}</p><p class="todo">Việc của bạn: {todo}</p>'
        f'<div class="bar"><span style="width:{cur * 25}%"></span></div></div>', unsafe_allow_html=True)
    return cur


def step_footer(wf, cur):
    st.write("")
    a, _, b = st.columns([1, 1.4, 1])
    if cur > 1:
        a.button("Bước trước", icon=":material/arrow_back:", key=f"{wf}_prev", on_click=_go, args=(wf, cur - 1))
    if cur < 4:
        b.button(f"Sang bước {cur + 1}", icon=":material/arrow_forward:", key=f"{wf}_next", type="primary",
                 on_click=_go, args=(wf, cur + 1))


def topic_pick(wf):
    W = WF[wf]
    a, b = st.columns([1.3, 1], gap="medium")
    sel = a.selectbox(W["pick"], W["topics"], key=f"{wf}_sel")
    own = b.text_input("Hoặc nhập chủ đề riêng", key=f"{wf}_own", max_chars=80,
                       placeholder="Ví dụ: Biến cố đồng khả năng" if wf == "concept" else "Ví dụ: Bất đẳng thức Chebyshev")
    return own.strip() or sel


def wf_state(wf, topic):
    """Kết quả của một quy trình gắn với chủ đề đang học; đổi chủ đề thì bắt đầu lại sạch sẽ."""
    R = S.get(f"{wf}_R")
    if not R or R.get("topic") != topic:
        if R:
            clear_prefix(f"w{R['id']}")
        R = {"topic": topic, "id": _token(), "wf": wf}
        S[f"{wf}_R"] = R
    return R


# ───────────────────────── KHỐI CHỨC NĂNG DÙNG CHUNG ─────────────────────────
def self_check(wf, R, topic):
    what = "định nghĩa khái niệm" if wf == "concept" else "định lí, công thức"
    with st.container(border=True):
        sec(f"Tự phát biểu lại {what}")
        st.caption("Viết bằng lời của bạn, không chép lại. AI chỉ ra chỗ thiếu dấu hiệu bản chất, giả thiết hoặc điều kiện.")
        txt = math_area("Phát biểu của bạn", f"w{R['id']}_self", height=110, max_chars=1500, collapsed=True,
                        placeholder="Ví dụ: ... là ... khi và chỉ khi ...")
        if half().button("Nhận xét phát biểu", icon=":material/rate_review:", key=f"b_{R['id']}_self"):
            if not txt.strip():
                st.warning("Hãy viết phát biểu của bạn trước.")
            else:
                with st.spinner("Đang đọc phát biểu..."):
                    g = ask_json(SYS_SELF, f"Chủ đề: {topic}\nLoại: {what}\nPhát biểu của học sinh:\n{txt.strip()}",
                                 max_tokens=1200)
                if g:
                    g.setdefault("chu_de", topic)
                    if g != R.get("self"):
                        log_grade(wf, g)
                    R["self"] = g
        if R.get("self"):
            g = R["self"]
            show_score(num(g.get("diem")))
            st.markdown(str(g.get("nhan_xet", "")))
            show_issues(g)


def oral_first(topic, kind):
    """Câu 1 lấy từ mẫu có sẵn, không tốn lượt AI."""
    if kind.startswith("Khái"):
        return f"Câu 1/5. Hãy phát biểu định nghĩa của «{topic}» bằng lời của bạn và nêu các dấu hiệu bản chất của khái niệm."
    return f"Câu 1/5. Hãy phát biểu định lí hoặc công thức «{topic}», nói rõ các đại lượng có mặt trong đó."


def oral_sys(O):
    n = sum(m["role"] == "assistant" for m in O["msgs"])
    plan = ORAL_CONCEPT if O["kind"].startswith("Khái") else ORAL_THEOREM
    return BASE + (
        f"Bạn là giám khảo vấn đáp lí thuyết XSTK, chủ đề: {O['topic']}. {plan} "
        f"Học sinh vừa trả lời câu {n}/5. Hãy nhận xét 1-2 câu (đúng, chưa đủ hay sai ở đâu, có gợi mở), KHÔNG giảng đáp án "
        f"đầy đủ, rồi hỏi duy nhất câu {n + 1}/5 theo đúng thứ tự trên, bắt đầu câu hỏi bằng \"Câu {n + 1}/5.\". "
        "Không cho điểm số trong lúc hỏi."
    )


def oral_block(wf, R, topic):
    kind = WF[wf]["kind"]
    O = R.get("oral")
    plan = ORAL_CONCEPT if wf == "concept" else ORAL_THEOREM
    st.caption("Giám khảo AI hỏi lần lượt 5 câu. " + plan.split(": ", 1)[1])
    if half().button("Bắt đầu vấn đáp" if not O else "Vấn đáp lại từ đầu", icon=":material/record_voice_over:",
                 type="primary" if not O else "secondary", key=f"b_{R['id']}_ostart"):
        O = R["oral"] = {"topic": topic, "kind": kind, "res": None,
                         "msgs": [{"role": "user", "content": "Hãy bắt đầu vấn đáp."},
                                  {"role": "assistant", "content": oral_first(topic, kind)}]}
    if not O:
        return
    for m in O["msgs"][1:]:
        st.chat_message(m["role"], avatar=":material/school:" if m["role"] == "assistant" else ":material/person:"
                        ).markdown(m["content"])
    done = bool(O.get("res"))
    answered = sum(m["role"] == "user" for m in O["msgs"]) - 1
    p = st.chat_input("Nhập câu trả lời của bạn...", disabled=done or answered >= 5, max_chars=1200, key=f"{wf}_chat")
    if p and not done and answered < 5:
        O["msgs"].append({"role": "user", "content": p})
        st.chat_message("user", avatar=":material/person:").markdown(p)
        answered += 1
        if answered < 5:  # câu trả lời thứ 5 không cần nhận xét riêng, để dành lượt AI cho phần chấm điểm
            with st.chat_message("assistant", avatar=":material/school:"):
                with st.spinner("Giám khảo đang nhận xét..."):
                    r = ask(oral_sys(O), msgs=O["msgs"], max_tokens=600)
                if r:
                    st.markdown(r)
                    O["msgs"].append({"role": "assistant", "content": r})
                else:
                    O["msgs"].pop()
                    answered -= 1
    if not done:
        st.progress(min(answered, 5) / 5, text=f"Đã trả lời {min(answered, 5)}/5 câu")
    if answered >= 5 and not done:
        st.info("Bạn đã trả lời đủ 5 câu. Bấm “Kết thúc và chấm điểm” để nhận kết quả.")
    if not done and answered >= 1 and half().button("Kết thúc và chấm điểm", icon=":material/fact_check:",
                                                  type="primary", key=f"b_{R['id']}_oend"):
        tr = "\n".join(f"{'Giám khảo' if m['role'] == 'assistant' else 'Học sinh'}: {m['content']}" for m in O["msgs"][1:])
        with st.spinner("Đang chấm phần vấn đáp..."):
            res = ask_json(SYS_ORAL_GRADE, f"Chủ đề: {O['topic']}\nDạng: {O['kind']}\nBản ghi:\n{tr}",
                           max_tokens=2500, think="medium")
        if res:
            res.setdefault("chu_de", O["topic"])
            O["res"] = res
            log_grade(wf, res)
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


# ── Luyện tập: trắc nghiệm và tự luận tạo riêng ──
def ok_mc(q):
    return (isinstance(q, dict) and q.get("de") and isinstance(q.get("lua_chon"), dict)
            and all(x in q["lua_chon"] for x in "ABCD") and q.get("dap_an") in list("ABCD"))


def ok_es(q):
    return isinstance(q, dict) and bool(q.get("de"))


def get_items(kind, wf, topic, level):
    """Kho đề dùng chung: mỗi (loại, quy trình, chủ đề, mức độ) chỉ sinh tối đa POOL_SIZE bộ; người sau nhận bộ chưa làm."""
    sh, key = shared(), (kind, wf, topic.strip().lower(), level)
    seen = S.setdefault("pool_seen", set())
    with sh.flight_lock(("gen", key)):
        pool = sh.pool_get(key)
        fresh = [i for i in range(len(pool)) if (key, i) not in seen]
        if fresh:
            i = random.choice(fresh)
        elif len(pool) >= POOL_SIZE:
            i = random.randrange(len(pool))
        else:
            system = (SYS_MC if kind == "mc" else SYS_ES) + " " + FOCUS.get(wf, "")
            d = ask_json(system, f"Chủ đề: {topic}. Mức độ: {level}.", max_tokens=6000, think="medium", cache=False) or {}
            items = [q for q in (d.get("cau") or []) if (ok_mc(q) if kind == "mc" else ok_es(q))][:5]
            if not items:
                return None
            sh.pool_add(key, items)
            i = len(sh.pool_get(key)) - 1
        seen.add((key, i))
        return copy.deepcopy(sh.pool_get(key)[i])


def mc_card(k, i, q, topic, wf):
    with st.container(border=True):
        st.markdown(f"**Câu {i + 1}.** {mathify(q['de'])}")
        pick = st.radio("Chọn đáp án", list("ABCD"), index=None, key=k, label_visibility="collapsed",
                        format_func=lambda x: f"{x}. {mathify(q['lua_chon'][x])}")
        a, b, c = st.columns(3)
        a.button("Gợi ý", icon=":material/lightbulb:", key=f"b_{k}_h", on_click=toggle, args=(f"{k}_h",))
        if b.button("Kiểm tra", icon=":material/task_alt:", type="primary", key=f"b_{k}_g"):
            if pick is None:
                st.warning("Hãy chọn một đáp án.")
            else:
                S[f"{k}_g"] = pick
                if not S.get(f"{k}_l"):  # chỉ ghi hồ sơ một lần cho mỗi câu
                    S[f"{k}_l"] = 1
                    m = (q.get("sai_lam") or {}).get(pick)
                    log(wf, topic, 10.0 if pick == q["dap_an"] else 0.0,
                        [] if pick == q["dap_an"] else [m if isinstance(m, dict) else {"loai": "Khác", "mo_ta": "Chọn sai đáp án"}], [])
        c.button("Lời giải", icon=":material/menu_book:", key=f"b_{k}_s", on_click=toggle, args=(f"{k}_s",))
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


def es_card(k, i, q, topic, wf):
    with st.container(border=True):
        st.markdown(f"**Câu {i + 1}.** {mathify(q['de'])}")
        ans = math_area("Bài làm của bạn", f"{k}_a", height=130)
        with st.expander("Hoặc tải ảnh/PDF bài làm"):
            fl = st.file_uploader("Ảnh/PDF", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True,
                                  key=f"{k}_f", label_visibility="collapsed")
        a, b, c = st.columns(3)
        a.button("Gợi ý", icon=":material/lightbulb:", key=f"b_{k}_h", on_click=toggle, args=(f"{k}_h",))
        if b.button("Chấm bài", icon=":material/task_alt:", type="primary", key=f"b_{k}_g"):
            if not (ans.strip() or fl):
                st.warning("Hãy nhập hoặc tải bài làm.")
            else:
                ref = f"ĐÁP ÁN THAM KHẢO VÀ THANG ĐIỂM (chỉ để chấm, không tiết lộ):\n{q.get('loi_giai', '')}\n{q.get('tieu_chi', '')}"
                with st.spinner("Đang chấm..."):
                    g = ask_json(SYS_GRADE, build(q["de"], None, ans, fl, ref), max_tokens=3500, think="medium")
                if g:
                    g.setdefault("chu_de", topic)
                    if g != S.get(f"{k}_g"):
                        log_grade(wf, g)
                    S[f"{k}_g"] = g
        c.button("Lời giải", icon=":material/menu_book:", key=f"b_{k}_s", on_click=toggle, args=(f"{k}_s",))
        if S.get(f"{k}_g"):
            show_grade(S[f"{k}_g"])
        if S.get(f"{k}_h"):
            st.info(q.get("chien_luoc", ""))
        if S.get(f"{k}_s"):
            st.success(q.get("loi_giai", ""))


def practice_block(wf, R, topic):
    with st.container(border=True):
        sec("Luyện tập")
        level = st.radio("Mức độ", ["Cơ bản", "Trung bình", "Nâng cao"], index=1, key=f"{wf}_lv", horizontal=True)
        a, b = st.columns(2)
        for col, kind, label, icon in ((a, "mc", "Tạo 5 câu trắc nghiệm", ":material/checklist:"),
                                       (b, "es", "Tạo 5 câu tự luận", ":material/edit_note:")):
            if col.button(label, icon=icon, type="primary", key=f"b_{R['id']}_{kind}"):
                with st.spinner("Đang chuẩn bị bài tập..."):
                    items = get_items(kind, wf, topic, level)
                if items:
                    old = R.get(kind)
                    if old:
                        clear_prefix(f"w{R['id']}{kind}{old['gid']}")
                    R[kind] = {"items": items, "gid": _token(), "level": level}
                else:
                    st.error("Không tạo được bài tập. Hãy thử lại.")
        st.caption("Gợi ý chỉ nêu chiến lược, không chứa đáp án. Mỗi câu kiểm tra/chấm đều được ghi vào hồ sơ.")
    have = [k for k in ("mc", "es") if R.get(k)]
    if not have:
        return
    names = {"mc": "Trắc nghiệm", "es": "Tự luận"}
    tabs = st.tabs([f"{names[k]} · {R[k]['level']} ({len(R[k]['items'])})" for k in have])
    for tab, kind in zip(tabs, have):
        with tab:
            for i, q in enumerate(R[kind]["items"]):
                k = f"w{R['id']}{kind}{R[kind]['gid']}_{i}"
                (mc_card if kind == "mc" else es_card)(k, i, q, topic, wf)


# ── Sơ đồ tư duy ──
MINDMAP_HTML = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@500;600;700&display=swap" rel="stylesheet">
<style>
html,body{margin:0;background:transparent;font-family:'Be Vietnam Pro',system-ui,'Segoe UI',Roboto,sans-serif}
#wrap{position:relative;border:1px solid #D9D2C3;border-radius:16px;background:#FBF9F4;overflow:hidden}
svg{display:block;width:100%;height:__H__px;touch-action:none;cursor:grab}
.bar{position:absolute;top:10px;right:10px;display:flex;gap:6px;flex-wrap:wrap;z-index:2}
.bar button{font:600 13px 'Be Vietnam Pro',sans-serif;border:1px solid #D9D2C3;background:#fff;color:#1F2A27;
  border-radius:10px;padding:7px 11px;cursor:pointer}
.bar button:hover{border-color:#0F5C5A;color:#0F5C5A}
.hint{position:absolute;left:12px;bottom:8px;font-size:12px;color:#5E6B66}
text{font-family:'Be Vietnam Pro',system-ui,sans-serif;font-size:14px;font-weight:500}
</style></head><body><div id="wrap"><div class="bar"><button id="ex">Mở tất cả</button><button id="co">Thu gọn</button>
<button id="fit">Vừa khung</button></div><svg id="mm"></svg>
<div class="hint">Bấm vào ô để mở/thu nhánh · kéo để di chuyển · lăn chuột hoặc chụm 2 ngón để phóng to</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const DATA = __DATA__;
const PAL = ["#0F5C5A","#B4531F","#4F6D3A","#8A6A12","#2F5D8C","#8A3B4E"];
function start(){
  const svg = d3.select("#mm"), H = __H__;
  const g = svg.append("g"), gl = g.append("g").attr("fill","none").attr("stroke-width",1.8), gn = g.append("g");
  const zoom = d3.zoom().scaleExtent([0.25, 2.5]).on("zoom", e => g.attr("transform", e.transform));
  svg.call(zoom).on("dblclick.zoom", null);
  const root = d3.hierarchy(DATA, d => d.con && d.con.length ? d.con : null);
  // đo bề rộng chữ thật (đủ dấu tiếng Việt) để ô chữ không bị cắt
  const meas = svg.append("text").attr("opacity", 0);
  let id = 0;
  const every = root.descendants();  // lấy hết trước, vì vòng lặp dưới sẽ thu gọn các nhánh sâu
  every.forEach(d => {
    d.id = id++;
    const top = d.depth ? d.ancestors().slice(-2)[0] : d;  // nhánh cấp 1 chứa nút này
    d.color = d.depth === 0 ? "#1F2A27" : PAL[root.children.indexOf(top) % PAL.length];
    meas.style("font-size", d.depth === 0 ? "16px" : "14px").style("font-weight", d.depth < 2 ? 600 : 500).text(d.data.ten || "?");
    d.w = meas.node().getComputedTextLength() + 26 + (d.children ? 18 : 0);
    d._children = d.children;
    if (d.depth >= 2) d.children = null;
  });
  meas.remove();
  const maxW = []; every.forEach(d => { maxW[d.depth] = Math.max(maxW[d.depth] || 0, d.w); });
  const off = [0]; for (let i = 1; i < maxW.length; i++) off[i] = off[i - 1] + maxW[i - 1] + 48;
  const tree = d3.tree().nodeSize([40, 1]).separation((a, b) => a.parent === b.parent ? 1 : 1.3);
  const link = d3.linkHorizontal().x(p => p.y).y(p => p.x);
  root.x0 = 0; root.y0 = 0;
  function update(src){
    tree(root);
    root.each(d => { d.y = off[d.depth]; });
    const nodes = root.descendants(), links = root.links(), t = svg.transition().duration(320);
    const node = gn.selectAll("g.n").data(nodes, d => d.id);
    const en = node.enter().append("g").attr("class", "n").attr("transform", `translate(${src.y0},${src.x0})`)
      .attr("opacity", 0).style("cursor", d => d._children ? "pointer" : "default")
      .on("click", (e, d) => { if (d._children) { d.children = d.children ? null : d._children; update(d); } });
    en.append("rect").attr("y", -15).attr("height", 30).attr("rx", 10).attr("width", d => d.w)
      .attr("fill", d => d.depth < 2 ? d.color : d.color + "1F").attr("stroke", d => d.depth < 2 ? "none" : d.color)
      .attr("stroke-width", 1.2);
    en.append("text").attr("x", 13).attr("dy", "0.35em").attr("fill", d => d.depth < 2 ? "#fff" : "#1F2A27")
      .style("font-size", d => d.depth === 0 ? "16px" : "14px").style("font-weight", d => d.depth < 2 ? 600 : 500)
      .text(d => d.data.ten || "?");
    en.filter(d => d._children).append("text").attr("class", "tg").attr("x", d => d.w - 17).attr("dy", "0.35em")
      .attr("fill", d => d.depth < 2 ? "#fff" : d.color).style("font-weight", 700);
    const all = en.merge(node);
    all.select("text.tg").text(d => d.children ? "−" : "+");
    all.transition(t).attr("transform", d => `translate(${d.y},${d.x})`).attr("opacity", 1);
    node.exit().transition(t).attr("transform", `translate(${src.y},${src.x})`).attr("opacity", 0).remove();
    const lk = gl.selectAll("path").data(links, d => d.target.id);
    const o = {x: src.x0, y: src.y0 + (src.w || 0)};
    const le = lk.enter().append("path").attr("stroke", d => d.target.color).attr("stroke-opacity", .55)
      .attr("d", () => link({source: o, target: o}));
    le.merge(lk).transition(t).attr("d", d => link({source: {x: d.source.x, y: d.source.y + d.source.w}, target: {x: d.target.x, y: d.target.y}}));
    lk.exit().transition(t).attr("d", () => { const p = {x: src.x, y: src.y + (src.w || 0)}; return link({source: p, target: p}); }).remove();
    root.each(d => { d.x0 = d.x; d.y0 = d.y; });
  }
  function fit(){
    const b = g.node().getBBox(), W = svg.node().clientWidth || 800;
    if (!b.width) return;
    const k = Math.min(1.4, 0.92 * Math.min(W / b.width, H / b.height));
    svg.transition().duration(350).call(zoom.transform,
      d3.zoomIdentity.translate(W / 2 - k * (b.x + b.width / 2), H / 2 - k * (b.y + b.height / 2)).scale(k));
  }
  update(root); setTimeout(fit, 380);
  d3.select("#ex").on("click", () => { root.each(d => { if (d._children) d.children = d._children; }); update(root); setTimeout(fit, 380); });
  d3.select("#co").on("click", () => { root.each(d => { if (d.depth >= 1 && d._children) d.children = null; }); update(root); setTimeout(fit, 380); });
  d3.select("#fit").on("click", fit);
}
(document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(start);
</script></body></html>"""


def mindmap_html(tree, height=560):
    data = json.dumps(tree, ensure_ascii=False).replace("</", "<\\/")
    return MINDMAP_HTML.replace("__DATA__", data).replace("__H__", str(height))


def outline(n, d=0):
    if not isinstance(n, dict):
        return ""
    s = "  " * d + f"- {n.get('ten', '')}\n"
    return s + "".join(outline(c, d + 1) for c in (n.get("con") or []) if d < 3)


def map_block(wf, R, topic):
    with st.container(border=True):
        sec("Sơ đồ tư duy")
        st.caption("Hệ thống hoá: khái niệm, công thức, dạng bài, sai lầm thường gặp, ứng dụng và liên hệ.")
        focus = "khái niệm" if wf == "concept" else "định lí, công thức"

        def run():
            t = ask_json(SYS_MAP, f"Chủ đề ({focus}): {topic}", max_tokens=2500)
            if isinstance(t, dict) and t.get("ten"):
                return t
            if t is not None:
                st.error("Không tạo được sơ đồ. Hãy thử lại.")
            return None

        ai_button(R, "map", "Vẽ sơ đồ tư duy", ":material/account_tree:", run, primary=False)
    t = R.get("map")
    if t:
        components.html(mindmap_html(t), height=572)
        with st.expander("Dạng danh sách (dễ đọc trên điện thoại)"):
            st.markdown(outline(t))
        half().download_button("Tải sơ đồ (.md)", outline(t), file_name="so_do_tu_duy.md", mime="text/markdown",
                           icon=":material/download:", key=f"dl_{R['id']}_map")


# ───────────────────────── QUY TRÌNH HỌC KHÁI NIỆM / HỌC ĐỊNH LÍ ─────────────────────────
def page_learn(wf):
    with st.container(border=True):
        topic = topic_pick(wf)
    R = wf_state(wf, topic)
    cur = step_nav(wf)
    q = f"Chủ đề: {topic}"
    if wf == "concept":
        if cur == 1:
            ai_button(R, "c1", "Gợi mở tình huống tiếp cận", ":material/explore:",
                      lambda: ask_cached(SYS_C1, q, max_tokens=1400))
            if R.get("c1"):
                result_box("Tiếp cận khái niệm", R["c1"])
                math_area("Ghi lại dự đoán của bạn (không bắt buộc, chỉ lưu trong phiên)", f"w{R['id']}_guess",
                          height=90, max_chars=1500)
        elif cur == 2:
            ai_button(R, "c2", "Xem thẻ khái niệm", ":material/style:", lambda: ask_cached(SYS_C2, q, max_tokens=1800))
            if R.get("c2"):
                result_box("Thẻ khái niệm", R["c2"])
            self_check(wf, R, topic)
        elif cur == 3:
            oral_block(wf, R, topic)
        else:
            practice_block(wf, R, topic)
            map_block(wf, R, topic)
    else:
        if cur == 1:
            a, b = st.columns(2)
            with a:
                ai_button(R, "t1", "Tính thử và dự đoán", ":material/science:",
                          lambda: ask_cached(SYS_T1, q, max_tokens=1400), solo=False)
            with b:
                ai_button(R, "t2", "Xem phát biểu định lí", ":material/gavel:",
                          lambda: ask_cached(SYS_T2, q, max_tokens=1800), primary=bool(R.get("t1")), solo=False)
            if R.get("t1"):
                result_box("Tiếp cận định lí", R["t1"])
            if R.get("t2"):
                result_box("Phát biểu định lí", R["t2"])
            self_check(wf, R, topic)
        elif cur == 2:
            a, b = st.columns(2)
            with a:
                ai_button(R, "t3h", "Gợi mở chứng minh", ":material/psychology:",
                          lambda: ask_cached(SYS_T3_HINT, q, max_tokens=1500, think="medium"), solo=False)
            with b:
                ai_button(R, "t3f", "Xem chứng minh đầy đủ", ":material/menu_book:",
                          lambda: ask_cached(SYS_T3_FULL, q, max_tokens=2500, think="medium"), primary=False, solo=False)
            if R.get("t3h"):
                result_box("Hướng dẫn chứng minh", R["t3h"])
            if R.get("t3f"):
                with st.expander("Chứng minh đầy đủ", expanded=True):
                    st.markdown(R["t3f"])
            math_area("Tự viết chứng minh của bạn (nháp, chỉ lưu trong phiên)", f"w{R['id']}_proof", height=120)
        elif cur == 3:
            oral_block(wf, R, topic)
        else:
            practice_block(wf, R, topic)
            map_block(wf, R, topic)
    step_footer(wf, cur)


# ───────────────────────── QUY TRÌNH GIẢI TOÁN ─────────────────────────
def load_problem(text):
    """Đưa một bài tương tự vào ô đề bài và quay về bước 1."""
    S.s_prob = text
    S.s_pfn = S.get("s_pfn", 0) + 1  # đổi key để xoá tệp đề cũ
    S.pop("solve_R", None)
    S.solve_step = 1


def page_solve():
    ftypes = ["png", "jpg", "jpeg", "pdf"]
    with st.container(border=True):
        sec("Đề bài")
        a, b = st.columns([1.5, 1], gap="large")
        with a:
            prob = math_area("Nhập đề bài", "s_prob", height=140, collapsed=True,
                             placeholder="Ví dụ: Một hộp có 5 bi đỏ và 3 bi xanh. Lấy ngẫu nhiên 2 bi...")
        pf = b.file_uploader("Hoặc tải ảnh/PDF đề bài", type=ftypes, accept_multiple_files=True,
                             key=f"s_pf{S.get('s_pfn', 0)}")
    sig = hashlib.sha1((prob.strip() + "|" + "|".join(f"{f.name}{f.size}" for f in pf or [])).encode()).hexdigest()
    R = S.get("solve_R")
    if not R or R.get("sig") != sig:
        if R:
            clear_prefix(f"w{R['id']}")
        R = S.solve_R = {"sig": sig, "id": _token(), "wf": "solve"}
    has_p = bool(prob.strip() or pf)
    cur = step_nav("solve")
    if not has_p:
        st.info("Nhập hoặc tải đề bài ở khung phía trên để bắt đầu.")
        step_footer("solve", cur)
        return

    if cur == 1:
        st.markdown("Tự trả lời trước: **Đề cho gì? Hỏi gì? Kí hiệu các biến cố, đại lượng thế nào?** "
                    "Sau đó đối chiếu với phần phân tích của AI.")
        math_area("Phân tích của bạn (nháp)", f"w{R['id']}_s1", height=100, max_chars=2000)
        ai_button(R, "s1", "Phân tích đề cùng AI", ":material/search:",
                  lambda: ask_cached(SYS_S1, build(prob, pf), max_tokens=1400))
        if R.get("s1"):
            result_box("Tìm hiểu đề", R["s1"])
    elif cur == 2:
        math_area("Dàn ý lời giải của bạn (nháp)", f"w{R['id']}_s2", height=100, max_chars=2000)
        ai_button(R, "s2", "Gợi ý chiến lược giải", ":material/lightbulb:",
                  lambda: ask_cached(SYS_HINT, build(prob, pf), max_tokens=1800, think="medium"))
        if R.get("s2"):
            result_box("Chương trình giải", R["s2"])
    elif cur == 3:
        with st.container(border=True):
            sec("Bài làm của bạn")
            a, b = st.columns([1.5, 1], gap="large")
            with a:
                wt = math_area("Gõ bài làm", f"w{R['id']}_wt", height=170)
            wf_ = b.file_uploader("Hoặc tải ảnh chụp/PDF bài làm", type=ftypes, accept_multiple_files=True,
                                  key=f"w{R['id']}_wf")
            if half().button("Chấm bài", icon=":material/task_alt:", type="primary", key=f"b_{R['id']}_grade"):
                if not (wt.strip() or wf_):
                    st.warning("Hãy gõ hoặc tải bài làm của bạn.")
                else:
                    with st.spinner("Đang đọc bài làm và chấm điểm..."):
                        g = ask_json(SYS_GRADE, build(prob, pf, wt, wf_), max_tokens=3500, think="medium")
                    if g:
                        if g != R.get("grade"):  # bấm chấm lại y hệt: dùng kết quả cũ, không ghi hồ sơ lần nữa
                            log_grade("solve", g)
                        R["grade"] = g
        if R.get("grade"):
            with st.container(border=True):
                st.markdown("#### Kết quả chấm")
                show_grade(R["grade"])
                st.caption("Sửa bài theo nhận xét rồi chấm lại, hoặc sang bước 4 để đối chiếu và mở rộng.")
    else:
        a, b, c = st.columns(3)
        with c:
            ai_button(R, "casio", "Bấm máy fx-580VN X", ":material/calculate:",
                      lambda: ask_cached(SYS_CASIO, build(prob, pf), max_tokens=2000, think="medium"),
                      primary=False, solo=False)
        with a:
            ai_button(R, "sol", "Bài giải chi tiết và nhìn lại", ":material/menu_book:",
                      lambda: ask_cached(SYS_SOL, build(prob, pf), max_tokens=3500, think="medium"), primary=False, solo=False)
        with b:
            ai_button(R, "sim", "Tạo bài tập tương tự", ":material/library_add:",
                      lambda: ask_json(SYS_SIMILAR, build(prob, pf), max_tokens=4500, think="medium"), solo=False)
        if R.get("sol"):
            result_box("Bài giải chi tiết", R["sol"])
        if R.get("casio"):
            result_box("Bấm máy Casio fx-580VN X cho bài này", R["casio"])
        sim = R.get("sim")
        if sim:
            items = [x for x in (sim.get("bai") or []) if isinstance(x, dict) and x.get("de")][:3]
            sec("Bài tập tương tự" + (f" · {sim['chu_de']}" if sim.get("chu_de") else ""))
            for i, x in enumerate(items):
                k = f"w{R['id']}_sim{i}"
                with st.container(border=True):
                    st.markdown(f"**Bài {i + 1}** · *{x.get('muc_do', '')}*\n\n{mathify(x['de'])}")
                    c1, c2, c3 = st.columns(3)
                    c1.button("Luyện theo 4 bước", icon=":material/play_arrow:", type="primary", key=f"b_{k}_go",
                              on_click=load_problem, args=(x["de"],))
                    c2.button("Gợi ý", icon=":material/lightbulb:", key=f"b_{k}_h", on_click=toggle, args=(f"{k}_h",))
                    c3.button("Lời giải", icon=":material/visibility:", key=f"b_{k}_s", on_click=toggle,
                              args=(f"{k}_s",))
                    if S.get(f"{k}_h"):
                        st.info(x.get("chien_luoc", ""))
                    if S.get(f"{k}_s"):
                        st.success(x.get("loi_giai", ""))
    step_footer("solve", cur)


# ───────────────────────── ĐĂNG NHẬP & KHUNG CHÍNH ─────────────────────────
def brand():
    st.markdown(f'<div class="brand"><span class="logo">{BELL}</span><span>XSTK Tutor'
                '<small>Trợ giảng Xác suất thống kê</small></span></div>', unsafe_allow_html=True)


def login():
    brand()
    hero()
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid, st.container(border=True):
        sec("Bắt đầu học")
        with st.form("login"):
            name = st.text_input("Tên của bạn", max_chars=40, placeholder="Ví dụ: Nguyễn An")
            go = st.form_submit_button("Vào học", type="primary", icon=":material/login:")
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
                    assert isinstance(p, dict) and isinstance(p.get("events"), list) and str(p.get("name", "")).strip()
                except (ValueError, AssertionError, AttributeError, UnicodeDecodeError):
                    st.error("Tệp hồ sơ không hợp lệ.")
                else:
                    p["name"] = str(p["name"]).strip()[:40]
                    save_profile(p)
                    S.user, S.profile = p["name"], p
                    st.rerun()


def footer():
    src = _key_and_source()[1]
    disclaimer()
    st.markdown(
        '<div class="foot"><span>XSTK Tutor · Học khái niệm · Học định lí · Giải toán 4 bước</span>'
        f'<span>{"AI sẵn sàng (" + src + ")" if src else "AI chưa được cấu hình"}</span></div>',
        unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="XSTK Tutor · Trợ giảng Xác suất thống kê", page_icon=":material/bar_chart:",
                       layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    if not S.get("user"):
        login()
        footer()
        return
    a, d, b, c = st.columns([3.6, 1.7, 1.7, 1.4], vertical_alignment="center")
    with a:
        brand()
    if d.button("Máy tính Casio", icon=":material/calculate:", key="b_casio"):
        casio_dialog()
    if b.button("Hồ sơ học tập", icon=":material/insights:", type="primary", key="b_prof"):
        profile_dialog()
    with c.popover(S.user, icon=":material/account_circle:"):
        st.download_button("Tải hồ sơ (.json)", json.dumps(S.profile, ensure_ascii=False),
                           file_name="ho_so_xstk.json", mime="application/json", icon=":material/download:")
        if st.button("Đổi người dùng", key="b_out", icon=":material/logout:"):
            for k in list(S.keys()):
                del S[k]
            st.rerun()
    hero()
    if not _has_key():
        st.warning("Chưa cấu hình GEMINI_API_KEY (GitHub Secrets hoặc Streamlit Secrets): các tính năng AI tạm thời chưa hoạt động.")
    if S.get("wf_nav") not in WF:
        S.wf_nav = S.get("wf_last", "concept")
    st.segmented_control("Quy trình", list(WF), key="wf_nav", label_visibility="collapsed",
                         format_func=lambda k: f"{WF[k]['icon']} {WF[k]['name']}")
    wf = S.get("wf_nav") or S.get("wf_last", "concept")
    S.wf_last = wf
    left = max(0, USER_DAILY - shared().used_of(("u", S.user.lower())))
    st.caption(f"Lượt AI còn lại hôm nay của bạn: {left}/{USER_DAILY}. Nội dung trùng với người khác được dùng lại, không tốn lượt.")
    if wf == "solve":
        page_solve()
    else:
        page_learn(wf)
    footer()


main()
