# app.py
import os, random, base64, cv2
import streamlit as st
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# ==============================
# Page & Styles
# ==============================
st.set_page_config(page_title="Miromi AI Virtual Fitting Booth", layout="wide")
st.markdown("""
<style>
/* page spacing */
.block-container{padding-top:2.0rem; padding-bottom:2rem; max-width:1200px;}
/* step badge */
.badge{display:inline-flex;gap:.5rem;align-items:center;font-weight:800;
background:#eef2ff;color:#3730a3;border-radius:10px;padding:.35rem .6rem;margin:.2rem 0 .8rem;}
/* card grid */
.mi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}
.mi-card{border:2px solid transparent;border-radius:16px;padding:10px;background:rgba(255,255,255,.65);
box-shadow:0 2px 10px rgba(0,0,0,.05);transition:transform .12s ease, box-shadow .2s ease, border-color .2s ease;}
.mi-card:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(0,0,0,.10);border-color:#9ec5fe;}
.mi-card.selected{border-color:#4a7cff;box-shadow:0 0 0 3px rgba(74,124,255,.18) inset;}
.mi-card img{width:100%;height:auto;border-radius:12px;display:block;}
.mi-title{margin:.5rem 0 .25rem;font-weight:700}
.mi-sub{margin:0;color:#6b7280;font-size:.9rem}
.stButton > button{width:100%;border-radius:12px;padding:.55rem .8rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.title("👗 Miromi AI Virtual Fitting Booth")
st.caption("Developed by THE PLATFORM COMPANY")

# ==============================
# Paths & helpers
# ==============================
BASE_REFS = "refs"
UPLOADS = "uploads"
OUTPUTS = "outputs"
MODELS = "models"
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(BASE_REFS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def find_rep_thumb(outfit_dir: str):
    """대표 썸네일: ref_1.(jpg/png) 우선, 없으면 폴더 첫 이미지"""
    exts = (".jpg", ".jpeg", ".png", ".webp")
    # ref_1 우선
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join(BASE_REFS, outfit_dir, f"ref_1{ext}")
        if os.path.exists(p):
            return p
    # fallback: 폴더 첫 이미지
    folder = os.path.join(BASE_REFS, outfit_dir)
    imgs = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    imgs.sort()
    if imgs:
        return os.path.join(folder, imgs[0])
    return None

def list_ref_images(outfit_dir: str):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    folder = os.path.join(BASE_REFS, outfit_dir)
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def img_to_data_uri(path: str) -> str:
    if path is None or not os.path.exists(path): return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = "png" if path.lower().endswith(".png") else "jpeg"
    return f"data:image/{ext};base64,{b64}"

def pretty_name(dir_name: str) -> str:
    # outfit_1 -> 의상 1
    if dir_name.startswith("outfit_"):
        try:
            n = int(dir_name.split("_", 1)[1])
            return f"의상 {n}"
        except:
            return dir_name
    return dir_name

# ==============================
# Load models (cached)
# ==============================
@st.cache_resource
def load_models():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(os.path.join(MODELS, "inswapper_128.onnx"), download=False)
    return app, swapper

try:
    app, swapper = load_models()
except Exception as e:
    st.error("모델 로딩에 실패했어요. models/inswapper_128.onnx 파일이 있는지 확인해주세요.")
    st.exception(e)
    st.stop()

# ==============================
# Step 1: 의상 선택 (카드형 4열)
# ==============================
st.markdown('<div class="badge">1️⃣ 의상을 선택하세요</div>', unsafe_allow_html=True)

outfit_dirs = sorted([d for d in os.listdir(BASE_REFS) if os.path.isdir(os.path.join(BASE_REFS, d))])
if not outfit_dirs:
    st.warning("refs 폴더 안에 outfit_1, outfit_2, ... 형태의 폴더를 만들어 주세요.")
else:
    if "chosen_outfit" not in st.session_state:
        st.session_state["chosen_outfit"] = None

    # 카드 그리드 렌더
    st.markdown('<div class="mi-grid">', unsafe_allow_html=True)
    for outfit in outfit_dirs:
        thumb = find_rep_thumb(outfit)
        data_uri = img_to_data_uri(thumb)
        selected = (st.session_state["chosen_outfit"] == outfit)
        sel_class = " selected" if selected else ""
        title = pretty_name(outfit)

        # 카드(이미지+라벨)
        st.markdown(
            f"""
            <div class="mi-card{sel_class}">
                <img src="{data_uri}" alt="{title}">
                <div class="mi-title">{title}</div>
                <p class="mi-sub">대표 이미지: ref_1</p>
            """,
            unsafe_allow_html=True,
        )
        # 선택 버튼 (Streamlit 위젯은 HTML 밖에 있어야 하므로 닫기 전에 삽입)
        clicked = st.button(f"선택하기 ({title})", key=f"choose_{outfit}")
        st.markdown("</div>", unsafe_allow_html=True)  # .mi-card 닫기

        if clicked:
            st.session_state["chosen_outfit"] = outfit
    st.markdown("</div>", unsafe_allow_html=True)  # .mi-grid 닫기

# 선택 결과 표시 & 대표 이미지
if st.session_state.get("chosen_outfit"):
    chosen = st.session_state["chosen_outfit"]
    rep = find_rep_thumb(chosen)
    if rep:
        st.success(f"선택된 의상: {pretty_name(chosen)}")
        st.image(rep, caption="대표 의상 미리보기", use_container_width=True)

# ==============================
# Step 2: 얼굴 촬영
# ==============================
st.markdown('<div class="badge">2️⃣ 얼굴을 촬영해주세요</div>', unsafe_allow_html=True)
photo = st.camera_input("Camera")

# ==============================
# Step 3: 랜덤 레퍼런스 선택 (선택된 의상 폴더에서)
# ==============================
ref_path = None
if photo and st.session_state.get("chosen_outfit"):
    user_img = Image.open(photo)
    user_path = os.path.join(UPLOADS, "user.jpg")
    user_img.save(user_path)

    outfit = st.session_state["chosen_outfit"]
    ref_candidates = list_ref_images(outfit)
    if not ref_candidates:
        st.warning(f"{outfit} 폴더에 이미지(.jpg/.png)가 없습니다.")
    else:
        ref_path = random.choice(ref_candidates)
        st.image(ref_path, caption=f"랜덤 선택된 이미지 ({pretty_name(outfit)})", use_container_width=True)

# ==============================
# Step 4: 얼굴 스왑
# ==============================
if photo and st.session_state.get("chosen_outfit") and ref_path:
    if st.button("✨ 얼굴 스왑 실행"):
        src = cv2.imread(os.path.join(UPLOADS, "user.jpg"))   # BGR
        dst = cv2.imread(ref_path)

        src_faces = app.get(src)
        dst_faces = app.get(dst)

        if len(src_faces) == 0:
            st.error("사용자 얼굴을 인식하지 못했습니다. 다시 촬영해주세요.")
        elif len(dst_faces) == 0:
            st.error("의상 이미지에서 얼굴을 찾지 못했습니다.")
        else:
            result = swapper.get(dst, dst_faces[0], src_faces[0], paste_back=True)
            result_path = os.path.join(OUTPUTS, "result.jpg")
            cv2.imwrite(result_path, result)
            st.image(result_path, caption="✅ 얼굴 스왑 결과", use_container_width=True)

            # 프린트 & 다운로드
            cols = st.columns(2)
            with cols[0]:
                if st.button("🖨️ 프린트하기"):
                    try:
                        os.startfile(result_path, "print")
                        st.success("프린터로 전송했습니다.")
                    except Exception as e:
                        st.error("프린터 전송에 실패했어요. 윈도우 환경에서만 동작합니다.")
                        st.exception(e)
            with cols[1]:
                with open(result_path, "rb") as f:
                    st.download_button("⬇️ 결과 이미지 다운로드", f, file_name="miromi_result.jpg", mime="image/jpeg")

else:
    st.info("의상을 먼저 선택하고 얼굴을 촬영한 뒤 스왑을 실행하세요.")
