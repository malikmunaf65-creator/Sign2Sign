"""
Sign2Sign Web — Real-time gesture recognition
Supports: Live Webcam (WebSocket) + Image/Dataset Upload
Deploy on Render from existing Sign2Sign repo
"""
import os, base64, json, io
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── MODEL LOAD ────────────────────────────────────────────────
MODEL_OK = False
model    = None
labels   = []

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    model    = tf.keras.models.load_model("gesture_model.keras")
    MODEL_OK = True
    print("✅ gesture_model.keras loaded")
except Exception as e:
    print(f"❌ Model load failed: {e}")

try:
    labels = list(np.load("label_map.npy", allow_pickle=True))
    print(f"✅ Labels: {labels}")
except Exception as e:
    print(f"⚠️ label_map.npy not found: {e}")
    labels = []

# ── DIRS ──────────────────────────────────────────────────────
os.makedirs("collected_gestures", exist_ok=True)
os.makedirs("gesture_data",       exist_ok=True)

app = FastAPI(title="Sign2Sign Web")

# Mount gesture_data if it exists
if os.path.exists("gesture_data"):
    app.mount("/gesture_data", StaticFiles(directory="gesture_data"), name="gesture_data")

# ── PREDICT HELPER ────────────────────────────────────────────
def predict_image(img_array: np.ndarray):
    """img_array: H×W×3 uint8 BGR or RGB — returns (gesture, conf%, top3)"""
    if not MODEL_OK or not labels:
        return "No Model", 0.0, []
    try:
        roi = cv2.resize(img_array, (64, 64)).astype(np.float32) / 255.0
        x   = np.expand_dims(roi, axis=0)
        preds      = model.predict(x, verbose=0)[0]
        top_idx    = preds.argsort()[-3:][::-1]
        top3       = [(labels[i] if i < len(labels) else "?",
                       round(float(preds[i]) * 100, 1)) for i in top_idx]
        return top3[0][0], top3[0][1], top3
    except Exception as e:
        print(f"Predict error: {e}")
        return "Error", 0.0, []

def extract_roi(frame: np.ndarray) -> np.ndarray:
    """Extract center square ROI from frame."""
    h, w = frame.shape[:2]
    s    = min(h, w) // 2
    x1, y1 = (w - s) // 2, (h - s) // 2
    return frame[y1:y1+s, x1:x1+s]

# ── WEBSOCKET ─────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    collect_count = {}
    try:
        while True:
            raw  = await ws.receive_text()
            msg  = json.loads(raw)
            b64  = msg.get("data", "").split(",")[-1]
            arr  = np.frombuffer(base64.b64decode(b64), np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if msg.get("type") == "predict":
                roi = extract_roi(frame)
                gesture, conf, top3 = predict_image(roi)
                await ws.send_text(json.dumps({
                    "type": "prediction",
                    "gesture": gesture,
                    "confidence": conf,
                    "top3": [{"label": g, "conf": c} for g, c in top3],
                }))

            elif msg.get("type") == "collect":
                name = msg.get("name", "").upper().strip()
                if not name:
                    continue
                roi = extract_roi(frame)
                roi_resized = cv2.resize(roi, (64, 64))
                save_dir = os.path.join("gesture_data", name)
                os.makedirs(save_dir, exist_ok=True)
                existing = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
                np.save(os.path.join(save_dir, f"{existing}.npy"), roi_resized)
                count = existing + 1
                collect_count[name] = count
                await ws.send_text(json.dumps({
                    "type": "collect_ack",
                    "name": name,
                    "count": count,
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")

# ── IMAGE UPLOAD PREDICT ──────────────────────────────────────
@app.post("/predict-image")
async def predict_image_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        arr = np.array(img)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gesture, conf, top3 = predict_image(arr_bgr)
        return JSONResponse({
            "gesture": gesture,
            "confidence": conf,
            "top3": [{"label": g, "conf": c} for g, c in top3],
            "model_ok": MODEL_OK,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── STATUS ────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    return {"model_ok": MODEL_OK, "labels": labels, "count": len(labels)}

# ── HOME ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# ═══════════════════════════════════════════════════════════════
# FULL FRONTEND
# ═══════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign2Sign — Gesture Recognition</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --em:#00e5a0;--em2:#00b87a;--amb:#f59e0b;--rose:#ff6b6b;--vi:#a855f7;
  --bg:#060d0a;--bg2:#0a1410;--bg3:#0f1d18;
  --glass:rgba(0,229,160,.05);--border:rgba(0,229,160,.14);
  --border2:rgba(245,158,11,.18);--text:rgba(255,255,255,.88);--muted:rgba(255,255,255,.3);
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;overflow-x:hidden;cursor:none}
#cd{position:fixed;z-index:9999;pointer-events:none;width:8px;height:8px;border-radius:50%;background:var(--em);transform:translate(-50%,-50%)}
#cr{position:fixed;z-index:9998;pointer-events:none;width:34px;height:34px;border-radius:50%;border:1px solid rgba(0,229,160,.4);transform:translate(-50%,-50%);transition:width .3s,height .3s}
#bgc{position:fixed;inset:0;z-index:0;pointer-events:none}
.sl{position:fixed;inset:0;z-index:1;pointer-events:none;background:repeating-linear-gradient(0deg,rgba(0,229,160,.01) 0,rgba(0,229,160,.01) 1px,transparent 1px,transparent 3px)}
.nb{position:fixed;border-radius:50%;pointer-events:none;filter:blur(80px);z-index:0}

/* NAV */
nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:14px 40px;display:flex;align-items:center;justify-content:space-between;backdrop-filter:blur(28px);background:rgba(6,13,10,.65);border-bottom:1px solid rgba(0,229,160,.07)}
.logo{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:800;letter-spacing:.12em;background:linear-gradient(135deg,var(--em),var(--amb));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-links{display:flex;gap:6px}
.nl{padding:6px 16px;border-radius:100px;font-size:.68rem;font-weight:600;letter-spacing:.16em;color:var(--muted);cursor:pointer;border:1px solid transparent;background:none;transition:all .2s;font-family:'Space Grotesk',sans-serif}
.nl:hover,.nl.active{color:var(--em);border-color:rgba(0,229,160,.25);background:rgba(0,229,160,.06)}
.nbadge{display:flex;align-items:center;gap:7px;padding:6px 16px;border-radius:100px;border:1px solid var(--border);background:var(--glass);font-size:.68rem;font-weight:600;color:var(--em);letter-spacing:.08em}
.bd{width:6px;height:6px;border-radius:50%;background:var(--em);box-shadow:0 0 6px var(--em);animation:blink 1.4s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

/* PAGES */
.page{display:none;min-height:100vh;padding:90px 24px 60px;position:relative;z-index:10}
.page.active{display:flex;flex-direction:column;align-items:center}

/* ── HERO ── */
.hero-tag{font-size:.62rem;font-weight:600;letter-spacing:.44em;color:rgba(0,229,160,.45);display:flex;align-items:center;gap:12px;margin-bottom:22px}
.hero-tag::before,.hero-tag::after{content:'';display:block;height:1px;width:32px;background:rgba(0,229,160,.22)}
.h1{font-family:'Syne',sans-serif;font-size:clamp(3rem,8vw,6.5rem);font-weight:800;line-height:.9;text-align:center;letter-spacing:-.02em;margin-bottom:22px}
.h1 span:nth-child(1){display:block;background:linear-gradient(180deg,#fff,rgba(255,255,255,.5));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.h1 span:nth-child(2){display:block;background:linear-gradient(135deg,var(--em),var(--amb));-webkit-background-clip:text;-webkit-text-fill-color:transparent;filter:drop-shadow(0 0 26px rgba(0,229,160,.3))}
.h1 span:nth-child(3){display:block;font-family:'Space Grotesk',sans-serif;font-size:.38em;letter-spacing:.22em;color:rgba(255,255,255,.16);font-weight:400;margin-top:10px}
.hero-p{font-size:1rem;color:var(--muted);line-height:1.75;max-width:460px;text-align:center;margin-bottom:34px}
.hero-btns{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;margin-bottom:48px}
.btn-p{padding:14px 36px;border-radius:100px;border:none;cursor:none;font-family:'Syne',sans-serif;font-size:.76rem;font-weight:700;letter-spacing:.16em;background:linear-gradient(135deg,var(--em),var(--amb));color:#060d0a;box-shadow:0 0 26px rgba(0,229,160,.28);transition:transform .25s,box-shadow .25s}
.btn-p:hover{transform:translateY(-3px) scale(1.03);box-shadow:0 0 48px rgba(0,229,160,.48)}
.btn-g{padding:13px 30px;border-radius:100px;border:1px solid rgba(0,229,160,.24);font-family:'Space Grotesk',sans-serif;font-size:.82rem;font-weight:500;letter-spacing:.1em;color:rgba(0,229,160,.7);background:none;cursor:none;transition:all .25s}
.btn-g:hover{background:rgba(0,229,160,.06);border-color:var(--em);color:var(--em)}

/* STAT GRID */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;width:100%;max-width:640px;margin-bottom:16px}
.sc{padding:18px 14px;text-align:center;background:var(--glass);border:1px solid rgba(255,255,255,.04)}
.sc:first-child{border-radius:12px 0 0 12px}.sc:last-child{border-radius:0 12px 12px 0}
.sn{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,var(--em),var(--amb));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sl2{font-size:.6rem;font-weight:500;letter-spacing:.2em;color:var(--muted);margin-top:3px}

/* ── MODE CARDS (home) ── */
.mode-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;width:100%;max-width:720px}
.mode-card{padding:30px 26px;border-radius:20px;background:var(--glass);border:1px solid var(--border);cursor:pointer;transition:all .3s;position:relative;overflow:hidden}
.mode-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--em),transparent);opacity:0;transition:opacity .3s}
.mode-card:hover{transform:translateY(-6px);border-color:rgba(0,229,160,.35);background:rgba(0,229,160,.06)}
.mode-card:hover::before{opacity:1}
.mode-card.amber{border-color:var(--border2)}
.mode-card.amber::before{background:linear-gradient(90deg,transparent,var(--amb),transparent)}
.mode-card.amber:hover{border-color:rgba(245,158,11,.4);background:rgba(245,158,11,.05)}
.mc-ico{font-size:2.2rem;margin-bottom:14px;display:block}
.mc-t{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:white;margin-bottom:8px}
.mc-d{font-size:.82rem;line-height:1.65;color:var(--muted)}
.mc-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:100px;background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.14);font-size:.65rem;font-weight:600;letter-spacing:.12em;color:var(--em);margin-top:14px}

/* ── LIVE CAM PAGE ── */
.live-wrap{display:grid;grid-template-columns:1fr 340px;gap:20px;width:100%;max-width:960px}
.cam-box{border-radius:18px;overflow:hidden;background:#000;border:1px solid var(--border);position:relative}
#video{width:100%;display:block;transform:scaleX(-1)}
#ov{position:absolute;inset:0;pointer-events:none}
.cam-hud{position:absolute;top:12px;left:12px;display:flex;align-items:center;gap:7px;font-size:.62rem;font-weight:600;letter-spacing:.16em;color:rgba(0,229,160,.7)}
.cam-hud-d{width:5px;height:5px;border-radius:50%;background:var(--em);box-shadow:0 0 5px var(--em);animation:blink 1.4s infinite}
.cam-guide{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);font-size:.62rem;letter-spacing:.18em;color:rgba(0,229,160,.4);background:rgba(0,0,0,.55);padding:4px 14px;border-radius:100px;white-space:nowrap}
.right-panel{display:flex;flex-direction:column;gap:14px}
.rc{padding:20px 18px;border-radius:16px;background:var(--glass);border:1px solid var(--border)}
.rc.amb{background:rgba(245,158,11,.04);border-color:var(--border2)}
.rc-eye{font-size:.58rem;font-weight:600;letter-spacing:.28em;color:var(--muted);display:block;margin-bottom:12px}
.g-big{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,var(--em),var(--amb));-webkit-background-clip:text;-webkit-text-fill-color:transparent;filter:drop-shadow(0 0 12px rgba(0,229,160,.25));line-height:1;transition:all .15s}
.cbar-track{height:5px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden;margin:8px 0 4px}
.cbar-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--em),var(--amb));transition:width .25s}
.cnum{font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;color:var(--amb)}
.t3r{display:flex;align-items:center;gap:9px;margin-bottom:9px}
.t3l{font-size:.74rem;font-weight:500;color:rgba(255,255,255,.45);width:68px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.t3t{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.t3f{height:100%;border-radius:2px;transition:width .25s}
.t3p{font-size:.68rem;color:var(--muted);width:36px;text-align:right}
.hist{max-height:100px;overflow-y:auto}
.hist::-webkit-scrollbar{width:2px}
.hist::-webkit-scrollbar-thumb{background:rgba(0,229,160,.2);border-radius:1px}
.htag{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:100px;background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.1);font-size:.66rem;color:rgba(255,255,255,.38);margin:2px}
.cam-ctrls{display:flex;gap:10px;margin-top:12px}
.cc{flex:1;padding:12px;border-radius:10px;border:none;cursor:none;font-family:'Syne',sans-serif;font-size:.68rem;font-weight:700;letter-spacing:.14em;transition:all .25s}
.cc-start{background:linear-gradient(135deg,var(--em),var(--em2));color:#060d0a;box-shadow:0 5px 20px rgba(0,229,160,.18)}
.cc-start:hover{transform:translateY(-2px);box-shadow:0 10px 32px rgba(0,229,160,.32)}
.cc-stop{background:rgba(255,107,107,.1);border:1px solid rgba(255,107,107,.22);color:var(--rose)}
.cc-stop:hover{background:rgba(255,107,107,.18)}

/* ── UPLOAD PAGE ── */
.upload-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px;width:100%;max-width:900px}
.drop-card{padding:36px 28px;border-radius:20px;background:var(--glass);border:2px dashed rgba(0,229,160,.2);text-align:center;transition:all .3s;cursor:pointer}
.drop-card.drag-over,.drop-card:hover{border-color:rgba(0,229,160,.55);background:rgba(0,229,160,.04)}
#fileInput{display:none}
.drop-ico{font-size:2.4rem;margin-bottom:14px;display:block}
.drop-t{font-family:'Syne',sans-serif;font-size:.88rem;font-weight:700;color:rgba(255,255,255,.65);margin-bottom:6px}
.drop-h{font-size:.78rem;color:rgba(255,255,255,.22)}
.drop-h span{color:var(--em)}
.img-preview{width:100%;border-radius:12px;overflow:hidden;display:none;margin-top:16px}
.img-preview img{width:100%;border-radius:12px;display:block}
.result-card{padding:28px 24px;border-radius:20px;background:var(--glass);border:1px solid var(--border)}
.result-card.amb{background:rgba(245,158,11,.04);border-color:var(--border2)}
.res-big{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,var(--em),var(--amb));-webkit-background-clip:text;-webkit-text-fill-color:transparent;filter:drop-shadow(0 0 14px rgba(0,229,160,.28));line-height:1;margin-bottom:16px}
.res-placeholder{font-size:.9rem;color:var(--muted);text-align:center;padding:30px 0}
.upload-btn{width:100%;padding:14px;border-radius:12px;border:none;cursor:pointer;font-family:'Syne',sans-serif;font-size:.76rem;font-weight:700;letter-spacing:.16em;background:linear-gradient(135deg,var(--em),var(--amb));color:#060d0a;box-shadow:0 6px 22px rgba(0,229,160,.2);transition:all .25s;margin-top:14px}
.upload-btn:hover{transform:translateY(-2px);box-shadow:0 12px 32px rgba(0,229,160,.32)}
.upload-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}

/* ── ADD GESTURE PAGE ── */
.add-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px;width:100%;max-width:860px}
.add-card{padding:32px 26px;border-radius:20px;background:var(--glass);border:1px solid var(--border)}
.add-card.amb{background:rgba(245,158,11,.04);border-color:var(--border2)}
.inp-lbl{font-size:.6rem;font-weight:600;letter-spacing:.26em;color:var(--muted);display:block;margin-bottom:8px}
.inp{width:100%;padding:12px 15px;border-radius:11px;background:rgba(0,0,0,.5);border:1px solid rgba(0,229,160,.14);color:white;font-family:'Space Grotesk',sans-serif;font-size:.95rem;outline:none;transition:border-color .25s;margin-bottom:14px}
.inp:focus{border-color:rgba(0,229,160,.48)}
.inp::placeholder{color:rgba(255,255,255,.18)}
.prog-wrap{margin-bottom:14px;display:none}
.prog-row{display:flex;justify-content:space-between;font-size:.68rem;color:var(--muted);margin-bottom:7px}
.pt{height:5px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden}
.pf{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--em),var(--amb));transition:width .25s}
.coll-btn{width:100%;padding:13px;border-radius:11px;border:none;cursor:none;font-family:'Syne',sans-serif;font-size:.74rem;font-weight:700;letter-spacing:.16em;background:linear-gradient(135deg,var(--em),var(--amb));color:#060d0a;box-shadow:0 5px 20px rgba(0,229,160,.18);transition:all .25s}
.coll-btn:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(0,229,160,.3)}
.coll-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.hint{padding:12px 14px;border-radius:10px;background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.13);font-size:.78rem;color:rgba(245,158,11,.7);line-height:1.65;margin-top:14px}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:100px;background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.13);font-size:.68rem;color:var(--em);font-weight:500}
.chip-d{width:5px;height:5px;border-radius:50%;background:var(--em)}

/* SECTION LABELS */
.sec-eye{font-size:.62rem;font-weight:600;letter-spacing:.4em;color:rgba(0,229,160,.4);margin-bottom:8px;text-align:center}
.sec-h{font-family:'Syne',sans-serif;font-size:clamp(1.6rem,3.5vw,2.6rem);font-weight:800;text-align:center;background:linear-gradient(135deg,#fff,rgba(255,255,255,.4));-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;margin-bottom:28px}

/* REVEAL */
.rv{opacity:0;transform:translateY(36px);transition:opacity .8s cubic-bezier(.22,.68,0,1),transform .8s cubic-bezier(.22,.68,0,1)}
.rv.d1{transition-delay:.1s}.rv.d2{transition-delay:.2s}.rv.d3{transition-delay:.3s}
.rv.on{opacity:1;transform:translateY(0)}

/* LOADER */
.ld{display:inline-flex;gap:4px}
.ld span{width:6px;height:6px;border-radius:50%;background:var(--em);animation:ldot .8s ease-in-out infinite}
.ld span:nth-child(2){animation-delay:.16s}.ld span:nth-child(3){animation-delay:.32s}
@keyframes ldot{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}

footer{position:relative;z-index:10;padding:36px 40px;border-top:1px solid rgba(255,255,255,.04);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px}
.fl{font-family:'Syne',sans-serif;font-size:.82rem;font-weight:800;letter-spacing:.12em;background:linear-gradient(135deg,var(--em),var(--amb));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.fc{font-size:.7rem;color:rgba(255,255,255,.18)}
@media(max-width:720px){.live-wrap,.upload-wrap,.add-wrap,.mode-grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}nav{padding:12px 16px}.nl{display:none}footer{padding:24px 20px}}
</style>
</head>
<body>
<div id="cd"></div><div id="cr"></div>
<canvas id="bgc"></canvas>
<div class="sl"></div>
<div class="nb" style="width:520px;height:360px;left:-70px;top:8%;background:radial-gradient(circle,rgba(0,229,160,.055),transparent 70%)"></div>
<div class="nb" style="width:580px;height:420px;right:-90px;top:40%;background:radial-gradient(circle,rgba(245,158,11,.05),transparent 70%)"></div>
<div class="nb" style="width:420px;height:420px;left:24%;bottom:3%;background:radial-gradient(circle,rgba(168,85,247,.04),transparent 70%)"></div>

<nav>
  <span class="logo">SIGN·2·SIGN</span>
  <div class="nav-links">
    <button class="nl active" onclick="go('home')">Home</button>
    <button class="nl" onclick="go('live')">Live Cam</button>
    <button class="nl" onclick="go('upload')">Upload</button>
    <button class="nl" onclick="go('add')">Add Gesture</button>
  </div>
  <div class="nbadge"><span class="bd"></span><span id="mstatus">LOADING...</span></div>
</nav>

<!-- ═══ HOME ═══ -->
<div class="page active" id="p-home">
  <div class="hero-tag">Real-Time Gesture Intelligence</div>
  <h1 class="h1">
    <span>SPEAK</span>
    <span>WITH HANDS</span>
    <span>Bridging Silence Into Connection</span>
  </h1>
  <p class="hero-p">Live webcam gesture recognition AND image upload prediction — powered by deep learning, built for the deaf and mute community.</p>
  <div class="hero-btns">
    <button class="btn-p" onclick="go('live')">🎥 Live Camera</button>
    <button class="btn-g" onclick="go('upload')">📁 Upload Image</button>
  </div>

  <div class="stats rv">
    <div class="sc"><div class="sn" id="st-count">—</div><div class="sl2">GESTURES</div></div>
    <div class="sc"><div class="sn">5</div><div class="sl2">FPS LIVE</div></div>
    <div class="sc"><div class="sn">64px</div><div class="sl2">ROI SIZE</div></div>
    <div class="sc"><div class="sn">100</div><div class="sl2">SAMPLES</div></div>
  </div>

  <div style="height:32px"></div>
  <div class="sec-eye rv d1">CHOOSE YOUR MODE</div>
  <div class="mode-grid rv d2">
    <div class="mode-card" onclick="go('live')">
      <span class="mc-ico">🎥</span>
      <div class="mc-t">Live Webcam</div>
      <p class="mc-d">Real-time gesture recognition using your webcam. Get instant predictions with confidence scores as you sign.</p>
      <div class="mc-badge"><span class="bd"></span>REAL-TIME WEBSOCKET</div>
    </div>
    <div class="mode-card amber" onclick="go('upload')">
      <span class="mc-ico">📁</span>
      <div class="mc-t">Upload Image</div>
      <p class="mc-d">Upload any image or photo from your dataset. Our model will predict the gesture with top-3 confidence breakdown.</p>
      <div class="mc-badge" style="background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.2);color:var(--amb)">DATASET PREDICTION</div>
    </div>
  </div>
</div>

<!-- ═══ LIVE CAM ═══ -->
<div class="page" id="p-live">
  <div class="sec-eye rv">LIVE RECOGNITION</div>
  <h2 class="sec-h rv d1" style="margin-bottom:20px">Point & Predict</h2>
  <div class="live-wrap">
    <div>
      <div class="cam-box rv">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="ov"></canvas>
        <div class="cam-hud"><span class="cam-hud-d"></span><span id="cs">CAMERA OFF</span></div>
        <div class="cam-guide" id="cg">Click START to begin</div>
      </div>
      <div class="cam-ctrls" style="margin-top:12px">
        <button class="cc cc-start" id="bStart" onclick="startCam()">▶ START</button>
        <button class="cc cc-stop" id="bStop" onclick="stopCam()" disabled>⏸ STOP</button>
        <button class="cc" style="background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.22);color:var(--vi)" onclick="go('add')">➕ ADD GESTURE</button>
      </div>
    </div>
    <div class="right-panel">
      <div class="rc">
        <span class="rc-eye">DETECTED GESTURE</span>
        <div class="g-big" id="gDisp">—</div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px">
          <span style="font-size:.68rem;color:var(--muted)">Confidence</span>
          <span class="cnum" id="cNum">0%</span>
        </div>
        <div class="cbar-track"><div class="cbar-fill" id="cBar" style="width:0%"></div></div>
      </div>
      <div class="rc amb">
        <span class="rc-eye">TOP PREDICTIONS</span>
        <div id="t3">
          <div class="t3r"><span class="t3l" style="color:var(--muted)">—</span><div class="t3t"><div class="t3f" style="width:0%;background:var(--em)"></div></div><span class="t3p">0%</span></div>
          <div class="t3r"><span class="t3l" style="color:var(--muted)">—</span><div class="t3t"><div class="t3f" style="width:0%;background:var(--amb)"></div></div><span class="t3p">0%</span></div>
          <div class="t3r"><span class="t3l" style="color:var(--muted)">—</span><div class="t3t"><div class="t3f" style="width:0%;background:var(--rose)"></div></div><span class="t3p">0%</span></div>
        </div>
      </div>
      <div class="rc">
        <span class="rc-eye">GESTURE HISTORY</span>
        <div class="hist" id="hist"><span class="htag">No predictions yet</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ UPLOAD ═══ -->
<div class="page" id="p-upload">
  <div class="sec-eye rv">DATASET PREDICTION</div>
  <h2 class="sec-h rv d1" style="margin-bottom:20px">Upload & Predict</h2>
  <div class="upload-wrap">
    <div>
      <div class="drop-card rv" id="dropCard" onclick="document.getElementById('fileInput').click()" ondragover="ev.preventDefault();this.classList.add('drag-over')" ondragleave="this.classList.remove('drag-over')" ondrop="handleDrop(event)">
        <input type="file" id="fileInput" accept="image/*" onchange="previewFile(this)">
        <span class="drop-ico">🖼️</span>
        <div class="drop-t">Drop image here</div>
        <div class="drop-h">or <span>browse files</span> · JPG / PNG / JPEG</div>
        <div class="img-preview" id="imgPrev"><img id="prevImg" src="" alt="preview"></div>
      </div>
      <button class="upload-btn" id="upBtn" onclick="uploadPredict()" disabled>⚡ PREDICT GESTURE</button>
    </div>
    <div>
      <div class="result-card rv d2" id="resCard">
        <div class="res-placeholder" id="resPlaceholder">Upload an image to see the prediction</div>
        <div id="resContent" style="display:none">
          <span class="rc-eye">DETECTED GESTURE</span>
          <div class="res-big" id="resGesture">—</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span style="font-size:.72rem;color:var(--muted)">Confidence</span>
            <span class="cnum" id="resConf">0%</span>
          </div>
          <div class="cbar-track"><div class="cbar-fill" id="resBar" style="width:0%"></div></div>
          <div style="margin-top:18px">
            <span class="rc-eye" style="display:block;margin-bottom:10px">TOP 3 PREDICTIONS</span>
            <div id="resT3"></div>
          </div>
        </div>
      </div>
      <div class="result-card amb rv d3" style="margin-top:14px;padding:18px 20px">
        <span class="rc-eye">SUPPORTED GESTURES</span>
        <div class="chips" id="labelsChips"><span style="font-size:.8rem;color:var(--muted)">Loading...</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ ADD GESTURE ═══ -->
<div class="page" id="p-add">
  <div class="sec-eye rv">TRAIN NEW SIGN</div>
  <h2 class="sec-h rv d1" style="margin-bottom:20px">Add Your Gesture</h2>
  <div class="add-wrap">
    <div class="add-card rv">
      <label class="inp-lbl">GESTURE NAME</label>
      <input class="inp" id="gName" placeholder="e.g. HELLO, YES, ROCK..." oninput="onNameInput()">
      <div class="prog-wrap" id="pw">
        <div class="prog-row"><span id="pLabel">Collecting...</span><span id="pCount">0 / 100</span></div>
        <div class="pt"><div class="pf" id="pf" style="width:0%"></div></div>
      </div>
      <button class="coll-btn" id="cBtn" onclick="startCollect()" disabled>📸 START COLLECTION</button>
      <div class="hint">
        ⚡ After collecting, re-train by running:<br>
        <strong style="color:var(--em)">python improved_train_model.py</strong><br>
        Then upload the new <strong>gesture_model.keras</strong> + <strong>label_map.npy</strong> to GitHub and redeploy.
      </div>
    </div>
    <div class="add-card amb rv d2">
      <label class="inp-lbl">TRAINED GESTURES</label>
      <div class="chips" id="gChips"><span style="font-size:.8rem;color:var(--muted)">Loading...</span></div>
      <div style="margin-top:22px;padding-top:18px;border-top:1px solid rgba(245,158,11,.1)">
        <label class="inp-lbl">COLLECTION STATUS</label>
        <div id="cStatus" style="font-size:.8rem;color:var(--muted)">Ready to collect samples.</div>
      </div>
    </div>
  </div>
</div>

<footer>
  <span class="fl">SIGN·2·SIGN</span>
  <span class="fc">Real-Time Gesture Recognition &nbsp;·&nbsp; FastAPI + TensorFlow + WebSocket</span>
</footer>

<script>
// ── BG STARS ─────────────────────────────────────────────────
const bgc=document.getElementById('bgc');
const bgx=bgc.getContext('2d');
function rb(){bgc.width=innerWidth;bgc.height=innerHeight}rb();
addEventListener('resize',rb);
const ST=Array.from({length:200},()=>({x:Math.random()*innerWidth,y:Math.random()*innerHeight,r:Math.random()*1.2+.2,a:Math.random(),da:(Math.random()*.003+.001)*(Math.random()<.5?1:-1)}));
function ds(){bgx.clearRect(0,0,bgc.width,bgc.height);ST.forEach(s=>{s.a=Math.max(.04,Math.min(1,s.a+s.da));if(s.a<=.04||s.a>=1)s.da*=-1;bgx.beginPath();bgx.arc(s.x,s.y,s.r,0,Math.PI*2);bgx.fillStyle=`rgba(200,255,225,${s.a.toFixed(2)})`;bgx.fill()});requestAnimationFrame(ds)}ds();

// ── CURSOR ────────────────────────────────────────────────────
const cd=document.getElementById('cd'),cr=document.getElementById('cr');
let mx=0,my=0,rx=0,ry=0;
document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY});
(function ac(){cd.style.left=mx+'px';cd.style.top=my+'px';rx+=(mx-rx)*.14;ry+=(my-ry)*.14;cr.style.left=rx+'px';cr.style.top=ry+'px';requestAnimationFrame(ac)})();

// ── PAGES ─────────────────────────────────────────────────────
const PAGES=['home','live','upload','add'];
function go(id){
  PAGES.forEach(p=>{
    document.getElementById('p-'+p).classList.remove('active');
    document.querySelectorAll('.nl')[PAGES.indexOf(p)].classList.remove('active');
  });
  document.getElementById('p-'+id).classList.add('active');
  document.querySelectorAll('.nl')[PAGES.indexOf(id)].classList.add('active');
  window.scrollTo({top:0,behavior:'smooth'});
  if(id==='live'&&!stream) {}
  if(id==='upload'||id==='add') loadStatus();
}

// ── REVEALS ───────────────────────────────────────────────────
const obs=new IntersectionObserver(e=>e.forEach(en=>{if(en.isIntersecting){en.target.classList.add('on');obs.unobserve(en.target)}}),{threshold:.1});
document.querySelectorAll('.rv').forEach(el=>obs.observe(el));

// ── STATUS ────────────────────────────────────────────────────
let allLabels=[];
async function loadStatus(){
  try{
    const d=await(await fetch('/api/status')).json();
    document.getElementById('mstatus').textContent=d.model_ok?'MODEL READY':'NO MODEL';
    document.getElementById('st-count').textContent=d.count||'0';
    allLabels=d.labels||[];
    renderChips(allLabels,'gChips');
    renderChips(allLabels,'labelsChips');
  }catch(e){document.getElementById('mstatus').textContent='OFFLINE'}
}
function renderChips(lbs,id){
  const c=document.getElementById(id);
  if(!lbs.length){c.innerHTML='<span style="font-size:.8rem;color:var(--muted)">No gestures yet.</span>';return}
  c.innerHTML=lbs.map(l=>`<span class="chip"><span class="chip-d"></span>${l}</span>`).join('');
}
loadStatus();

// ── WEBSOCKET ─────────────────────────────────────────────────
let ws=null;
function connectWS(){
  const pr=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${pr}://${location.host}/ws`);
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.type==='prediction') showPred(d);
    if(d.type==='collect_ack') onCollectAck(d);
  };
  ws.onclose=()=>{ws=null;setTimeout(connectWS,2500)};
  ws.onerror=()=>ws?.close();
}
connectWS();

// ── LIVE CAM ──────────────────────────────────────────────────
let stream=null,predT=null,sc=null,scx=null;
const vid=document.getElementById('video'),ov=document.getElementById('ov'),ovx=ov.getContext('2d');
const hist=[];

async function startCam(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
    vid.srcObject=stream;
    await new Promise(r=>vid.onloadedmetadata=r);
    ov.width=vid.videoWidth;ov.height=vid.videoHeight;
    sc=document.createElement('canvas');sc.width=vid.videoWidth;sc.height=vid.videoHeight;scx=sc.getContext('2d');
    document.getElementById('bStart').disabled=true;
    document.getElementById('bStop').disabled=false;
    document.getElementById('cs').textContent='LIVE';
    document.getElementById('cg').textContent='Place hand in center frame';
    drawROI();
    predT=setInterval(sendFrame,200);
  }catch(e){alert('Camera access denied. Please allow camera permissions and try again.')}
}
function stopCam(){
  clearInterval(predT);predT=null;
  stream?.getTracks().forEach(t=>t.stop());stream=null;vid.srcObject=null;
  document.getElementById('bStart').disabled=false;
  document.getElementById('bStop').disabled=true;
  document.getElementById('cs').textContent='CAMERA OFF';
  document.getElementById('cg').textContent='Click START to begin';
  ovx.clearRect(0,0,ov.width,ov.height);
}
function drawROI(){
  if(!stream)return;
  ovx.clearRect(0,0,ov.width,ov.height);
  const w=ov.width,h=ov.height,s=Math.min(w,h)*.5;
  const x=(w-s)/2,y=(h-s)/2,cs=18;
  ovx.strokeStyle='rgba(0,229,160,.35)';ovx.lineWidth=1.5;ovx.strokeRect(x,y,s,s);
  ovx.strokeStyle='rgba(0,229,160,.9)';ovx.lineWidth=2.5;
  [[x,y,1,1],[x+s,y,-1,1],[x,y+s,1,-1],[x+s,y+s,-1,-1]].forEach(([px,py,sx2,sy])=>{
    ovx.beginPath();ovx.moveTo(px,py+sy*cs);ovx.lineTo(px,py);ovx.lineTo(px+sx2*cs,py);ovx.stroke();
  });
  requestAnimationFrame(drawROI);
}
function sendFrame(){
  if(!ws||ws.readyState!==1||!stream)return;
  scx.drawImage(vid,0,0);
  ws.send(JSON.stringify({type:'predict',data:sc.toDataURL('image/jpeg',.7)}));
}
function showPred(d){
  document.getElementById('gDisp').textContent=d.gesture||'—';
  const c=d.confidence||0;
  document.getElementById('cNum').textContent=c.toFixed(1)+'%';
  document.getElementById('cBar').style.width=c+'%';
  const rows=document.querySelectorAll('#t3 .t3r');
  const cols=['var(--em)','var(--amb)','var(--rose)'];
  (d.top3||[]).forEach((it,i)=>{
    if(!rows[i])return;
    rows[i].querySelector('.t3l').textContent=it.label;
    rows[i].querySelector('.t3f').style.width=it.conf+'%';
    rows[i].querySelector('.t3f').style.background=cols[i];
    rows[i].querySelector('.t3p').textContent=it.conf.toFixed(1)+'%';
  });
  if(d.gesture&&d.confidence>35){
    hist.unshift(`${d.gesture} (${c.toFixed(1)}%)`);
    if(hist.length>12)hist.pop();
    document.getElementById('hist').innerHTML=hist.map(h=>`<span class="htag">${h}</span>`).join('');
  }
}

// ── IMAGE UPLOAD ──────────────────────────────────────────────
let selectedFile=null;
function previewFile(input){
  if(!input.files[0])return;
  selectedFile=input.files[0];
  const reader=new FileReader();
  reader.onload=e=>{
    document.getElementById('prevImg').src=e.target.result;
    document.getElementById('imgPrev').style.display='block';
    document.getElementById('upBtn').disabled=false;
  };
  reader.readAsDataURL(selectedFile);
}
function handleDrop(e){
  e.preventDefault();
  document.getElementById('dropCard').classList.remove('drag-over');
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('image/')){
    selectedFile=f;
    const reader=new FileReader();
    reader.onload=ev=>{
      document.getElementById('prevImg').src=ev.target.result;
      document.getElementById('imgPrev').style.display='block';
      document.getElementById('upBtn').disabled=false;
    };
    reader.readAsDataURL(f);
  }
}
async function uploadPredict(){
  if(!selectedFile)return;
  const btn=document.getElementById('upBtn');
  btn.innerHTML='<span class="ld"><span></span><span></span><span></span></span> PREDICTING...';
  btn.disabled=true;
  const fd=new FormData();fd.append('file',selectedFile);
  try{
    const res=await fetch('/predict-image',{method:'POST',body:fd});
    const d=await res.json();
    document.getElementById('resPlaceholder').style.display='none';
    document.getElementById('resContent').style.display='block';
    document.getElementById('resGesture').textContent=d.gesture||'Unknown';
    const c=d.confidence||0;
    document.getElementById('resConf').textContent=c.toFixed(1)+'%';
    document.getElementById('resBar').style.width=c+'%';
    const cols=['var(--em)','var(--amb)','var(--rose)'];
    document.getElementById('resT3').innerHTML=(d.top3||[]).map((it,i)=>`
      <div class="t3r">
        <span class="t3l" style="color:${cols[i]}">${it.label}</span>
        <div class="t3t"><div class="t3f" style="width:${it.conf}%;background:${cols[i]}"></div></div>
        <span class="t3p">${it.conf.toFixed(1)}%</span>
      </div>`).join('');
  }catch(e){document.getElementById('resPlaceholder').style.display='block';document.getElementById('resPlaceholder').textContent='❌ Prediction failed. Try again.'}
  btn.textContent='⚡ PREDICT GESTURE';btn.disabled=false;
}

// ── COLLECT ───────────────────────────────────────────────────
let collecting=false,cCount=0,cTimer=null;
const TOTAL=100;
function onNameInput(){
  const v=document.getElementById('gName').value.trim();
  document.getElementById('cBtn').disabled=!v;
  if(v) document.getElementById('cBtn').textContent=`📸 COLLECT "${v.toUpperCase()}"`;
  else  document.getElementById('cBtn').textContent='📸 START COLLECTION';
}
function startCollect(){
  const name=document.getElementById('gName').value.trim().toUpperCase();
  if(!name)return;
  if(!stream){
    go('live');
    setTimeout(()=>alert('Please click START to enable your camera first, then go to Add Gesture.'),500);
    return;
  }
  if(collecting){cancelCollect();return}
  collecting=true;cCount=0;
  document.getElementById('pw').style.display='block';
  document.getElementById('cBtn').textContent='⏹ CANCEL';
  document.getElementById('cStatus').textContent=`Collecting "${name}"...`;
  cTimer=setInterval(()=>{
    if(!ws||ws.readyState!==1||!stream)return;
    scx.drawImage(vid,0,0);
    ws.send(JSON.stringify({type:'collect',name,data:sc.toDataURL('image/jpeg',.7)}));
  },300);
}
function onCollectAck(d){
  cCount=d.count;
  const pct=(cCount/TOTAL)*100;
  document.getElementById('pf').style.width=pct+'%';
  document.getElementById('pCount').textContent=`${cCount} / ${TOTAL}`;
  document.getElementById('pLabel').textContent=`Collecting: ${d.name}`;
  if(cCount>=TOTAL){
    clearInterval(cTimer);collecting=false;
    document.getElementById('cBtn').textContent='✅ COLLECTED!';
    document.getElementById('cStatus').textContent=`✅ ${TOTAL} samples for "${d.name}" saved! Now run: python improved_train_model.py`;
    setTimeout(()=>{
      document.getElementById('cBtn').textContent='📸 START COLLECTION';
      document.getElementById('pw').style.display='none';
      cCount=0;loadStatus();
    },4000);
  }
}
function cancelCollect(){
  clearInterval(cTimer);collecting=false;cCount=0;
  document.getElementById('pw').style.display='none';
  document.getElementById('cBtn').textContent='📸 START COLLECTION';
}
</script>
</body>
</html>
"""
