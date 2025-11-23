import cv2
import time
from ultralytics import YOLO
import datetime
import os
import requests
from dotenv import load_dotenv
from collections import deque

# ==========================================
# CONFIGURACIÓN
# ==========================================
FPS = 20
SEGUNDOS_ANTES = 5
SEGUNDOS_DESPUES = 5
TIEMPO_EN_ZONA = 10      # tiempo quieto para alerta (segundos)
TIEMPO_ALERTA_GLOBAL = 30  # cooldown global entre alertas

BUFFER_FRAMES = FPS * SEGUNDOS_ANTES
RESOLUCION_ENVIO = (1280, 720)  # resolución usada para grabar/enviar

# DVR Buffer (5s antes del evento)
buffer_video = deque(maxlen=BUFFER_FRAMES)

grabando = False
frames_post_evento = 0
video_writer = None
ruta_video = None

# ==========================================
# CARGAR ENV
# ==========================================
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("❌ ERROR: No se pudo cargar TELEGRAM_TOKEN o TELEGRAM_CHAT_ID desde .env")
    exit()

# ==========================================
# TELEGRAM
# ==========================================
def enviar_telegram(msg: str):
    try:
        requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            params={"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        )
    except Exception as e:
        print(f"⚠️ Error enviando mensaje Telegram: {e}")

def enviar_foto(ruta: str):
    try:
        with open(ruta, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID},
                files={"photo": f}
            )
    except Exception as e:
        print(f"⚠️ Error enviando foto: {e}")

def enviar_video(ruta: str):
    try:
        with open(ruta, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",
                data={"chat_id": TELEGRAM_CHAT_ID},
                files={"video": f}
            )
    except Exception as e:
        print(f"⚠️ Error enviando video: {e}")

# ==========================================
# DETECCIÓN DE CÁMARAS
# ==========================================
def escanear_camaras():
    cams = []
    print("\n🔍 Escaneando cámaras funcionales...")
    for i in range(10):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            ret, _ = cam.read()
            if ret:
                print(f"✔ Cámara funcional → ID {i}")
                cams.append(i)
            else:
                print(f"✖ Cámara detectada pero sin señal → ID {i}")
        cam.release()
    return cams

def seleccionar_camara():
    cams = escanear_camaras()
    if not cams:
        print("❌ No hay cámaras funcionales disponibles.")
        exit()
    usb = [c for c in cams if c > 0]
    cam_id = usb[0] if usb else cams[0]
    print(f"\n🎥 Se usará la cámara → ID {cam_id}")
    return cam_id, f"CAM-{cam_id}"

# ==========================================
# INICIALIZAR CÁMARA Y FONDO
# ==========================================
def inicializar_camara(cam_id):
    print("\n⚙️ Inicializando cámara...")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara ID {cam_id}")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUCION_ENVIO[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCION_ENVIO[1])

    print("🔄 Calentando cámara...")
    for _ in range(30):
        ret, _ = cap.read()
        if not ret:
            time.sleep(0.1)
        else:
            time.sleep(0.05)

    ret, fondo = cap.read()
    if not ret:
        print("❌ No se pudo obtener frame inicial de la cámara.")
        cap.release()
        exit()

    fondo_gray = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY)
    fondo_gray = cv2.GaussianBlur(fondo_gray, (21, 21), 0)

    print("✅ Cámara inicializada correctamente. Sistema listo.")
    return cap, fondo_gray

# ==========================================
# FUNCIÓN DVR INICIAR GRABACIÓN
# ==========================================
def iniciar_grabacion(carpeta):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"alerta_video_{timestamp}.mp4")
    alto, ancho = RESOLUCION_ENVIO[1], RESOLUCION_ENVIO[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta, fourcc, FPS, (ancho, alto))
    return out, ruta

# ==========================================
# SISTEMA DE SEGURIDAD
# ==========================================
carpeta_alertas = os.path.join(os.getcwd(), "capturas_alerta")
os.makedirs(carpeta_alertas, exist_ok=True)

cam_source, cam_name = seleccionar_camara()
cap, fondo = inicializar_camara(cam_source)

model = YOLO("yolov8n.pt")

objetos_interes = {0: "Persona", 2: "Carro", 16: "Perro"}
objetos_sospechosos = {0, 2}

objetos_en_zona = {}
ultima_alerta = 0

enviar_telegram(f"🛡️ Sistema de vigilancia IA iniciado en {cam_name}.")

# ==========================================
# LOOP PRINCIPAL
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer frame de la cámara. Saliendo...")
        break

    now = time.time()
    buffer_video.append(frame.copy())

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(fondo, blur)
    _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    mov_total = cv2.countNonZero(thresh)

    results = model(frame)[0]
    detectados = [d for d in results.boxes if int(d.cls) in objetos_interes]

    alerta_detectada = False
    ruta_foto = None

    indices_actuales = set(range(len(detectados)))
    keys_antiguas = list(objetos_en_zona.keys())
    for k in keys_antiguas:
        if k not in indices_actuales:
            objetos_en_zona.pop(k, None)

    for i, obj in enumerate(detectados):
        cls_id = int(obj.cls)
        nombre = objetos_interes[cls_id]
        xmin, ymin, xmax, ymax = map(int, obj.xyxy[0])
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, nombre, (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if i not in objetos_en_zona:
            objetos_en_zona[i] = now
        else:
            permanencia = now - objetos_en_zona[i]
            cv2.putText(frame, f"{int(permanencia)}s en zona", (xmin, ymax + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if permanencia >= TIEMPO_EN_ZONA and now - ultima_alerta > TIEMPO_ALERTA_GLOBAL:
                alerta_detectada = True
                foto_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ruta_foto = os.path.join(carpeta_alertas, f"alerta_permanencia_{nombre}_{foto_ts}.jpg")
                cv2.imwrite(ruta_foto, frame)
                ultima_alerta = now
                enviar_telegram(f"🚨 {nombre} permaneció más de {TIEMPO_EN_ZONA}s — {cam_name}")
                enviar_foto(ruta_foto)

        if i in objetos_en_zona:
            permanencia_actual = now - objetos_en_zona[i]
            if permanencia_actual >= 5 and cls_id in objetos_sospechosos:
                if now - ultima_alerta > TIEMPO_ALERTA_GLOBAL:
                    alerta_detectada = True
                    foto_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    ruta_foto = os.path.join(carpeta_alertas, f"alerta_intrusion_{nombre}_{foto_ts}.jpg")
                    cv2.imwrite(ruta_foto, frame)
                    ultima_alerta = now
                    enviar_telegram(f"🚨 Intrusión detectada ({nombre}) en {cam_name}.")
                    enviar_foto(ruta_foto)

    if alerta_detectada and not grabando:
        enviar_telegram(f"🔥 ALERTA en {cam_name} — enviando video (DVR: {SEGUNDOS_ANTES}s antes + {SEGUNDOS_DESPUES}s después)")
        video_writer, ruta_video = iniciar_grabacion(carpeta_alertas)
        for f in buffer_video:
            resized = cv2.resize(f, RESOLUCION_ENVIO)
            video_writer.write(resized)
        grabando = True
        frames_post_evento = 0

    if grabando:
        resized = cv2.resize(frame, RESOLUCION_ENVIO)
        video_writer.write(resized)
        frames_post_evento += 1
        if frames_post_evento >= SEGUNDOS_DESPUES * FPS:
            video_writer.release()
            grabando = False
            enviar_video(ruta_video)
            enviar_telegram("🎥 Video de alerta enviado.")

    cv2.imshow(f"Seguridad IA – {cam_name}", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        print("👋 Cerrando sistema por tecla ESC.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Sistema detenido correctamente.")
