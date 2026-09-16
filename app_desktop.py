import os
import sys
import time
import threading
import webbrowser
import socket
import traceback

# PyInstaller --windowed rejimida sys.stdout va sys.stderr None bo'ladi.
# Uvicorn va print() xatolik bermasligi uchun NullWriter o'rnatamiz:
class NullWriter:
    def write(self, s):
        pass
    def flush(self):
        pass
    def isatty(self):
        return False

if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()
if sys.stdin is None:
    try:
        sys.stdin = open(os.devnull, 'r')
    except Exception:
        pass

# Log fayli manzili
app_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(app_dir, "app_desktop.log")

def log(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass

import uvicorn
from backend.main import app

def find_free_port():
    """Bo'sh portni topish (8000 dan boshlab sinab ko'radi)"""
    for port in [8000, 8080, 8888]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    # Agar ular band bo'lsa, erkin port oladi
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def start_server(port):
    """FastAPI serverini ishga tushirish"""
    try:
        log(f"FastAPI uvicorn serveri ishga tushirilmoqda: port {port}")
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            access_log=False
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception:
        log(f"Server ishga tushishida xatolik:\n{traceback.format_exc()}")

def wait_for_server(port, timeout=25):
    """Server to'liq ulanishlarni qabul qilishga tayyor bo'lishini tekshirish"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.2)
    return False

def main():
    try:
        log("=== Kiber AI dasturi ishga tushdi ===")
        port = find_free_port()
        log(f"Tanlangan port: {port}")
        
        server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
        server_thread.start()
        
        # Server to'liq ulanishga tayyor bo'lishini kutamiz (maksimum 25 soniya)
        log("Server tayyor bo'lishi kutilmoqda...")
        server_ready = wait_for_server(port, timeout=25)
        if not server_ready:
            log("OGOHLANTIRISH: Server 25 soniya ichida javob bermadi!")
        else:
            log("Server muvaffaqiyatli ishga tushdi va ulanishga tayyor!")
        
        url = f"http://127.0.0.1:{port}"
        
        # config.json faylini o'qib, oyna sarlavhasini olish
        app_name = "Kiber AI - Sanoat Diagnostika Tizimi"
        try:
            import json
            config_path = os.path.join(app_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    app_name = config.get("app_name", app_name)
        except Exception as e:
            log(f"config.json o'qishda xatolik (window sarlavhasi uchun): {e}")

        # 1-variant: PyWebView orqali mustaqil Desktop oyna sifatida ochish
        opened_in_webview = False
        try:
            import webview
            log("PyWebView oynasi ochilmoqda...")
            window = webview.create_window(
                title=app_name,
                url=url,
                width=1366,
                height=850,
                min_size=(960, 640),
                background_color="#0a0f1a"
            )
            webview.start()
            opened_in_webview = True
            log("PyWebView oynasi yopildi. Dastur yakunlanmoqda.")
        except Exception as e:
            log(f"PyWebView ishlamadi ({e}). Brauzerda ochiladi...")
            opened_in_webview = False
            
        # 2-variant: Agar PyWebView ishlamasa, brauzerda ochish va serverni saqlab turish
        if not opened_in_webview:
            log(f"Standart brauzer ochilmoqda: {url}")
            webbrowser.open(url)
            # Brauzer yopilmaguncha yoki server to'xtamaguncha kutib turish
            while server_thread.is_alive():
                time.sleep(1)
                
    except Exception:
        log(f"Asosiy oqimda xatolik:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
