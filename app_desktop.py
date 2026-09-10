import os
import sys
import time
import threading
import webbrowser
import socket
import uvicorn
from backend.main import app

def find_free_port():
    """Bo'sh portni topish"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_server(port):
    """FastAPI serverini ishga tushirish"""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

def main():
    port = find_free_port()
    
    # Serverni alohida oqimda ishga tushirish
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()
    
    # Server to'liq ishga tushishini kutish
    time.sleep(1.2)
    
    url = f"http://127.0.0.1:{port}"
    print(f"==================================================")
    print(f"  Kiber AI Desktop dasturi muvaffaqiyatli ishga tushdi!")
    print(f"  Manzil: {url}")
    print(f"==================================================")
    
    # GUI oyna sifatida ochish (agar pywebview o'rnatilgan bo'lsa)
    try:
        import webview
        window = webview.create_window(
            title="Kiber AI - Sanoat Diagnostika Tizimi",
            url=url,
            width=1366,
            height=850,
            min_size=(960, 640),
            background_color="#0a0f1a"
        )
        webview.start()
    except Exception:
        # Agar webview bo'lmasa, standart brauzerda ochadi
        webbrowser.open(url)
        print("Dasturni yopish uchun ushbu terminal oynasini yoping yoki Ctrl+C bosing.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Dastur to'xtatildi.")

if __name__ == "__main__":
    main()
