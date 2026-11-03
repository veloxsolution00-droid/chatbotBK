"""One-command launcher for the Offline Document Intelligence Chatbot.

Run from the project root (the folder that contains `app/`):

    python run.py               # share with other devices on your network (default)
    python run.py --local-only  # this PC only
    python run.py --reload      # auto-restart backend when code changes (development)

Press Ctrl+C once to stop everything.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_HOST = "127.0.0.1"   # backend stays private; only the UI talks to it
BACKEND_PORT = 8000
UI_PORT = 8501
STARTUP_TIMEOUT = 120        # backend runs recovery/reconcile on startup, so allow time


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_lan_ip():
    """Best-effort lookup of this PC's address on the local network (no internet needed)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # nothing is actually sent
        return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "YOUR-PC-IP"
    finally:
        s.close()


def wait_for_backend(proc, timeout=STARTUP_TIMEOUT):
    """Poll /health until the backend answers. Returns the health dict, or None."""
    url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:      # backend crashed while starting
            return None
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(0.5)
    return None


def report_health(h):
    """Warn about the things that most often break an offline setup."""
    ollama = h.get("ollama") or {}
    if not ollama.get("available"):
        print("  WARNING: Ollama is not reachable. Start it (open the Ollama app or run "
              "`ollama serve`). Uploads and chat will not work until it is running.")
    else:
        for kind in ("generation", "embedding"):
            name = ollama.get(f"{kind}_model")
            if name and not ollama.get(f"{kind}_model_available"):
                print(f"  WARNING: {kind} model '{name}' is not installed. "
                      f"Run once (needs internet): ollama pull {name}")
    if not (h.get("chromadb") or {}).get("available"):
        print("  WARNING: ChromaDB is unavailable. Check the backend log above.")
    if not (h.get("metadata") or {}).get("available"):
        print("  WARNING: the metadata database is unavailable (locked or unreadable).")


def stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    parser = argparse.ArgumentParser(description="Start backend + Streamlit UI together.")
    parser.add_argument("--local-only", action="store_true",
                        help="only allow this PC to open the UI (default: share on your network)")
    parser.add_argument("--reload", action="store_true",
                        help="auto-restart the backend when code changes (development)")
    args = parser.parse_args()

    ui_host = "127.0.0.1" if args.local_only else "0.0.0.0"

    # Sanity checks before starting anything.
    for rel in ("app/main.py", "app/ui/streamlit_app.py"):
        if not os.path.exists(os.path.join(ROOT, rel)):
            print(f"Cannot find {rel}. Put run.py in the project root, next to the app/ folder.")
            return 1
    for port in (BACKEND_PORT, UI_PORT):
        if port_in_use(port):
            print(f"Port {port} is already in use. Close the old backend/UI "
                  f"(or the other terminal that started them) and try again.")
            return 1

    backend_cmd = [sys.executable, "-m", "uvicorn", "app.main:app",
                   "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)]
    if args.reload:
        backend_cmd.append("--reload")
    ui_cmd = [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py",
              "--server.address", ui_host, "--server.port", str(UI_PORT),
              "--server.headless", "true", "--browser.gatherUsageStats", "false"]

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    ui_env = dict(env, API_URL=f"http://{BACKEND_HOST}:{BACKEND_PORT}")

    backend = ui = None
    try:
        print(f"Starting backend on http://{BACKEND_HOST}:{BACKEND_PORT} ...")
        backend = subprocess.Popen(backend_cmd, cwd=ROOT, env=env)

        health = wait_for_backend(backend)
        if health is None:
            print("Backend did not start. Check the errors printed above.")
            return 1
        print("Backend is up.")
        report_health(health)

        print(f"Starting UI on port {UI_PORT} ...")
        ui = subprocess.Popen(ui_cmd, cwd=ROOT, env=ui_env)

        print("\n  On this PC:       http://localhost:%d" % UI_PORT)
        if args.local_only:
            print("  Sharing:          off (--local-only)\n")
        else:
            print("  On your network:  http://%s:%d" % (get_lan_ip(), UI_PORT))
            print("  Anyone who opens that address uses the same documents as you.\n")

        # Exit if either process dies.
        while backend.poll() is None and ui.poll() is None:
            time.sleep(1)
        print("A process exited, shutting down.")
        return 1
    except KeyboardInterrupt:
        print("\nStopping...")
        return 0
    finally:
        stop(ui)
        stop(backend)


if __name__ == "__main__":
    sys.exit(main())
