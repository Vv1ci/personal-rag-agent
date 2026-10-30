"""Desktop process supervisor. Only terminates children it started."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
from urllib.request import urlopen
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
CONTROL_PORT = 18768
APP_ID = 'personal-rag-launcher-v1'


def get_json(url):
    with urlopen(url, timeout=2) as response:
        return json.load(response)


def free_port(port):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def main():
    os.chdir(ROOT)
    root = tk.Tk()
    root.title('个人知识库 · 启动中心')
    root.geometry('600x340')
    state = {'app': APP_ID, 'ready': False, 'url': 'http://127.0.0.1:8501'}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/status':
                self.send_error(404); return
            data = json.dumps(state).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.end_headers(); self.wfile.write(data)
        def log_message(self, *args):
            pass
    try:
        server = ThreadingHTTPServer(('127.0.0.1', CONTROL_PORT), Handler)
    except OSError:
        try:
            existing = get_json(f'http://127.0.0.1:{CONTROL_PORT}/status')
            if existing.get('app') != APP_ID:
                raise RuntimeError('启动器端口被其他程序占用')
            if existing.get('ready'):
                webbrowser.open(existing['url'])
            else:
                messagebox.showinfo('已运行', '启动中心已打开，请在已有窗口查看进度。')
        except Exception as exc:
            messagebox.showerror('无法启动', str(exc))
        root.destroy(); return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    children, logs = [], []
    stopping = threading.Event()
    operation = threading.Lock()
    status = tk.StringVar(value='检查环境后会自动启动；首次使用可点击“初始化模型”。')
    tk.Label(root, text='个人知识库', font=('Microsoft YaHei', 20)).pack(pady=15)
    tk.Label(root, textvariable=status, wraplength=550, justify='left').pack(pady=12)

    def say(text):
        root.after(0, status.set, text)

    def spawn(args, label, env=None):
        if stopping.is_set():
            raise RuntimeError('启动已取消')
        folder = ROOT / 'logs'; folder.mkdir(exist_ok=True)
        log = (folder / f'launcher-{label}.log').open('a', encoding='utf-8')
        logs.append(log)
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        children.append(process)
        return process

    def wait_for(url, process, timeout=90):
        until = time.time() + timeout
        while time.time() < until:
            if stopping.is_set() or process.poll() is not None:
                raise RuntimeError('服务启动失败或已停止，请查看 logs 中的日志')
            try:
                with urlopen(url, timeout=2) as response:
                    if response.status == 200: return
            except Exception:
                time.sleep(.5)
        raise RuntimeError('服务启动超时，请查看 logs 中的日志')

    def stop_children():
        state['ready'] = False
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        children.clear()
        for log in logs: log.close()
        logs.clear()

    def run(setup=False):
        if not operation.acquire(blocking=False): return
        stopping.clear()
        try:
            missing = [m for m in ['fastapi', 'uvicorn', 'streamlit', 'chromadb', 'sentence_transformers', 'dotenv', 'multipart'] if importlib.util.find_spec(m) is None]
            if missing:
                raise RuntimeError('缺少依赖：' + ', '.join(missing) + '。请先双击“安装依赖.cmd”。')
            from rag.config import settings
            base = settings.ollama_base_url.rstrip('/')
            ollama = shutil.which('ollama') or str(Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Ollama/ollama.exe')
            try: get_json(base + '/api/tags')
            except Exception:
                if base != 'http://127.0.0.1:11434' and base != 'http://localhost:11434':
                    raise RuntimeError('配置的 Ollama 服务不可达：' + base)
                if not Path(ollama).is_file():
                    raise RuntimeError('未找到 Ollama，请先安装 Ollama，再启动知识库。')
                say('正在启动 Ollama…')
                process = spawn([ollama, 'serve'], 'ollama')
                wait_for(base + '/api/tags', process)
            if setup:
                say('正在下载语言模型，进度写入 logs/launcher-model.log…')
                env = dict(os.environ, OLLAMA_HOST=base)
                process = spawn([ollama, 'pull', settings.ollama_model], 'model', env)
                if process.wait() != 0: raise RuntimeError('语言模型下载失败，请查看日志')
                say('正在准备嵌入模型，首次下载可能较慢…')
                process = spawn([sys.executable, 'download_embedding_model.py'], 'embedding')
                if process.wait() != 0: raise RuntimeError('嵌入模型下载失败，请查看日志')
            names = [item['name'] for item in get_json(base + '/api/tags').get('models', [])]
            if settings.ollama_model not in names and settings.ollama_model + ':latest' not in names:
                raise RuntimeError('未找到语言模型，请点击“初始化模型”。')
            say('正在检查本地嵌入模型…')
            process = spawn([sys.executable, '-c', 'from rag.embeddings.local_embedding import load_embedding_model; load_embedding_model()'], 'check')
            if process.wait() != 0: raise RuntimeError('未找到可用嵌入模型，请点击“初始化模型”。')
            if state['ready']:
                webbrowser.open(state['url']); return
            if not free_port(settings.api_port) or not free_port(8501):
                raise RuntimeError('API 或页面端口已占用，请先关闭此前手动启动的知识库。')
            env = dict(os.environ, PYTHONUTF8='1', APP_ENV='production', API_HOST='127.0.0.1', API_BASE_URL=f'http://127.0.0.1:{settings.api_port}')
            say('正在启动知识库服务…')
            api = spawn([sys.executable, '-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', str(settings.api_port)], 'api', env)
            api._rag_service = True
            wait_for(env['API_BASE_URL'] + '/health', api)
            ui = spawn([sys.executable, '-m', 'streamlit', 'run', 'ui/streamlit_app.py', '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true', '--browser.gatherUsageStats', 'false'], 'ui', env)
            ui._rag_service = True
            wait_for(state['url'] + '/_stcore/health', ui)
            state['ready'] = True
            say('已启动。可打开页面；关闭本窗口会停止本次启动的服务。')
            webbrowser.open(state['url'])
        except Exception as exc:
            stop_children()
            say(str(exc))
        finally:
            operation.release()

    def start(setup=False): threading.Thread(target=run, args=(setup,), daemon=True).start()
    def monitor():
        if state['ready'] and any(p.poll() is not None for p in children if getattr(p, '_rag_service', False)):
            state['ready'] = False
            stop()
            say('服务已退出，请查看 logs 后重新启动。')
        root.after(2000, monitor)
    def stop():
        stopping.set()
        # Terminate child downloads too; wait for startup worker before final cleanup.
        for process in list(children):
            if process.poll() is None: process.terminate()
        def finish():
            with operation: stop_children()
            say('已停止。可重新启动。')
        threading.Thread(target=finish, daemon=True).start()
    buttons = tk.Frame(root); buttons.pack(pady=15)
    for text, command in [('启动 / 打开', lambda: start()), ('初始化模型', lambda: start(True)), ('停止', stop)]:
        tk.Button(buttons, text=text, command=command, width=16).pack(side='left', padx=6)
    tk.Label(root, text='文档、向量和对话留在本机。模型初始化需要联网。').pack()
    def close():
        stop()
        def finish():
            with operation: stop_children()
            server.shutdown()
            root.after(0, root.destroy)
        threading.Thread(target=finish, daemon=True).start()
    root.protocol('WM_DELETE_WINDOW', close)
    root.after(300, start)
    root.after(2000, monitor)
    root.mainloop()


if __name__ == '__main__':
    main()
