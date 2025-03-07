import argparse
import os
import signal
import psutil
from contextlib import asynccontextmanager
from typing import Any, Dict, Tuple
from subprocess import Popen
import logging
from logging.handlers import RotatingFileHandler

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

load_dotenv(override=True)

MAX_BOTS_PER_ROOM = 1
daily_helpers = {}
security = HTTPBearer(auto_error=False)

JWT_PRIVATE_KEY = "S^per$ec@8WKey!".encode()
JWT_ALGO = "HS256"
SESSION_LIFETIME = 180
JWT_TTL = 200000

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = RotatingFileHandler('chat_api.log', maxBytes=10485760, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class ProcessManager:
    def __init__(self):
        self.bot_procs: Dict[int, Tuple[Popen, str]] = {}
        
    def child_handler(self, signum, frame):
        try:
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                if pid in self.bot_procs:
                    self.bot_procs.pop(pid)
        except ChildProcessError:
            pass

    def start_bot(self, bot_file: str, room_url: str, token: str, product_id: str) -> Popen:
        signal.signal(signal.SIGCHLD, self.child_handler)
        cmd = ["python3", "-m", bot_file, "-u", room_url, "-t", token,
               "--product_id", product_id]
        proc = Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            preexec_fn=os.setsid
        )
        self.bot_procs[proc.pid] = (proc, room_url)
        return proc

    def cleanup_zombie(self):
        for pid, (proc, _) in list(self.bot_procs.items()):
            if proc.poll() is not None:
                try:
                    psutil.Process(pid).wait()
                    self.bot_procs.pop(pid)
                except psutil.NoSuchProcess:
                    self.bot_procs.pop(pid)

    def cleanup_all(self):
        for pid, (proc, _) in self.bot_procs.items():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    pass
        self.bot_procs.clear()

    def get_bot_count(self, room_url: str) -> int:
        return sum(
            1 for proc in self.bot_procs.values() 
            if proc[1] == room_url and proc[0].poll() is None
        )

process_mgr = ProcessManager()

def get_bot_file():
    bot_implementation = os.getenv("BOT_IMPLEMENTATION", "gemini").lower().strip()
    if not bot_implementation:
        bot_implementation = "gemini"
    if bot_implementation not in ["openai", "gemini"]:
        raise ValueError(f"Invalid BOT_IMPLEMENTATION: {bot_implementation}")
    return f"bot-{bot_implementation}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    process_mgr.cleanup_all()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_event():
    process_mgr.cleanup_all()

async def create_room_and_token() -> tuple[str, str]:
    room = await daily_helpers["rest"].create_room(DailyRoomParams())
    if not room.url:
        raise HTTPException(status_code=500, detail="Failed to create room")

    token = await daily_helpers["rest"].get_token(room.url)
    if not token:
        raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room.url}")

    return room.url, token

@app.post("/connect")
async def rtvi_connect(request: Request) -> Dict[Any, Any]:
    # body = await request.json()

    room_url, token = await create_room_and_token()

    # product_id = body.get("product_id")
    product_id = "005"

    try:
        process_mgr.cleanup_zombie()
        bot_file = get_bot_file()
        proc = process_mgr.start_bot(bot_file, room_url, token, product_id)
    except Exception as e:
        logger.error(f"Failed to start bot process: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start bot process: {e}")

    return {"room_url": room_url, "token": token}

@app.get("/status/{pid}")
def get_status(pid: int):
    if pid not in process_mgr.bot_procs:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    proc = process_mgr.bot_procs[pid][0]
    status = "running" if proc.poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})

if __name__ == "__main__":
    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )