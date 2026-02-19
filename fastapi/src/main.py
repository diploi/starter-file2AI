from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import asyncio
import json
import os
import tempfile
import subprocess
from openai import AsyncOpenAI
import base64

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        # Replace stale connection for the same session (e.g. page refresh)
        old = self.active_connections.get(session_id)
        if old:
            try:
                await old.close()
            except Exception:
                pass
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    def get_connection(self, session_id: str) -> Optional[WebSocket]:
        return self.active_connections.get(session_id)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_text(json.dumps(message))

manager = ConnectionManager()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("DIPLOI_AI_GATEWAY_TOKEN"), base_url=os.getenv("DIPLOI_AI_GATEWAY_URL"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = ""):
    if not session_id:
        await websocket.close(code=4000, reason="session_id query parameter required")
        return

    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(session_id)

async def transcribe_audio(audio_bytes: bytes, filename: str, content_type: str) -> str:
    """Transcribe audio bytes using OpenAI Whisper API."""
    # Determine file extension from content type
    ext_map = {
        "audio/webm": ".webm",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/mp4": ".m4a",
        "video/webm": ".webm",
        "video/mp4": ".mp4",
    }
    ext = ext_map.get(content_type, ".webm")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with open(tmp.name, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
    return transcript.text


def extract_video_frames(video_bytes: bytes, content_type: str, max_frames: int = 4) -> List[bytes]:
    """Extract evenly-spaced frames from a video using ffmpeg."""
    ext_map = {"video/webm": ".webm", "video/mp4": ".mp4", "video/quicktime": ".mov"}
    ext = ext_map.get(content_type, ".webm")

    frames: List[bytes] = []
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp_video:
        tmp_video.write(video_bytes)
        tmp_video.flush()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Get video duration
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    tmp_video.name,
                ],
                capture_output=True, text=True,
            )
            try:
                duration = float(probe.stdout.strip())
            except (ValueError, AttributeError):
                duration = 10.0  # fallback

            # Calculate interval to get evenly-spaced frames
            interval = max(duration / (max_frames + 1), 0.5)

            # Extract frames as JPEG
            subprocess.run(
                [
                    "ffmpeg", "-i", tmp_video.name,
                    "-vf", f"fps=1/{interval}",
                    "-frames:v", str(max_frames),
                    "-q:v", "2",
                    os.path.join(tmp_dir, "frame_%03d.jpg"),
                ],
                capture_output=True,
            )

            # Read extracted frames
            for frame_file in sorted(os.listdir(tmp_dir)):
                frame_path = os.path.join(tmp_dir, frame_file)
                with open(frame_path, "rb") as f:
                    frames.append(f.read())

    return frames


async def extract_video_audio(video_bytes: bytes, content_type: str) -> Optional[bytes]:
    """Extract audio track from a video file using ffmpeg."""
    print(f'CONTENT TYPE {content_type}')
    ext_map = {"video/webm": ".webm", "video/mp4": ".mp4", "video/quicktime": ".mov"}
    ext = ext_map.get(content_type, ".webm")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp_video:
        tmp_video.write(video_bytes)
        tmp_video.flush()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp_audio:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", tmp_video.name,
                    "-vn", "-acodec", "libmp3lame", "-q:a", "4",
                    "-y", tmp_audio.name,
                ],
                capture_output=True,
            )
            if result.returncode == 0 and os.path.getsize(tmp_audio.name) > 0:
                with open(tmp_audio.name, "rb") as f:
                    return f.read()
    return None


async def process_with_openai(prompt: str, files: List[UploadFile], websocket: WebSocket):
    print('FILES BEING REVIEWED <<<<<<<<<<<<<<')
    print('PARAMS RECEIVED <<<<<<<<<<<<<<')
    try:
        print('TRY CATCH BLOCK RUNNING>>>>>>>>>>>>>>>>>>>>>>.')
        # Send processing status
        await manager.send_personal_message({
            "type": "processing",
            "content": "Analyzing files..."
        }, websocket)

        # Prepare messages for OpenAI
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process files
        for file in files:
            content = file['content']
            content_type = file['content_type'] or ''
            filename = file['filename']
            print(f'Processing file: {filename} ({content_type})')

            if content_type.startswith('image/'):
                # Image: encode to base64 and send as image_url
                base64_image = base64.b64encode(content).decode('utf-8')
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{base64_image}"
                    }
                })

            elif content_type.startswith('audio/'):
                # Audio: transcribe with Whisper and include transcript
                await manager.send_personal_message({
                    "type": "processing",
                    "content": f"Transcribing audio: {filename}..."
                }, websocket)
                try:
                    transcript = await transcribe_audio(content, filename, content_type)
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"\n\n[Audio transcription from {filename}]:\n{transcript}"
                    })
                except Exception as e:
                    print(f"Audio transcription failed: {e}")
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"\n\n[Audio file: {filename} — transcription failed: {str(e)}]"
                    })

            elif content_type.startswith('video/'):
                print('VIDEO RECEIVED')
                # Video: extract frames as images + transcribe audio track
                await manager.send_personal_message({
                    "type": "processing",
                    "content": f"Analyzing video: {filename}..."
                }, websocket)

                # Extract and attach frames
                # try:
                #     frames = extract_video_frames(content, content_type)
                #     if frames:
                #         messages[0]["content"].append({
                #             "type": "text",
                #             "text": f"\n\n[Frames extracted from video: {filename} ({len(frames)} frames)]"
                #         })
                #         for frame_bytes in frames:
                #             base64_frame = base64.b64encode(frame_bytes).decode('utf-8')
                #             messages[0]["content"].append({
                #                 "type": "image_url",
                #                 "image_url": {
                #                     "url": f"data:image/jpeg;base64,{base64_frame}"
                #                 }
                #             })
                #     else:
                #         messages[0]["content"].append({
                #             "type": "text",
                #             "text": f"\n\n[Video: {filename} — could not extract frames]"
                #         })
                # except Exception as e:
                #     print(f"Frame extraction failed: {e}")
                #     messages[0]["content"].append({
                #         "type": "text",
                #         "text": f"\n\n[Video: {filename} — frame extraction failed: {str(e)}]"
                #     })

                # Extract and transcribe audio track
                try:
                    print('EXTRACTING AUDIO')
                    print(f'CONTENT TYPE RECEIVED BEFORE EXTRACTION {content_type}')
                    audio_bytes = await extract_video_audio(content, content_type)
                    if audio_bytes:
                        transcript = await transcribe_audio(audio_bytes, filename, "audio/mpeg")
                        messages[0]["content"].append({
                            "type": "text",
                            "text": f"\n\n[Audio transcription from video {filename}]:\n{transcript}"
                        })
                except Exception as e:
                    print(f"Video audio transcription failed: {e}")

            else:
                # Other files: try to decode as text
                try:
                    text_content = content.decode('utf-8')
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"\n\nFile: {filename}\n{text_content}"
                    })
                except Exception:
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"\n\n[File: {filename} - Binary file, content not displayed]"
                    })

        # Call OpenAI API
        await manager.send_personal_message({
            "type": "processing",
            "content": "Generating response..."
        }, websocket)

        response = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            max_tokens=1000
        )

        # Send response back
        answer = response.choices[0].message.content
        await manager.send_personal_message({
            "type": "response",
            "content": answer
        }, websocket)

    except Exception as e:
        print(f"Error processing with OpenAI: {str(e)}")
        await manager.send_personal_message({
            "type": "error",
            "content": f"Error: {str(e)}"
        }, websocket)

@app.post("/api/process")
async def process_files(
    prompt: str = Form(...),
    session_id: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    """
    Endpoint to receive files and prompt, then process with OpenAI.
    session_id correlates this request with the correct WebSocket connection.
    """
    if not prompt and not files:
        raise HTTPException(status_code=400, detail="Prompt or files required")

    websocket = manager.get_connection(session_id)
    if not websocket:
        raise HTTPException(
            status_code=400,
            detail="No WebSocket connection found for this session. Please refresh and try again."
        )

    file_data = []
    for file in files:
        content = await file.read()
        file_data.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "content": content
        })

    asyncio.create_task(process_with_openai(prompt, file_data, websocket))
    return {"status": "processing", "message": "Request received"}


@app.get("/", response_class=HTMLResponse)
def read_root(): 
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastAPI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
        * {
            font-family: sans-serif;
            font-size: 16px;
        }

        html,
        body {
            margin: 0;
            min-height: 100vh;
            background: #202328;
            color: #fff;
        }

        body {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 32px;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
        }

        h1 {
            font-size: 24px;
        }

        p,
        form,
        hr {
            max-width: min(400px, 100%);
        }

        p {
            text-align: center;
            opacity: 0.8;
            line-height: 1.5;
        }

        button,
        .button {
            padding: 10px 18px;
            align-self: center;
            text-decoration: none;
            background: #6650fa;
            border-radius: 64px;
            border: none;
            color: #fff;
            cursor: pointer;
        }

        a {
            font-size: inherit;
            color: inherit;
        }

        hr {
            display: block;
            margin: 32px 0;
            width: 100%;
            height: 2px;
            background: #31363f;
            border: none;
        }

        a:last-child {
            margin-top: 32px;
        }

        code {
            font-family: monospace;
            font-size: 14px;
            background: #31363f;
            padding: 2px 4px;
            border-radius: 4px;
        }
        </style>
    </head>
    <body>
        <img
            alt="FastAPI logo"
            src="https://github.com/diploi/component-fastapi/raw/main/.diploi/icon.svg"
            width="64"
            height="64"
        />
        
        <h1>FastAPI</h1>

        <p>
            Your FastAPI application is up and running! You can start editing the code in
            <code>src/main.py</code> to build your backend API. 
            In development stage, FastAPI will automatically reload as you make changes.
            <br><br>
            <b> Install dependencies: </b><br> 
            Please use <code>uv add package_name</code> to add Python packages to your environment.
        </p>

        <hr />

        <a href="https://diploi.com/"
        ><img width="54" height="16" src="https://diploi.com/logo-white.svg"
        /></a>
    </body>
    </html>
    """