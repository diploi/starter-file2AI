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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
        logger.info(f"WebSocket connection accepted for session: {session_id}")
        # Replace stale connection for the same session (e.g. page refresh)
        old = self.active_connections.get(session_id)
        if old:
            try:
                await old.close()
                logger.info(f"Closed stale connection for session: {session_id}")
            except Exception as e:
                logger.warning(f"Error closing stale connection for session {session_id}: {e}")
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        logger.info(f"WebSocket disconnected for session: {session_id}")

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
        logger.warning("WebSocket connection rejected: missing session_id")
        await websocket.close(code=4000, reason="session_id query parameter required")
        return

    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
        manager.disconnect(session_id)

async def transcribe_audio(audio_bytes: bytes, filename: str, content_type: str) -> str:
    """Transcribe audio bytes using OpenAI Whisper API."""
    logger.info(f"Starting audio transcription for file: {filename} ({content_type})")
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
            logger.info(f"Transcription completed for {filename}: {transcript}")
            
            # Handle DIPLOI AI Gateway's non-standard response format
            # The gateway wraps the response in a 'result' field
            if transcript.text is None and hasattr(transcript, 'result') and isinstance(transcript.result, dict):
                text_content = transcript.result.get('text')
                if text_content:
                    logger.info(f"Extracted text from transcript.result: {text_content}")
                    return text_content
                else:
                    logger.error(f"transcript.text is None and no text found in result for {filename}")
                    raise ValueError(f"Transcription returned no text for {filename}")
            elif transcript.text:
                return transcript.text
            else:
                logger.error(f"transcript.text is None for {filename}. Full transcript: {transcript}")
                raise ValueError(f"Transcription returned None for {filename}")
    
    return transcript.text


def extract_video_frames(video_bytes: bytes, content_type: str, max_frames: int = 4) -> List[bytes]:
    """Extract evenly-spaced frames from a video using ffmpeg."""
    logger.info(f"Extracting {max_frames} frames from video ({content_type})")
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
                logger.debug(f"Video duration: {duration}s")
            except (ValueError, AttributeError):
                logger.warning("Could not determine video duration, using fallback value")
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

    logger.info(f"Successfully extracted {len(frames)} frames from video")
    return frames


async def extract_video_audio(video_bytes: bytes, content_type: str) -> Optional[bytes]:
    """Extract audio track from a video file using ffmpeg."""
    logger.info(f"Extracting audio from video ({content_type})")
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
                audio_size = os.path.getsize(tmp_audio.name)
                logger.info(f"Successfully extracted audio: {audio_size} bytes")
                with open(tmp_audio.name, "rb") as f:
                    return f.read()
            else:
                logger.warning(f"Audio extraction failed or produced no output (return code: {result.returncode})")
    return None


async def process_with_openai(prompt: str, files: List[UploadFile], websocket: WebSocket):
    logger.info(f"Processing request with prompt length: {len(prompt)} and {len(files)} files")
    try:
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
            logger.info(f"Processing file: {filename} ({content_type}, {len(content)} bytes)")

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
                    logger.error(f"Audio transcription failed for {filename}: {e}", exc_info=True)
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"\n\n[Audio file: {filename} — transcription failed: {str(e)}]"
                    })

            elif content_type.startswith('video/'):
                logger.info(f"Processing video file: {filename}")
                # Video: extract frames as images + transcribe audio track
                await manager.send_personal_message({
                    "type": "processing",
                    "content": f"Analyzing video: {filename}..."
                }, websocket)

                # Extract and transcribe audio track
                try:
                    audio_bytes = await extract_video_audio(content, content_type)
                    if audio_bytes:
                        transcript = await transcribe_audio(audio_bytes, filename, "audio/mpeg")
                        messages[0]["content"].append({
                            "type": "text",
                            "text": f"\n\n[Audio transcription from video {filename}]:\n{transcript}"
                        })
                    else:
                        logger.info(f"No audio track found in video: {filename}")
                except Exception as e:
                    logger.error(f"Video audio transcription failed for {filename}: {e}", exc_info=True)

            else:
                # Other files: try to decode as text
                try:
                    text_content = content.decode('utf-8')
                    logger.info(f"Decoded text file: {filename}")
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"\n\nFile: {filename}\n{text_content}"
                    })
                except Exception:
                    logger.info(f"Binary file cannot be decoded as text: {filename}")
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"\n\n[File: {filename} - Binary file, content not displayed]"
                    })

        # Call OpenAI API
        await manager.send_personal_message({
            "type": "processing",
            "content": "Generating response..."
        }, websocket)

        logger.info("Calling OpenAI API for chat completion")
        response = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            max_tokens=1000
        )

        # Send response back
        answer = response.choices[0].message.content
        logger.info(f"OpenAI response received: {len(answer)} characters")
        await manager.send_personal_message({
            "type": "response",
            "content": answer
        }, websocket)

    except Exception as e:
        logger.error(f"Error processing with OpenAI: {str(e)}", exc_info=True)
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
    logger.info(f"Received process request for session: {session_id} with {len(files)} files")
    if not prompt and not files:
        logger.warning(f"Invalid request from session {session_id}: missing prompt and files")
        raise HTTPException(status_code=400, detail="Prompt or files required")

    websocket = manager.get_connection(session_id)
    if not websocket:
        logger.error(f"No WebSocket connection found for session: {session_id}")
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
    logger.info(f"Processing task created for session: {session_id}")
    return {"status": "processing", "message": "Request received"}