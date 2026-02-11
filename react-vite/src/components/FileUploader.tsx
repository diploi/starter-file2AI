import React, { useState, useRef, DragEvent, ChangeEvent, useEffect, useCallback } from 'react';
import './FileUploader.css';

interface UploadedFile {
  file: File;
  preview?: string;
  id: string;
}

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'error' | 'processing';
  content: string;
  timestamp: Date;
  files?: string[];
}

export const FileUploader: React.FC = () => {
  const [attachedFile, setAttachedFile] = useState<UploadedFile | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isRecordingAudio, setIsRecordingAudio] = useState(false);
  const [isRecordingVideo, setIsRecordingVideo] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaChunksRef = useRef<Blob[]>([]);
  const videoPreviewRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  let restartCount = 0;
  // WebSocket connection
  useEffect(() => {
    const wsUrl = 'wss://my-dev-2--file2ai-yk98.diploi.me/ws';
    restartCount++;

    const websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      console.log('WebSocket connected');
      console.log('Restart count: ' + restartCount);
    };

    websocket.onmessage = (event) => {
      console.log(event);
      console.log('MESSAGE RECEIVED');
      const data = JSON.parse(event.data);

      if (data.type === 'response') {
        console.log('RESPONSE RECEIVED');
        setMessages(prev => prev.filter(m => m.type !== 'processing'));
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          type: 'assistant',
          content: data.content,
          timestamp: new Date(),
        }]);
        setIsProcessing(false);
      } else if (data.type === 'error') {
        setMessages(prev => prev.filter(m => m.type !== 'processing'));
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          type: 'error',
          content: data.content,
          timestamp: new Date(),
        }]);
        setIsProcessing(false);
      } else if (data.type === 'processing') {
        console.log('Processing:', data.content);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setMessages(prev => prev.filter(m => m.type !== 'processing'));
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        type: 'error',
        content: 'Connection error. Please try again.',
        timestamp: new Date(),
      }]);
      setIsProcessing(false);
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
    };

    setWs(websocket);
    console.log("THIS RAN OUTSIDE OF CLEANUP FUNCTION");

    return () => {
      console.log("CLEANUP RUNNING");
      console.log('Restart count after cleanup: ' + restartCount);
      websocket.close();
    };
  }, []);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const attachFile = useCallback((file: File) => {
    const uploadedFile: UploadedFile = {
      file,
      id: `${file.name}-${Date.now()}-${Math.random()}`,
    };

    // Create preview for images, audio, and video
    if (file.type.startsWith('image/') || file.type.startsWith('audio/') || file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file);
      uploadedFile.preview = url;
    }

    setAttachedFile(uploadedFile);
  }, []);

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    attachFile(fileList[0]);
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const droppedFiles = e.dataTransfer.files;
    handleFiles(droppedFiles);
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    // Reset input so the same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const removeFile = () => {
    if (attachedFile?.preview) {
      URL.revokeObjectURL(attachedFile.preview);
    }
    setAttachedFile(null);
  };

  const stopMediaStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoPreviewRef.current) {
      videoPreviewRef.current.srcObject = null;
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const startAudioRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      mediaChunksRef.current = [];

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          mediaChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(mediaChunksRef.current, { type: 'audio/webm' });
        const file = new File([blob], `audio-recording-${Date.now()}.webm`, { type: 'audio/webm' });
        attachFile(file);
        stopMediaStream();
        setIsRecordingAudio(false);
      };

      recorder.start();
      setIsRecordingAudio(true);
    } catch (err) {
      console.error('Failed to start audio recording:', err);
      alert('Could not access microphone. Please check permissions.');
    }
  }, [attachFile, stopMediaStream]);

  const startVideoRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      mediaStreamRef.current = stream;
      mediaChunksRef.current = [];

      // Show live preview
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        videoPreviewRef.current.play();
      }

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          mediaChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(mediaChunksRef.current, { type: 'video/webm' });
        const file = new File([blob], `video-recording-${Date.now()}.webm`, { type: 'video/webm' });
        attachFile(file);
        stopMediaStream();
        setIsRecordingVideo(false);
      };

      recorder.start();
      setIsRecordingVideo(true);
    } catch (err) {
      console.error('Failed to start video recording:', err);
      alert('Could not access camera/microphone. Please check permissions.');
    }
  }, [attachFile, stopMediaStream]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const handleUpload = async () => {
    if (!prompt.trim() && !attachedFile) {
      alert('Please add a prompt or a file');
      return;
    }

    setIsProcessing(true);

    const fileName = attachedFile ? attachedFile.file.name : undefined;
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      type: 'user',
      content: prompt,
      timestamp: new Date(),
      files: fileName ? [fileName] : undefined,
    }]);

    setMessages(prev => [...prev, {
      id: `processing-${Date.now()}`,
      type: 'processing',
      content: 'Processing your request...',
      timestamp: new Date(),
    }]);

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);

      if (attachedFile) {
        formData.append('files', attachedFile.file);
      }

      console.log('FORMDATA TO SEND');
      console.log(formData.get('prompt'));
      console.log(formData.get('files'));

      const response = await fetch('https://my-dev-2--file2ai-yk98.diploi.me/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      setPrompt('');
      removeFile();
    } catch (error) {
      console.error('Upload error:', error);
      setMessages(prev => prev.filter(m => m.type !== 'processing'));
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        type: 'error',
        content: `Error: ${error instanceof Error ? error.message : 'Upload failed'}`,
        timestamp: new Date(),
      }]);
      setIsProcessing(false);
    }
  };

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const isRecording = isRecordingAudio || isRecordingVideo;

  return (
    <div className="file-uploader-container">
      {/* Messages Section */}
      {messages.length > 0 && (
        <div className="messages-section">
          <div className="messages-header">
            <h4>Conversation</h4>
            <button
              className="clear-messages-button"
              onClick={() => setMessages([])}
            >
              Clear
            </button>
          </div>
          <div className="messages-list">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message message-${message.type}`}
              >
                <div className="message-content">
                  {message.type === 'processing' ? (
                    <div className="processing-indicator">
                      <div className="spinner"></div>
                      <span>{message.content}</span>
                    </div>
                  ) : (
                    <>
                      {message.files && message.files.length > 0 && (
                        <div className="message-files">
                          {message.files.join(', ')}
                        </div>
                      )}
                      <p>{message.content}</p>
                    </>
                  )}
                </div>
                <div className="message-time">
                  {formatTime(message.timestamp)}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
      )}

      {/* Prompt Input */}
      <div className="prompt-section">
        <label htmlFor="prompt" className="prompt-label">
          Your Prompt
        </label>
        <textarea
          id="prompt"
          className="prompt-input"
          placeholder="Describe what you want to do with the file..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          disabled={isProcessing}
        />
      </div>

      {/* Video Recording Preview */}
      {isRecordingVideo && (
        <div className="video-preview-container">
          <video ref={videoPreviewRef} className="video-preview" muted playsInline />
          <div className="recording-overlay">
            <span className="recording-dot"></span> Recording video...
          </div>
        </div>
      )}

      {/* Audio Recording Indicator */}
      {isRecordingAudio && (
        <div className="audio-recording-indicator">
          <span className="recording-dot"></span>
          <span>Recording audio...</span>
        </div>
      )}

      {/* Media Buttons */}
      <div className="media-buttons">
        <button
          className={`media-button ${isRecordingAudio ? 'recording' : ''}`}
          onClick={isRecordingAudio ? stopRecording : startAudioRecording}
          disabled={isProcessing || isRecordingVideo}
          title={isRecordingAudio ? 'Stop recording' : 'Record audio'}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="media-button-icon">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
          {isRecordingAudio ? 'Stop' : 'Audio'}
        </button>

        <button
          className={`media-button ${isRecordingVideo ? 'recording' : ''}`}
          onClick={isRecordingVideo ? stopRecording : startVideoRecording}
          disabled={isProcessing || isRecordingAudio}
          title={isRecordingVideo ? 'Stop recording' : 'Record video'}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" className="media-button-icon">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          {isRecordingVideo ? 'Stop' : 'Video'}
        </button>
      </div>

      {/* File Upload Dropzone */}
      {!attachedFile && !isRecording && (
        <div
          className={`file-uploader-dropzone ${isDragging ? 'dragging' : ''} ${isProcessing ? 'disabled' : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={isProcessing ? undefined : handleClick}
          role="button"
          tabIndex={isProcessing ? -1 : 0}
          aria-label="Upload file"
        >
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileInputChange}
            className="file-input-hidden"
            aria-hidden="true"
            disabled={isProcessing}
          />

          <div className="dropzone-content">
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>

            <h3 className="dropzone-title">
              {isDragging ? 'Drop file here' : 'Upload a file (optional)'}
            </h3>

            <p className="dropzone-description">
              Tap to browse or drag and drop a file here
            </p>
          </div>
        </div>
      )}

      {/* Attached File */}
      {attachedFile && (
        <div className="attached-file-section">
          <div className="file-card">
            <button
              className="remove-file-button"
              onClick={(e) => {
                e.stopPropagation();
                removeFile();
              }}
              disabled={isProcessing}
              aria-label={`Remove ${attachedFile.file.name}`}
            >
              ×
            </button>

            {attachedFile.preview && attachedFile.file.type.startsWith('image/') ? (
              <div className="file-preview">
                <img src={attachedFile.preview} alt={attachedFile.file.name} />
              </div>
            ) : attachedFile.preview && attachedFile.file.type.startsWith('audio/') ? (
              <div className="file-preview-media">
                <audio src={attachedFile.preview} controls />
              </div>
            ) : attachedFile.preview && attachedFile.file.type.startsWith('video/') ? (
              <div className="file-preview-media">
                <video src={attachedFile.preview} controls className="video-thumbnail" />
              </div>
            ) : (
              <div className="file-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
            )}

            <div className="file-info">
              <p className="file-name" title={attachedFile.file.name}>
                {attachedFile.file.name}
              </p>
              <p className="file-size">
                {formatFileSize(attachedFile.file.size)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Submit Button */}
      <button
        className="upload-button"
        onClick={handleUpload}
        disabled={isProcessing || isRecording || (!prompt.trim() && !attachedFile)}
      >
        {isProcessing ? (
          <>
            <div className="button-spinner"></div>
            Processing...
          </>
        ) : (
          <>Submit</>
        )}
      </button>
    </div>
  );
};
