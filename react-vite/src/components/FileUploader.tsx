import React, { useState, useRef, useCallback } from 'react';
import './FileUploader.css';
import { UploadedFile } from '../types';
import { getSessionId, getDefaultPrompt } from '../utils';
import { useWebSocket } from '../hooks/useWebSocket';
import { useMediaRecorder } from '../hooks/useMediaRecorder';
import { MessageList } from './MessageList';
import { PromptInput } from './PromptInput';
import { MediaControls } from './MediaControls';
import { FileDropzone } from './FileDropzone';
import { AttachedFileCard } from './AttachedFileCard';

const api_endpoint = import.meta.env.VITE_API_ROOT_ENDPOINT ? `${import.meta.env.VITE_API_ROOT_ENDPOINT}/api/process` : 'https://api--file2ai.diploi.me/api/process'

export const FileUploader: React.FC = () => {
  const sessionIdRef = useRef<string>(getSessionId());
  const storageKey = `messages_${sessionIdRef.current}`;

  const [attachedFile, setAttachedFile] = useState<UploadedFile | null>(null);
  const [prompt, setPrompt] = useState('');

  const {
    messages,
    isProcessing,
    copiedMessageId,
    addUserMessage,
    addProcessingMessage,
    addErrorMessage,
    clearMessages,
    setIsProcessing,
    copyMessage,
    messagesEndRef,
  } = useWebSocket(sessionIdRef.current, storageKey);

  const attachFile = useCallback((file: File) => {
    const uploadedFile: UploadedFile = {
      file,
      id: `${file.name}-${Date.now()}-${Math.random()}`,
    };

    if (file.type.startsWith('image/') || file.type.startsWith('audio/') || file.type.startsWith('video/')) {
      uploadedFile.preview = URL.createObjectURL(file);
    }

    setAttachedFile(uploadedFile);

    setPrompt(prev => {
      if (prev.trim() === '') {
        return getDefaultPrompt(file.type);
      }
      return prev;
    });
  }, []);

  const {
    isRecordingAudio,
    isRecordingVideo,
    isRecording,
    videoPreviewRef,
    startAudioRecording,
    startVideoRecording,
    stopRecording,
  } = useMediaRecorder(attachFile);

  const removeFile = useCallback(() => {
    if (attachedFile?.preview) {
      URL.revokeObjectURL(attachedFile.preview);
    }
    setAttachedFile(null);
  }, [attachedFile]);

  const handleFilesSelected = useCallback((fileList: FileList | null) => {
    if (fileList && fileList.length > 0) {
      attachFile(fileList[0]);
    }
  }, [attachFile]);

  const handleUpload = async () => {
    if (!prompt.trim() && !attachedFile) {
      alert('Please add a prompt or a file');
      return;
    }

    setIsProcessing(true);

    const fileName = attachedFile ? attachedFile.file.name : undefined;
    addUserMessage(prompt, fileName ? [fileName] : undefined);
    addProcessingMessage();

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('session_id', sessionIdRef.current);

      if (attachedFile) {
        formData.append('files', attachedFile.file);
      }


      const response = await fetch(api_endpoint, {
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
      addErrorMessage(`Error: ${error instanceof Error ? error.message : 'Upload failed'}`);
      setIsProcessing(false);
    }
  };

  return (
    <div className="file-uploader-container">
      <MessageList
        messages={messages}
        copiedMessageId={copiedMessageId}
        onCopyMessage={copyMessage}
        onClearMessages={clearMessages}
        messagesEndRef={messagesEndRef}
      />

      <PromptInput
        prompt={prompt}
        onPromptChange={setPrompt}
        isProcessing={isProcessing}
      />

      <MediaControls
        isRecordingAudio={isRecordingAudio}
        isRecordingVideo={isRecordingVideo}
        isProcessing={isProcessing}
        videoPreviewRef={videoPreviewRef}
        onStartAudioRecording={startAudioRecording}
        onStartVideoRecording={startVideoRecording}
        onStopRecording={stopRecording}
      />

      {!attachedFile && !isRecording && (
        <FileDropzone
          isProcessing={isProcessing}
          onFilesSelected={handleFilesSelected}
        />
      )}

      {attachedFile && (
        <AttachedFileCard
          file={attachedFile}
          isProcessing={isProcessing}
          onRemove={removeFile}
        />
      )}

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
