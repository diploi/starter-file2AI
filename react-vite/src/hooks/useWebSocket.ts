import { useState, useEffect, useRef, useCallback } from 'react';
import { Message } from '../types';
import { loadMessages } from '../utils';

export interface UseWebSocketReturn {
  messages: Message[];
  isProcessing: boolean;
  copiedMessageId: string | null;
  addUserMessage: (content: string, files?: string[]) => void;
  addProcessingMessage: () => void;
  addErrorMessage: (content: string) => void;
  clearMessages: () => void;
  setIsProcessing: React.Dispatch<React.SetStateAction<boolean>>;
  copyMessage: (messageId: string, content: string) => Promise<void>;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
}

const api_endpoint = import.meta.env.VITE_API_ROOT_ENDPOINT || ''
const wsRoot = api_endpoint.replace(/^http/, 'ws') || '';

export function useWebSocket(sessionId: string, storageKey: string): UseWebSocketReturn {
  const [messages, setMessages] = useState<Message[]>(() => loadMessages(storageKey));
  const [isProcessing, setIsProcessing] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // WebSocket connection
  useEffect(() => {
    let websocket: WebSocket | null = null;
    const connect = () => {
      const wsUrl = `${wsRoot}/ws?session_id=${sessionId}`;
      websocket = new WebSocket(wsUrl);

      websocket.onopen = () => {
        console.log('WebSocket connected, session:', sessionId);
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'response') {
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
        }
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      websocket.onclose = () => {
        console.log('WebSocket disconnected');
      };
    };

    connect();

    return () => {
      websocket?.close();
    };
  }, [sessionId]);

  // Persist messages to localStorage
  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(messages));
  }, [messages, storageKey]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addUserMessage = useCallback((content: string, files?: string[]) => {
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      type: 'user',
      content,
      timestamp: new Date(),
      files,
    }]);
  }, []);

  const addProcessingMessage = useCallback(() => {
    setMessages(prev => [...prev, {
      id: `processing-${Date.now()}`,
      type: 'processing',
      content: 'Processing your request...',
      timestamp: new Date(),
    }]);
  }, []);

  const addErrorMessage = useCallback((content: string) => {
    setMessages(prev => prev.filter(m => m.type !== 'processing'));
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      type: 'error',
      content,
      timestamp: new Date(),
    }]);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    localStorage.removeItem(storageKey);
  }, [storageKey]);

  const copyMessage = useCallback(async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (error) {
      console.error('Copy message failed:', error);
    }
  }, []);

  return {
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
  };
}
