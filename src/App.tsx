import { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type } from '@google/genai';
import { Mic, Square, Loader2, Globe, MessageSquare, Languages, Trash2, Beaker } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface TranscriptionData {
  detected_language: string;
  original_text: string;
  english_translation: string;
}

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [transcription, setTranscription] = useState<TranscriptionData | null>(null);
  const [history, setHistory] = useState<TranscriptionData[]>([]);

  const aiRef = useRef<GoogleGenAI | null>(null);
  const sessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  useEffect(() => {
    aiRef.current = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    return () => {
      stopRecording();
    };
  }, []);

  const startRecording = async () => {
    try {
      setError(null);
      setIsConnecting(true);
      setTranscription(null);
      setHistory([]);

      if (!aiRef.current) throw new Error("GenAI client not initialized");

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      const gainNode = audioContext.createGain();
      gainNode.gain.value = 0;

      source.connect(processor);
      processor.connect(gainNode);
      gainNode.connect(audioContext.destination);

      const sessionPromise = aiRef.current.live.connect({
        model: "gemini-3.1-flash-live-preview",
        callbacks: {
          onopen: () => {
            setIsConnecting(false);
            setIsRecording(true);
            
            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcm16 = new Int16Array(inputData.length);
              for (let i = 0; i < inputData.length; i++) {
                let s = Math.max(-1, Math.min(1, inputData[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
              }
              
              const buffer = new ArrayBuffer(pcm16.length * 2);
              const view = new DataView(buffer);
              for (let i = 0; i < pcm16.length; i++) {
                view.setInt16(i * 2, pcm16[i], true);
              }
              
              const bytes = new Uint8Array(buffer);
              let binary = '';
              for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
              }
              const base64 = btoa(binary);

              sessionPromise.then(session => {
                session.sendRealtimeInput({
                  audio: { data: base64, mimeType: 'audio/pcm;rate=16000' }
                });
              }).catch(console.error);
            };
          },
          onmessage: (message: LiveServerMessage) => {
            if (message.toolCall) {
              const call = message.toolCall.functionCalls[0];
              if (call.name === "updateTranscription") {
                const args = call.args as unknown as TranscriptionData;
                setTranscription(args);
                setHistory(prev => {
                  // Simple deduplication/updating logic for history
                  const last = prev[prev.length - 1];
                  if (last && last.original_text === args.original_text) {
                    return prev;
                  }
                  // If the new text is an extension of the last one, replace it
                  if (last && args.original_text.startsWith(last.original_text)) {
                    const newHistory = [...prev];
                    newHistory[newHistory.length - 1] = args;
                    return newHistory;
                  }
                  return [...prev, args];
                });

                sessionPromise.then(session => {
                  session.sendToolResponse({
                    functionResponses: [{
                      id: call.id,
                      name: call.name,
                      response: { status: "ok" }
                    }]
                  });
                }).catch(console.error);
              }
            }
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setError(err instanceof Error ? err.message : String(err));
            stopRecording();
          },
          onclose: () => {
            stopRecording();
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: "You are an AI-powered real-time speech processing assistant. Continuously listen to incoming audio. Perform real-time transcription. Automatically detect the language. If the detected language is NOT English, translate the transcribed text into English in real time. If the spoken language is already English, simply return the transcription. Continuously call the `updateTranscription` function with the detected language, original text, and English translation as the user speaks. Do not wait for the user to finish speaking; process in real time. Keep updating the transcript as new audio comes in. Avoid hallucinations; only transcribe what is actually spoken. If audio is unclear, indicate uncertainty with [unclear]. Do not speak any audio, only call the function.",
          tools: [{
            functionDeclarations: [{
              name: "updateTranscription",
              description: "Update the live transcription and translation.",
              parameters: {
                type: Type.OBJECT,
                properties: {
                  detected_language: { type: Type.STRING, description: "The detected language of the speech." },
                  original_text: { type: Type.STRING, description: "The live transcription in the original language." },
                  english_translation: { type: Type.STRING, description: "The English translation, or the same as original if English." }
                },
                required: ["detected_language", "original_text", "english_translation"]
              }
            }]
          }]
        }
      });

      sessionRef.current = sessionPromise;

    } catch (err) {
      console.error("Failed to start recording:", err);
      setError(err instanceof Error ? err.message : String(err));
      setIsConnecting(false);
      setIsRecording(false);
    }
  };

  const clearTranscription = () => {
    setTranscription(null);
    setHistory([]);
  };

  const runTestCase = () => {
    if (isRecording) stopRecording();
    clearTranscription();
    
    const testData = [
      { detected_language: "French", original_text: "Bonjour", english_translation: "Hello" },
      { detected_language: "French", original_text: "Bonjour, je", english_translation: "Hello, I" },
      { detected_language: "French", original_text: "Bonjour, je voudrais", english_translation: "Hello, I would like" },
      { detected_language: "French", original_text: "Bonjour, je voudrais un café", english_translation: "Hello, I would like a coffee" },
      { detected_language: "French", original_text: "Bonjour, je voudrais un café s'il vous plaît.", english_translation: "Hello, I would like a coffee please." }
    ];

    let i = 0;
    const interval = setInterval(() => {
      if (i < testData.length) {
        const data = testData[i];
        setTranscription(data);
        setHistory(prev => {
          const last = prev[prev.length - 1];
          if (last && last.original_text === data.original_text) return prev;
          if (last && data.original_text.startsWith(last.original_text)) {
            const newHistory = [...prev];
            newHistory[newHistory.length - 1] = data;
            return newHistory;
          }
          return [...prev, data];
        });
        i++;
      } else {
        clearInterval(interval);
      }
    }, 600);
  };

  const stopRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (sessionRef.current) {
      sessionRef.current.then((session: any) => session.close()).catch(console.error);
      sessionRef.current = null;
    }
    setIsRecording(false);
    setIsConnecting(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans selection:bg-blue-200">
      <div className="max-w-4xl mx-auto p-6 lg:p-12 flex flex-col gap-8 h-screen">
        
        {/* Header */}
        <header className="flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-600 text-white rounded-xl shadow-sm">
              <Globe className="w-6 h-6" />
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">Live Translator</h1>
          </div>
          <p className="text-gray-500">Real-time transcription and translation powered by Gemini.</p>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col gap-6 overflow-hidden">
          
          {/* Controls */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between bg-white p-4 rounded-2xl shadow-sm border border-gray-100 gap-4">
            <div className="flex items-center gap-4">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isConnecting}
                className={cn(
                  "relative flex items-center justify-center w-14 h-14 rounded-full transition-all duration-300 shadow-sm",
                  isRecording 
                    ? "bg-red-50 text-red-600 hover:bg-red-100" 
                    : "bg-blue-600 text-white hover:bg-blue-700 hover:shadow-md hover:-translate-y-0.5",
                  isConnecting && "opacity-70 cursor-not-allowed"
                )}
              >
                {isConnecting ? (
                  <Loader2 className="w-6 h-6 animate-spin" />
                ) : isRecording ? (
                  <Square className="w-5 h-5 fill-current" />
                ) : (
                  <Mic className="w-6 h-6" />
                )}
                
                {/* Pulse effect when recording */}
                {isRecording && (
                  <span className="absolute inset-0 rounded-full border-2 border-red-500 animate-ping opacity-20"></span>
                )}
              </button>
              
              <div className="flex flex-col">
                <span className="font-medium text-gray-900">
                  {isConnecting ? "Connecting..." : isRecording ? "Listening..." : "Ready to record"}
                </span>
                <span className="text-sm text-gray-500">
                  {isRecording ? "Speak into your microphone" : "Click the microphone to start"}
                </span>
              </div>
            </div>
            
            <div className="flex items-center gap-3 flex-wrap">
              {transcription && (
                <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-700 rounded-full text-sm font-medium">
                  <Languages className="w-4 h-4" />
                  {transcription.detected_language}
                </div>
              )}
              
              <div className="h-8 w-px bg-gray-200 hidden sm:block"></div>
              
              <button
                onClick={runTestCase}
                disabled={isRecording || isConnecting}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors"
                title="Run a simulated transcription test"
              >
                <Beaker className="w-4 h-4" />
                Test Case
              </button>
              
              <button
                onClick={clearTranscription}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-600 bg-white border border-gray-200 rounded-lg hover:bg-red-50 transition-colors"
                title="Clear transcription history"
              >
                <Trash2 className="w-4 h-4" />
                Clear
              </button>
            </div>
          </div>

          {error && (
            <div className="p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 text-sm">
              {error}
            </div>
          )}

          {/* Transcription Display */}
          <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-6 min-h-0">
            
            {/* Original Text */}
            <div className="flex flex-col bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
              <div className="flex items-center gap-2 px-6 py-4 border-b border-gray-50 bg-gray-50/50">
                <MessageSquare className="w-4 h-4 text-gray-400" />
                <h2 className="font-medium text-gray-700">Original</h2>
              </div>
              <div className="flex-1 p-6 overflow-y-auto">
                {history.length === 0 && !transcription ? (
                  <p className="text-gray-400 italic">Waiting for speech...</p>
                ) : (
                  <div className="flex flex-col gap-4">
                    {history.map((item, i) => (
                      <p key={i} className="text-gray-800 text-lg leading-relaxed">
                        {item.original_text}
                      </p>
                    ))}
                    {transcription && !history.includes(transcription) && (
                      <p className="text-gray-800 text-lg leading-relaxed animate-pulse">
                        {transcription.original_text}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* English Translation */}
            <div className="flex flex-col bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
              <div className="flex items-center gap-2 px-6 py-4 border-b border-gray-50 bg-blue-50/30">
                <Globe className="w-4 h-4 text-blue-500" />
                <h2 className="font-medium text-gray-700">English Translation</h2>
              </div>
              <div className="flex-1 p-6 overflow-y-auto bg-blue-50/10">
                {history.length === 0 && !transcription ? (
                  <p className="text-gray-400 italic">Waiting for speech...</p>
                ) : (
                  <div className="flex flex-col gap-4">
                    {history.map((item, i) => (
                      <p key={i} className="text-gray-800 text-lg leading-relaxed">
                        {item.english_translation}
                      </p>
                    ))}
                    {transcription && !history.includes(transcription) && (
                      <p className="text-gray-800 text-lg leading-relaxed animate-pulse">
                        {transcription.english_translation}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>

          </div>

          {/* JSON Output View (for debugging/requirements) */}
          <div className="bg-gray-900 rounded-2xl p-6 shadow-sm overflow-x-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-gray-400 text-sm font-mono uppercase tracking-wider">Live JSON Output</h3>
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-green-500"></div>
              </div>
            </div>
            <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap">
              {transcription ? JSON.stringify(transcription, null, 2) : '{\n  "status": "waiting"\n}'}
            </pre>
          </div>

        </main>
      </div>
    </div>
  );
}
