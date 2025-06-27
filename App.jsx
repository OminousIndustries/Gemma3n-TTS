import React, { useState, useEffect, useRef } from "react";

// ----------------- Icons -----------------
const IconMic = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
    <line x1="12" y1="19" x2="12" y2="22"></line>
  </svg>
);

const IconLoader = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="2" x2="12" y2="6" />
    <line x1="12" y1="18" x2="12" y2="22" />
    <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
    <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
    <line x1="2" y1="12" x2="6" y2="12" />
    <line x1="18" y1="12" x2="22" y2="12" />
    <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
    <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
  </svg>
);

// ----------------- Main Component -----------------
export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [conversation, setConversation] = useState([]);
  const [error, setError] = useState(null);
  const [isTTSEnabled, setIsTTSEnabled] = useState(true);

  // image-related state
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imagePrompt, setImagePrompt] = useState("");

  // audio recording refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // ----------------- AUDIO RECORDING -----------------
  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        await processAudio(audioBlob);
        stream.getTracks().forEach((t) => t.stop());
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setTimeout(stopRecording, 4000);
    } catch (err) {
      setError("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  };

  const blobToBase64 = (blob) =>
    new Promise((res, rej) => {
      const reader = new FileReader();
      reader.onloadend = () => res(reader.result.split(",")[1]);
      reader.onerror = rej;
      reader.readAsDataURL(blob);
    });

  const processAudio = async (audioBlob) => {
    setIsProcessing(true);
    try {
      const b64 = await blobToBase64(audioBlob);
      setConversation((p) => [...p, { role: "user", parts: [{ type: "audio" }] }]);
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: b64 }),
      });
      if (!res.ok) throw new Error(await res.text());
      const { text } = await res.json();
      addReply(text);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // ----------------- IMAGE + PROMPT -----------------
  const submitImageQuestion = async () => {
    if (!imageFile || !imagePrompt.trim()) return;
    setIsProcessing(true);
    setError(null);
    try {
      setConversation((p) => [
        ...p,
        { role: "user", parts: [{ type: "image", url: imagePreview }, { type: "text", text: imagePrompt }] },
      ]);
      const form = new FormData();
      form.append("prompt", imagePrompt);
      form.append("image", imageFile, imageFile.name);

      const res = await fetch("http://localhost:8000/ask_image", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const { text } = await res.json();
      addReply(text);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsProcessing(false);
      setImageFile(null);
      setImagePreview(null);
      setImagePrompt("");
    }
  };

  // ----------------- TTS -----------------
  const addReply = (text) => {
    setConversation((p) => [...p, { role: "model", parts: [{ text }] }]);
    if (isTTSEnabled && "speechSynthesis" in window) {
      const utter = new SpeechSynthesisUtterance(text.replace(/[-•▪●◦—–]/g, "-"));
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    }
  };

  // ----------------- Scroll to bottom -----------------
  const bottomRef = useRef(null);
  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [conversation]);

  // ----------------- RENDER -----------------
  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* header */}
      <header className="p-4 border-b border-gray-700/50 bg-gray-900/80 flex justify-between items-center">
        <h1 className="text-lg font-bold">Gemma-3n Voice Assistant</h1>
        <label className="flex items-center space-x-2">
          <span className="text-sm">TTS</span>
          <input type="checkbox" checked={isTTSEnabled} onChange={() => setIsTTSEnabled(!isTTSEnabled)} />
        </label>
      </header>

      {/* messages */}
      <main className="flex-1 overflow-y-auto p-4 space-y-6">
        {conversation.length === 0 && (
          <div className="text-center text-gray-500 pt-20">
            <IconMic className="w-16 h-16 mx-auto mb-4" />
            <p>Press the record button to start a conversation.</p>
          </div>
        )}

        {conversation.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-md p-3 rounded-2xl ${msg.role === "user" ? "bg-blue-600" : "bg-gray-700"}`}>
              {msg.parts[0].type === "audio" ? (
                <p className="italic">Audio message</p>
              ) : msg.parts[0].type === "image" ? (
                <>
                  <img src={msg.parts[0].url} alt="upload" className="max-w-xs mb-2 rounded-lg" />
                  <p>{msg.parts[1].text}</p>
                </>
              ) : (
                <p>{msg.parts[0].text}</p>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      {/* image upload + question */}
      <section className="px-4 space-y-4">
        <input
          type="file"
          accept="image/*"
          id="img-input"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files[0];
            if (!file) return;
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
          }}
        />
        <label
          htmlFor="img-input"
          className="block w-full border-2 border-dashed border-gray-600 text-center p-6 rounded-xl cursor-pointer"
        >
          {imagePreview ? (
            <img src={imagePreview} alt="preview" className="mx-auto max-h-48 rounded-lg" />
          ) : (
            "Click or drag an image here"
          )}
        </label>

        {imagePreview && (
          <>
            <textarea
              value={imagePrompt}
              onChange={(e) => setImagePrompt(e.target.value)}
              placeholder="Ask a question about the image…"
              className="w-full bg-gray-800 p-3 rounded-lg resize-none"
            />
            <button
              onClick={submitImageQuestion}
              disabled={!imagePrompt.trim() || isProcessing}
              className="w-full bg-indigo-600 hover:bg-indigo-500 py-2 rounded-lg disabled:bg-gray-500"
            >
              Ask
            </button>
          </>
        )}
      </section>

      {/* footer mic button */}
      <footer className="p-4 flex flex-col items-center">
        {error && <p className="text-red-400 mb-2">{error}</p>}
        <button
          onClick={startRecording}
          disabled={isRecording || isProcessing}
          className="w-20 h-20 bg-indigo-600 rounded-full flex items-center justify-center disabled:bg-gray-500"
        >
          {isRecording ? <div className="w-8 h-8 bg-red-500 rounded animate-pulse" /> : isProcessing ? <IconLoader className="w-8 h-8 animate-spin" /> : <IconMic className="w-8 h-8" />}
        </button>
        <p className="text-xs text-gray-500 mt-2">
          {isRecording ? "Recording…" : isProcessing ? "Processing…" : "Tap to speak (4 s)"}
        </p>
      </footer>
    </div>
  );
}
