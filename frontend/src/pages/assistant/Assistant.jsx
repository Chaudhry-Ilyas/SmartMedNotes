import React, { useState } from "react";
import { FiSend, FiMic } from "react-icons/fi";
import axios from "axios";

function Assistant() {
  const [query, setQuery] = useState("");
  const [conversation, setConversation] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [chatStarted, setChatStarted] = useState(false);
  const [loading, setLoading] = useState(false);

  const baseURL = "https://mabdullah03-smart-med-notes.hf.space/rag";


  const handleSend = async () => {
    if (!query.trim() || loading) return;

    setChatStarted(true);
    setLoading(true);

    try {
      setConversation((prev) => [...prev, { type: "user", text: query }]);

      const res = await axios.post(`${baseURL}`, { query }, {
        headers: { "Content-Type": "application/json" },
      });

      setConversation((prev) => [
        ...prev,
        { type: "assistant", text: res.data.response }
      ]);
    } catch (error) {
      console.error("Error fetching response:", error);
      setConversation((prev) => [
        ...prev,
        { type: "assistant", text: "Sorry, I'm having trouble responding. Please try again." },
      ]);
    }
    setLoading(false);
    setQuery("");
  };

  const startListening = () => {
    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = "en-US";
    recognition.start();
    setIsListening(true);

    recognition.onresult = (event) => {
      setQuery(event.results[0][0].transcript);
      setIsListening(false);
    };

    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 p-4">
      {!chatStarted && (
        <div className="flex justify-center items-center h-2/3">
          <h1 className="text-5xl font-bold text-black">Your Personal Assistant</h1>
        </div>
      )}

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {conversation.map((msg, index) => (
          <div
            key={index}
            className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`p-3 rounded-lg max-w-xs ${
                msg.type === "user"
                  ? "bg-blue-500 text-white self-end"
                  : "bg-gray-200 text-black self-start"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="flex justify-start">
            <div className="p-3 bg-gray-300 text-black rounded-lg max-w-xs animate-pulse">
              Generating response...
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center p-4 bg-white shadow-lg rounded-lg mb-4">
        <button
          className="mr-2 p-3 bg-gray-300 rounded-full"
          onClick={startListening}
          disabled={loading}
        >
          <FiMic className="w-5 h-5" />
        </button>
        <input
          type="text"
          className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500"
          placeholder="e.g. How to cure a fracture?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={loading}
        />
        <button
          className="ml-3 bg-blue-500 text-white p-3 rounded-full"
          onClick={handleSend}
          disabled={loading}
        >
          <FiSend className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}

export default Assistant;
