'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

export default function Chat() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content }),
      });

      if (!response.ok) throw new Error('Failed to fetch from backend');

      const data = await response.json();
      setMessages((prev) => [...prev, { role: 'assistant', content: data.response }]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Error connecting to the AI Persona.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto p-4 bg-gray-50">
      <div className="flex-1 overflow-y-auto mb-4 border rounded-lg p-4 bg-white shadow-sm">
        {messages.map((m, index) => (
          <div key={index} className={`mb-4 ${m.role === 'user' ? 'text-right' : 'text-left'}`}>
            <span className={`inline-block p-3 rounded-lg ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-black'}`}>
                <ReactMarkdown>{m.content}</ReactMarkdown>
              </span>
          </div>
        ))}
        {isLoading && <div className="text-gray-500 italic text-left">Sanchari's AI is thinking...</div>}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2" suppressHydrationWarning>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask Sanchari about her experience, projects, or book a meeting..."
          className="flex-1 p-3 border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white text-black placeholder-gray-500"
          suppressHydrationWarning
        />
        <button type="submit" disabled={isLoading} className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400 font-medium" suppressHydrationWarning>
          Send
        </button>
      </form>
    </div>
  );
}