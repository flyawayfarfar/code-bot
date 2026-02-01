import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

// Icons
const SendIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
)

const SettingsIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
)

const BotIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
  </svg>
)

const UserIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
  </svg>
)

const SpinnerIcon = () => (
  <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
  </svg>
)

// Message component
function Message({ message, isUser }) {
  const markdownComponents = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '')
      return !inline && match ? (
        <SyntaxHighlighter
          style={oneDark}
          language={match[1]}
          PreTag="div"
          className="rounded-lg my-2 text-sm"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className="bg-gray-700 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
          {children}
        </code>
      )
    },
    p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>,
    ul: ({ children }) => <ul className="list-disc list-inside mb-3 pl-4 space-y-1">{children}</ul>,
    ol: ({ children }) => <ol className="list-decimal list-inside mb-3 pl-4 space-y-1">{children}</ol>,
    li: ({ children }) => <li className="pl-2 break-words">{children}</li>,
    h1: ({ children }) => <h1 className="text-2xl font-bold mb-3 mt-5">{children}</h1>,
    h2: ({ children }) => <h2 className="text-xl font-bold mb-3 mt-4">{children}</h2>,
    h3: ({ children }) => <h3 className="text-lg font-semibold mb-2 mt-3">{children}</h3>,
    a: ({ children, href }) => (
      <a href={href} className="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ),
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-gray-500 pl-4 italic my-3 text-gray-300">{children}</blockquote>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto my-3">
        <table className="min-w-full border border-gray-600 divide-y divide-gray-600">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-gray-700">{children}</thead>,
    th: ({ children }) => <th className="border border-gray-600 px-4 py-2 text-left font-semibold">{children}</th>,
    td: ({ children }) => <td className="border border-gray-600 px-4 py-2">{children}</td>,
  }

  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center self-start">
          <BotIcon />
        </div>
      )}
      <div className={`max-w-[85%] rounded-2xl px-5 py-3.5 ${
        isUser 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-800 text-gray-100'
      }`}>
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-invert max-w-none text-gray-200">
            <ReactMarkdown components={markdownComponents}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {message.toolsUsed && message.toolsUsed.length > 0 && (
          <div className="mt-3 pt-2 border-t border-gray-700 text-xs text-gray-400">
            <span className="font-medium">Tools used:</span> {message.toolsUsed.join(', ')}
          </div>
        )}
      </div>
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center self-start">
          <UserIcon />
        </div>
      )}
    </div>
  )
}

// Settings Panel
function SettingsPanel({ settings, onSettingsChange, onClose }) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md shadow-xl" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold text-white">Settings</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Provider</label>
            <select
              value={settings.provider}
              onChange={(e) => onSettingsChange({ ...settings, provider: e.target.value })}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="google">Google AI</option>
              <option value="openai">OpenAI</option>
              <option value="ollama">Ollama</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">API URL</label>
            <input
              type="text"
              value={settings.apiUrl}
              onChange={(e) => onSettingsChange({ ...settings, apiUrl: e.target.value })}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="http://localhost:8000"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <label className="block text-sm font-medium text-gray-300">Conversation Mode</label>
              <p className="text-xs text-gray-500">Enables memory and context awareness</p>
            </div>
            <button
              onClick={() => onSettingsChange({ ...settings, conversationMode: !settings.conversationMode })}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.conversationMode ? 'bg-blue-600' : 'bg-gray-600'
              }`}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                settings.conversationMode ? 'translate-x-6' : 'translate-x-1'
              }`} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// Main App
function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [conversationId, setConversationId] = useState(null)
  const [settings, setSettings] = useState({
    provider: 'google',
    apiUrl: 'http://localhost:8000',
    conversationMode: false
  })
  
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input.trim() }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const endpoint = settings.conversationMode ? '/conversation' : '/chat'
      const body = settings.conversationMode
        ? { 
            query: userMessage.content, 
            conversation_id: conversationId 
          }
        : { 
            query: userMessage.content, 
            provider: settings.provider 
          }

      const response = await fetch(`${settings.apiUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      if (!response.ok) throw new Error('Failed to get response')

      const data = await response.json()
      
      if (settings.conversationMode && data.conversation_id) {
        setConversationId(data.conversation_id)
      }

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        toolsUsed: data.tools_used || [],
        reasoning: data.reasoning
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please check your connection and try again.'
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewChat = () => {
    setMessages([])
    setConversationId(null)
    inputRef.current?.focus()
  }

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <BotIcon />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-white">Code Assistant</h1>
              {settings.conversationMode && conversationId && (
                <p className="text-xs text-gray-400">Session active</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Mode Toggle */}
            <div className="flex items-center bg-gray-700 rounded-lg p-0.5">
              <button
                onClick={() => setSettings(s => ({ ...s, conversationMode: false }))}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  !settings.conversationMode 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                Basic
              </button>
              <button
                onClick={() => setSettings(s => ({ ...s, conversationMode: true }))}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  settings.conversationMode 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                Agent
              </button>
            </div>
            <button
              onClick={handleNewChat}
              className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
            >
              New Chat
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              title="Settings"
            >
              <SettingsIcon />
            </button>
          </div>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          {messages.length === 0 ? (
            <div className="text-center py-20">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <BotIcon />
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">How can I help you today?</h2>
              <p className="text-gray-400 max-w-md mx-auto">
                Ask me anything about your codebase. I'll search through your documents and provide helpful answers.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, idx) => (
                <Message key={idx} message={msg} isUser={msg.role === 'user'} />
              ))}
              {isLoading && (
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <BotIcon />
                  </div>
                  <div className="bg-gray-800 rounded-2xl px-4 py-3 flex items-center gap-2 text-gray-400">
                    <SpinnerIcon />
                    <span>Thinking...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      {/* Input */}
      <footer className="bg-gray-800 border-t border-gray-700 px-4 py-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about your codebase..."
              disabled={isLoading}
              className="flex-1 bg-gray-700 border border-gray-600 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-xl transition-colors flex items-center gap-2"
            >
              {isLoading ? <SpinnerIcon /> : <SendIcon />}
              <span className="hidden sm:inline">Send</span>
            </button>
          </div>
          <div className="mt-2 flex items-center justify-center gap-4 text-xs text-gray-500">
            <span>Provider: {settings.provider}</span>
            <span>•</span>
            <span>{settings.conversationMode ? 'Conversation mode' : 'Single query mode'}</span>
          </div>
        </form>
      </footer>

      {/* Settings Modal */}
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onSettingsChange={setSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  )
}

export default App
