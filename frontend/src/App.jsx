import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Ask me anything about your indexed vehicle documents.",
      citations: [],
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [engine, setEngine] = useState("native");
  const [vectorStore, setVectorStore] = useState("faiss");
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [showScrollButton, setShowScrollButton] = useState(false);

  const messagesRef = useRef(null);
  const bottomRef = useRef(null);

  function scrollToBottom(behavior = "smooth") {
    bottomRef.current?.scrollIntoView({
      behavior,
    });
  }

  useEffect(() => {
    scrollToBottom("smooth");
  }, [messages, loading]);

  function handleScroll() {
    const element = messagesRef.current;

    if (!element) return;

    const distanceFromBottom =
      element.scrollHeight -
      element.scrollTop -
      element.clientHeight;

    setShowScrollButton(distanceFromBottom > 120);
  }

  async function sendMessage() {
    const question = input.trim();

    if (!question || loading) {
      return;
    }

    const userMessage = {
      role: "user",
      text: question,
      citations: [],
    };

    setMessages((current) => [
      ...current,
      userMessage,
    ]);

    setInput("");
    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/ask`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question,
            top_k: 5,
            engine,
            vector_store: vectorStore,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `API error ${response.status}`
        );
      }

      const data = await response.json();

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: data.answer,
          citations: data.citations ?? [],
        },
      ]);
    } catch (error) {
      console.error(error);

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text:
            "Something went wrong while contacting AutoSpec.",
          citations: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      sendMessage();
    }
  }

  return (
    <main className="app">
      <section className="chat-shell">
        <header className="header">
          <div>
            <h1>AutoSpec</h1>
            <span>
              Vehicle document assistant
            </span>
          </div>

          <div className="settings-wrapper">
            <button
              className="settings-button"
              onClick={() =>
                setSettingsOpen(
                  (current) => !current
                )
              }
            >
              ⚙
            </button>

            {settingsOpen && (
              <div className="settings-panel">
                <label>
                  Engine

                  <select
                    value={engine}
                    onChange={(event) =>
                      setEngine(
                        event.target.value
                      )
                    }
                  >
                    <option value="native">
                      Native
                    </option>

                    <option value="langchain">
                      LangChain
                    </option>

                    <option value="llamaindex">
                      LlamaIndex
                    </option>
                  </select>
                </label>

                <label>
                  Vector store

                  <select
                    value={vectorStore}
                    onChange={(event) =>
                      setVectorStore(
                        event.target.value
                      )
                    }
                  >
                    <option value="faiss">
                      FAISS
                    </option>

                    <option value="chroma">
                      Chroma
                    </option>
                  </select>
                </label>

                <div className="engine-info">
                  {engine} · {vectorStore}
                </div>
              </div>
            )}
          </div>
        </header>

        <div
          className="messages"
          ref={messagesRef}
          onScroll={handleScroll}
        >
          {messages.map(
            (message, index) => (
              <div
                key={index}
                className={`message-row ${message.role}`}
              >
                <div className="bubble">
                  <p>{message.text}</p>

                  {message.citations.length >
                    0 && (
                    <div className="citations">
                      {message.citations.map(
                        (
                          citation,
                          citationIndex
                        ) => (
                          <span
                            key={`${citation.source}-${citation.page}-${citationIndex}`}
                          >
                            {
                              citation.source
                            }{" "}
                            · p.
                            {
                              citation.page
                            }
                          </span>
                        )
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          )}

          {loading && (
            <div className="message-row assistant">
              <div className="bubble typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {showScrollButton && (
          <button
            className="scroll-button"
            onClick={() =>
              scrollToBottom()
            }
          >
            ↓
          </button>
        )}

        <div className="composer-area">
          <div className="composer">
            <textarea
              value={input}
              onChange={(event) =>
                setInput(
                  event.target.value
                )
              }
              onKeyDown={handleKeyDown}
              placeholder="Ask about your vehicle documents..."
              rows={1}
            />

            <button
              className="send-button"
              onClick={sendMessage}
              disabled={
                loading ||
                !input.trim()
              }
            >
              ↑
            </button>
          </div>

          <span className="hint">
            Enter to send · Shift + Enter
            for new line
          </span>
        </div>
      </section>
    </main>
  );
}

export default App;