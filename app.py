
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Mini.AI — local Ollama connection, no API key
HOST = "127.0.0.1"
PORT = 8000
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:e2b"

BASE_DIR = Path(__file__).resolve().parent

ERROR_MESSAGES = {
    "hy": "Չհաջողվեց միանալ Mini AI-ի հետ ⚠️",
    "en": "Couldn't connect to Mini AI ⚠️",
    "ru": "Не удалось подключиться к Mini AI ⚠️",
}


def get_language(data):
    language = str(
        data.get("language", data.get("lang", "hy"))
    ).lower()

    if language.startswith("en"):
        return "en"
    if language.startswith("ru"):
        return "ru"
    return "hy"


class MiniAIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            directory=str(BASE_DIR),
            **kwargs
        )

    def send_json(self, data, status=200):
        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/chat":
            self.send_json(
                {"error": "Not found"},
                404
            )
            return

        try:
            length = int(
                self.headers.get("Content-Length", "0")
            )

            if length <= 0 or length > 1_000_000:
                self.send_json({
                    "response": ERROR_MESSAGES["hy"],
                    "reply": ERROR_MESSAGES["hy"],
                    "message": ERROR_MESSAGES["hy"],
                    "error": True
                })
                return

            data = json.loads(
                self.rfile.read(length).decode("utf-8")
            )

            language = get_language(data)
            error_message = ERROR_MESSAGES[language]

            # Accept common frontend message field names.
            user_message = str(
                data.get(
                    "message",
                    data.get("prompt", data.get("text", ""))
                )
            ).strip()

            if not user_message:
                self.send_json({
                    "response": "Գրիր հաղորդագրություն։" if language == "hy"
                    else "Please enter a message." if language == "en"
                    else "Напишите сообщение.",
                    "reply": "Գրիր հաղորդագրություն։" if language == "hy"
                    else "Please enter a message." if language == "en"
                    else "Напишите сообщение.",
                    "error": False
                })
                return

            # Use recent conversation history when provided.
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Mini.AI, a helpful AI assistant. "
                        "Reply in the same language as the user's latest message. "
                        "Be clear, friendly, and useful."
                    )
                }
            ]

            history = data.get("history", [])
            if isinstance(history, list):
                for item in history[-20:]:
                    if not isinstance(item, dict):
                        continue

                    role = item.get("role")
                    content = item.get(
                        "content",
                        item.get("message", "")
                    )

                    if role in ("user", "assistant") and isinstance(content, str):
                        if content.strip():
                            messages.append({
                                "role": role,
                                "content": content[:12000]
                            })

            # Avoid duplicating the current message if it is already
            # included at the end of the submitted history.
            if not (
                messages[-1]["role"] == "user"
                and messages[-1]["content"] == user_message
            ):
                messages.append({
                    "role": "user",
                    "content": user_message[:12000]
                })

            payload = json.dumps({
                "model": MODEL,
                "messages": messages,
                "stream": False
            }).encode("utf-8")

            request = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            with urllib.request.urlopen(
                request,
                timeout=180
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            answer = (
                result.get("message", {}).get("content", "")
            ).strip()

            if not answer:
                raise RuntimeError("Empty response from Ollama")

            self.send_json({
                "response": answer,
                "reply": answer,
                "message": answer,
                "error": False
            })

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            RuntimeError
        ):
            language = "hy"

            try:
                language = get_language(data)
            except (NameError, TypeError):
                pass

            message = ERROR_MESSAGES[language]
            self.send_json({
                "response": message,
                "reply": message,
                "message": message,
                "error": True
            })

        except Exception:
            message = ERROR_MESSAGES.get(
                get_language(data) if "data" in locals() else "hy",
                ERROR_MESSAGES["hy"]
            )
            self.send_json({
                "response": message,
                "reply": message,
                "message": message,
                "error": True
            })

    def log_message(self, format, *args):
        # Keep normal requests quiet in the terminal.
        pass


if __name__ == "__main__":
    os.chdir(BASE_DIR)

    server = ThreadingHTTPServer(
        (HOST, PORT),
        MiniAIHandler
    )

    print("=" * 42)
    print("Mini.AI is running!")
    print(f"Open: http://localhost:{PORT}")
    print(f"Model: {MODEL}")
    print("API keys: not required")
    print("Stop: Ctrl+C")
    print("=" * 42)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMini.AI stopped.")
    finally:
        server.server_close()
