import httpx


class OllamaGenerator:
    def __init__(
        self,
        model: str = "qwen3:14b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120.0,
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"].strip()
