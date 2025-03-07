
## Quick Start

### First, start the bot server:

1. Navigate to the server directory:
   ```bash
   cd server
   ```
2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy env.example to .env and configure:
   - Add your API keys
   - Choose your bot implementation:
     ```ini
     BOT_IMPLEMENTATION=      # Options: 'openai' (default) or 'gemini'
     ```
5. Start the server:
   ```bash
   python server.py
   ```
6. Use NGROK for IOS or Android Implementation:
   ```bash
   ngrok http 7860
   ```
7. To start demo web server of ionic visit cas-ai-ionic-test

## Project Structure

```
cas-ai-backend/
├── conversational-ai-server/              # Bot server implementation
│   ├── bot-openai.py    # OpenAI bot implementation
│   ├── bot-gemini.py    # Gemini bot implementation
│   ├── runner.py        # Server runner utilities
│   ├── server.py        # FastAPI server
│   └── requirements.txt
```

## Important Note

The bot server must be running for any of the client implementations to work. Start the server first before trying any of the client apps.
