# LiveKit AI Voice Agent with Twilio & AWS S3

This repository contains the code for building and deploying an AI voice agent using the LiveKit Agents 1.0 framework, integrated with Twilio for phone calls and AWS S3 for call recording storage.

The project demonstrates how to create a complete voice agent pipeline featuring Speech-to-Text (STT), Language Model (LLM), Text-to-Speech (TTS), Voice Activity Detection (VAD), Turn Taking, and Noise Reduction.

## Features

*   **LiveKit Agents 1.0:** Core pipeline for the AI voice agent.
*   **Twilio Integration:** Handle both inbound and outbound phone calls.
*   **AWS S3 Recording:** Record and store all call audio.
*   **Tool Use:** Example implementation of giving the agent access to external tools (e.g., a weather tool).
*   **Automated Twilio Setup:** A Python script (`setup_twilio_livekit.py`) to simplify Twilio SIP trunk configuration.
*   **Customizable Components:** Easily swap out STT, LLM, and TTS models.

## Prerequisites

Before you begin, you will need accounts and credentials for the following services:

*   **LiveKit Cloud:** API Key and Secret.
*   **AWS:** Account with S3 access. You'll need an IAM user with permissions (`s3:GetObject`, `s3:PutObject`, `s3:ListBucket`) for a specific S3 bucket, and their Access Key ID and Secret Access Key. You will also need your AWS region.
*   **Twilio:** Account SID, Auth Token, and an active phone number capable of handling voice calls.
*   **AI Service Providers:** API keys for your chosen STT, LLM, and TTS models (e.g., Deepgram, Grok, ElevenLabs, Cartesia, OpenAI, etc.).

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Gather Credentials:**
    Collect all the API keys and credentials listed in the Prerequisites section. Store them securely (e.g., in environment variables).

4.  **Configure AWS S3:**
    Update the AWS S3 configuration in `agent.py` (or use environment variables if preferred) with your AWS region and S3 bucket name.

5.  **Run Twilio/LiveKit Setup Script:**
    This script configures Twilio to work with LiveKit. You will be prompted to enter details:
    ```bash
    python setup_twilio_livekit.py
    ```
    Provide a base name for resources, your Twilio phone number (e.g., `+44...`), a username for SIP authentication, and a strong password (minimum 12 characters, including mixed case, numbers, and symbols). You will also need your LiveKit SIP URI from the LiveKit Cloud dashboard (Settings -> Project).

6.  **Update Agent Code:**
    After running the setup script, it will output a LiveKit outbound SIP trunk ID. Copy this value.
    Open `agent.py`, find the `create_SIP_participant` call, and replace the placeholder SIP trunk ID with the one provided by the script.
    Also, ensure the agent name defined in `agent.py` matches the name you used when running `setup_twilio_livekit.py`.

7.  **Configure Agent Models/Prompt:**
    In `agent.py`, update the code to use your preferred STT, LLM, and TTS models by providing their respective API keys and model names. Customize the agent's system prompt/instructions within the `Assistant` class.

## Running the Agent

To start the agent and have it listen for calls:

```bash
python agent.py dev