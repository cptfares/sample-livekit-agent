# agent.py

from dotenv import load_dotenv

import json
import os
import aiohttp
from typing import Annotated, Any
from livekit import api
from livekit.agents import function_tool, RunContext

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    google,
    cartesia,
    rime,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.english import EnglishModel

# Load environment variables
load_dotenv()

# Defines the core behavior and capabilities of our voice assistant.
class Assistant(Agent):
    # The constructor initialises the agent with a set of instructions
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant. You can check the weather in a given location.")
    
    # This method is a tool that the agent can use to get the current weather.
    # The @function_tool decorator exposes this method to the agent's LLM,
    # allowing it to be called when the user asks for the weather.
    @function_tool()
    async def get_weather(
        self,
        context: RunContext,
        location: Annotated[
            str, "The city and state, e.g. San Francisco, CA"
        ],
    ) -> str:
        """Get the current weather in a given location"""
        api_key = os.environ.get("OPENWEATHER_API_KEY")
        if not api_key:
            return "OpenWeather API key is not set."

        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"Sorry, I couldn't get the weather. Status code: {response.status}"
                
                data = await response.json()
                
                if "weather" not in data or not data["weather"]:
                    return "Sorry, I couldn't find any weather data for that location."
                
                description = data["weather"][0]["description"]
                temp = data["main"]["temp"]
                
                return f"The weather in {location} is {description} with a temperature of {temp}°C."

# The entrypoint is the main function that runs when a new job for the agent starts.
# It sets up the agent's connection to a LiveKit room and manages its lifecycle.
async def entrypoint(ctx: agents.JobContext):
    # Connect the agent to the LiveKit room associated with the job.
    await ctx.connect()

    # This block attempts to start a recording (egress) of the room's audio.
    # The recording is saved to an S3 bucket.
    try:
        lkapi = api.LiveKitAPI()

        req = api.RoomCompositeEgressRequest(
            room_name=ctx.room.name,
            audio_only=True,
            file_outputs=[
                api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=f"{ctx.room.name}.ogg",
                    # S3 configuration for uploading the recording.
                    s3=api.S3Upload(
                        access_key=os.environ.get("AWS_S3_ACCESS_KEY"),
                        secret=os.environ.get("AWS_S3_SECRET_KEY"),
                        region="eu-north-1",
                        bucket="livekit-calls"
                    )
                )
            ],
        )
        print("Starting room egress...")
        egress_info = await lkapi.egress.start_room_composite_egress(req)
        await lkapi.aclose()
        egress_id = getattr(egress_info, "egress_id", None) or getattr(egress_info, "egressId", None)
        print(f"Egress started successfully. Egress ID: {egress_id}")
    except Exception as e:
        print(f"Error starting egress: {e}")

    # Check for a phone number in the job metadata to determine if this is an outbound call.
    phone_number = None
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
            phone_number = metadata.get("phone_number")
        except json.JSONDecodeError:
            print("Error: Invalid JSON in job metadata")

    # If a phone number is provided, initiate an outbound SIP call.
    if phone_number:
        print(f"Attempting to place outbound call to: {phone_number}")
        try:
            # Use the LiveKit API to create a new SIP participant, effectively making a call.
            await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id='ST_p7QcFHJKXXUM', # The specific SIP trunk to use.
                sip_call_to=phone_number,
                participant_identity=phone_number, # Identity for the participant in the room.
                wait_until_answered=True, # Wait for the call to be answered before proceeding.
            ))
            print(f"Call to {phone_number} was answered.")
        except api.TwirpError as e:
            # Handle errors during SIP call creation, like the call not being answered.
            print(f"Error creating SIP participant: {e.message}")
            await ctx.shutdown()
            return

    # Set up the agent's session with various services (plugins).
    session = AgentSession(
        stt=cartesia.STT(model="ink-whisper"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=rime.TTS(model="arcana", speaker="astra"),
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
    )

    # Start the agent session, which begins processing audio from the room.
    await session.start(
        room=ctx.room,
        agent=Assistant(), # Use the Assistant agent we defined earlier.
        room_input_options=RoomInputOptions(
            # Apply noise cancellation to the audio input.
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # If this is not an outbound call (i.e., no phone number was provided),
    # the agent should start the conversation.
    if not phone_number:
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )

# This is the main execution block. It runs the agent worker when the script is executed
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="livekit-tutorial-hugo" # A unique name for this agent worker.
    ))