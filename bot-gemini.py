#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Speech-to-speech model

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""

import asyncio
import os
import sys
import argparse

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIBotTranscriptionProcessor,
    RTVIConfig,
    RTVIMetricsProcessor,
    RTVIProcessor,
    RTVISpeakingProcessor,
    RTVIUserTranscriptionProcessor,
)
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
import boto3
import docx
from io import BytesIO
import requests





load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument("-u", "--url", type=str, required=True, help="URL of the Daily room to join")
    parser.add_argument("-t", "--token", type=str, required=True, help="Access token for the Daily room")
    parser.add_argument("--product_id", type=str, required=True, help="Product ID for the session")
    return parser.parse_args()

async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport with specific audio parameters
    - Gemini Live multimodal model integration
    - Voice activity detection
    - Animation processing
    - RTVI event handling
    """
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)
        args = parse_arguments()
        room_url = args.url
        token = args.token
        product_id = args.product_id
        logger.info(f"ALL INPUT ARGUMENTSRoom URL: {room_url}, Token: {token}, Product ID: {product_id}")


        # Set up Daily transport with specific audio/video parameters for Gemini
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        # Initialize the Gemini Multimodal Live model
        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
            transcribe_user_audio=True,
            transcribe_model_audio=True,
        )

        def extract_docx_from_url(url):
            try:
                if url.startswith("https://") or url.startswith("http://"):
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an error for failed requests
                    
                    file_stream = BytesIO(response.content)
                elif url.startswith("s3://"):
                    import boto3
                    # Parse S3 URL
                    parts = url[5:].split("/", 1)
                    bucket_name, object_key = parts[0], parts[1]

                    # Initialize S3 client
                    s3 = boto3.client('s3')
                    response = s3.get_object(Bucket=bucket_name, Key=object_key)
                    file_stream = BytesIO(response['Body'].read())
                else:
                    raise ValueError("Invalid URL format")

                # Read the content of the Word file
                doc = docx.Document(file_stream)
                text = "\n".join([para.text for para in doc.paragraphs])
                return text

            except Exception as e:
                logger.error(f"Error extracting text from URL: {url}, Error: {str(e)}")
                return None

        # Example usage
        s3_link = f"https://vizaro2.s3.us-east-1.amazonaws.com/Products/{product_id}/{product_id}.docx"
        logger.info(f"Extracting text from URL: {s3_link}")
        product_description = extract_docx_from_url(s3_link)
        logger.info(f"Extracted text: {product_description}")

        messages = [
            {
                "role": "user",
                "content": f"""                      

                            Identity
                            A friendly and persuasive male AI sales assistant helping customers choose the best products from our collection.

                            Greeting
                            "Hello, welcome to Vizaro!" -- Then start with the product specific greeting directly without break. (this line will be product specific, a catchy line that sales person says. Take help of examples below to make one for the product in the product discreption.)

                            Context
                            Focused on assisting, recommending, and persuading customers by highlighting product features and benefits in a natural and engaging way.

                            Job Responsibilities
                            Greet customers warmly and engage in a friendly, helpful conversation.
                            Provide concise and compelling product details, focusing on why it’s a great choice.
                            Assist in decision-making by asking relevant questions and providing comparisons if needed.
                            Encourage the customer to make a purchase by emphasizing value, uniqueness, or limited availability.
                            Try to convence the customer to buy the product by emphasizing value, uniqueness, or limited availability.
                            
                            Sales Handling Process
                            Understand the customer's needs by asking engaging questions.
                            Respond dynamically using the provided product description, highlighting the best-selling points.
                            Handle objections smoothly by emphasizing product benefits.
                            If the customer is unsure, suggest complementary products or create urgency (e.g., "Limited stock available!").
                            Finalize the conversation with a call to action:
                            If they show interest: "Shall I add this to your cart?"
                            If they hesitate: "Take your time, but don’t wait too long—we have limited stock!"
                            Reply to each question dynamically based on the product description. Behave like a friendly in-store assistant.

                            Tone & Style
                            Warm, engaging, and persuasive, like a friendly in-store assistant.
                            Maintain a professional yet casual tone, making the customer feel comfortable.
                            Avoid overwhelming the customer with too much information—focus on the most compelling aspects.
                            
                            Response Guidelines
                            Keep responses concise and engaging (1 -2 line max). Not too long, not too short that looks incoplete.
                            Always highlight the key selling points dynamically based on the product description.
                            Encourage interaction and guide the customer toward a purchase decision.
                            Assume information if not provided. 
                            Language: Indian Easy English. currency: Rupees
                            After this prompt, do not reply any think, directly start with the greeting.

                            Important Information:
                            STRICTLY STICK TO THE INFORMATION PROVIDED IN THE PRODUCT DESCRIPTION. DO NOT MAKE UP ANYTHING. ONLY ASSUME INFORMATION IF NOT PROVIDED IN THE PRODUCT DESCRIPTION.



                            Example Response (for Abyssal Planter Pot)
                            "This stunning lotus-inspired planter isn’t just a pot—it’s a statement piece! Its sculptural elegance adds a modern touch to any space, and it’s made from eco-friendly PLA plastic. Plus, it's lightweight yet sturdy—perfect for both indoors and outdoors. Would you like me to add one to your cart today?"
                            Examples for the greeting:
                            For a Yodi Watering Can: "Spotted our stylish watering can? It’s not just for plants—it’s a design upgrade for your garden! Let me tell you why plant lovers adore it.
                            "Product description": {product_description}

                            """,
                "Product description": product_description
            },
        ]

        logger.info(f"Messages: {messages}")

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        ta = TalkingAnimation()

        #
        # RTVI events for Pipecat client UI
        #

        # This will send `user-*-speaking` and `bot-*-speaking` messages.
        rtvi_speaking = RTVISpeakingProcessor()

        # This will emit UserTranscript events.
        rtvi_user_transcription = RTVIUserTranscriptionProcessor()

        # This will emit BotTranscript events.
        rtvi_bot_transcription = RTVIBotTranscriptionProcessor()

        # This will send `metrics` messages.
        rtvi_metrics = RTVIMetricsProcessor()

        # Handles RTVI messages from the client
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                context_aggregator.user(),
                llm,
                rtvi_speaking,
                rtvi_user_transcription,
                rtvi_bot_transcription,
                ta,
                rtvi_metrics,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        await task.queue_frame(quiet_frame)

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
