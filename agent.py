import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import noise_cancellation, silero, sarvam, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import llm,stt,tts,inference
from liovekit.agents import AgentStateChangedEvent, MertricsCollectedEvent, metrics
import os

load_dotenv()
logger = logging.getLogger(__name__)

class Assistant(Agent):
    def __init__(self)-> None:
        super().__init__(
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support. "
                "Help the caller fix issues without rambling, and keep replies under 3 sentences."
            ),
        )


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=sarvam.STT(
            language="unknown",
            model="saaras:v3",
            mode="transcribe",
        ),

        llm=llm.FallbackAdapter(
            [
                openai.LLM( 
                    model="llama-3.1-8b-instant",
                    api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.3,
                ),
                openai.LLM(  
                    model="gemma2-9b-it",
                    api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.3,
                ),
            ]
        ),
        tts=sarvam.TTS(  
            target_language_code="en-IN",
            model="bulbul:v3",
            speaker="ritu",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation = True
    )

    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MertricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await ctx.connect()

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))