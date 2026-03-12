import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, room_io
from livekit.plugins import noise_cancellation, silero, sarvam, groq, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import llm, RunContext, mcp
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
from livekit.agents.llm import function_tool
import os
import time
import httpx

load_dotenv()
logger = logging.getLogger(__name__)

class Assistant(Agent):
    def __init__(self)-> None:
        super().__init__(
            instructions=(
                "You are a concise voice AI for tech support. "
                "Keep replies under 3 sentences. "
                "Use lookup_weather for current weather questions. "
                "For LiveKit documentation questions, use get_docs_overview for browsing, "
                "use docs_search to find relevant pages, and use get_pages only after search "
                "when you need the full page content before answering."
            ),
        )

    @function_tool()
    async def lookup_weather(self,context:RunContext,location:str)-> dict:
        """Looks up the current weather information for the given location."""
        logger.info("Looking up weather for %s",location)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                geo_response = await client.get('https://geocoding-api.open-meteo.com/v1/search',
                                                params={'name': location, 'count': 1})
                geo_response.raise_for_status()
                geo_data = geo_response.json()
                if not geo_data.get('results'):
                    return {"error": "Location not found"}
                lat = geo_data['results'][0]['latitude']
                lon = geo_data['results'][0]['longitude']
                place_name = geo_data['results'][0]['name']

                weather_response = await client.get("https://api.open-meteo.com/v1/forecast",
                                                        params={
                                                            'latitude': lat,
                                                            'longitude': lon,
                                                            'current': 'temperature_2m,weather_code',
                                                            'temperature_unit': 'celsius',
                                                        },
                                                        )
                weather_response.raise_for_status()
                weather_data = weather_response.json().get('current', {})
                return {
                    "location": place_name,
                    "temperature": weather_data.get('temperature_2m'),
                    "weather_code": weather_data.get('weather_code'),
                }
        except Exception as e:
            logger.error("Error looking up weather: %s", e)
            return {"error": "Failed to retrieve weather information"}


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=sarvam.STT(
            language="unknown",
            model="saaras:v3",
            mode="transcribe",
        ),

        llm=llm.FallbackAdapter(
            [
                google.LLM(
                    model="gemini-2.5-flash-lite",
                    temperature=0.3,
                ),
                groq.LLM(
                    model="groq/compound",
                    temperature=0.3,
                ),
                groq.LLM(
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                ),
            ],attempt_timeout=15
        ),
        tts=sarvam.TTS(  
            target_language_code="en-IN",
            model="bulbul:v3",
            speaker="ritu",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation = True,
        mcp_servers=[
            mcp.MCPServerHTTP(
                url="https://docs.livekit.io/mcp/",
                transport_type="streamable_http",
                allowed_tools=["get_docs_overview", "docs_search", "get_pages"],
                timeout=15,
                client_session_timeout_seconds=15,
            ),
        ],
    )

    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == 'speaking':
            if last_eou_metrics:
                elapsed = time.time() - last_eou_metrics.timestamp
                logger.info(f"Time to first Audio: {elapsed:.2f}s")

    await ctx.connect()

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )
    


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))