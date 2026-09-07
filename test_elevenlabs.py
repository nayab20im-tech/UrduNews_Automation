from tts.tts_engine import TTSEngine
from pathlib import Path
temp_dir = Path("DemoOutput/tts_output")
temp_dir.mkdir(parents=True, exist_ok=True)
engine = TTSEngine(audio_dir=temp_dir)
print(f"Backend resolved: {engine.resolve_backend()}")
result = engine.synthesize(999, "یہ ایک تجربہ ہے۔")
print(f"Synthesis status: {result.status}, Backend used: {result.backend_used}, Path: {result.audio_path}")
