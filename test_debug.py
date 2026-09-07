import logging
logging.basicConfig(level=logging.DEBUG)

import os
from elevenlabs.client import ElevenLabs
import tempfile
from pathlib import Path
import shutil
import subprocess

api_key = os.getenv("ELEVENLABS_API_KEY", "")
print("API KEY length:", len(api_key))

client = ElevenLabs(api_key=api_key)
text = "یہ ایک تجربہ ہے۔"

try:
    audio_gen = client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM", # Valid voice id 
        output_format="mp3_44100_128",
        text=text,
        model_id="eleven_multilingual_v2",
    )
    mp3_tmp = Path(tempfile.mktemp(suffix=".mp3"))
    with open(mp3_tmp, "wb") as f:
        for chunk in audio_gen:
            if chunk:
                f.write(chunk)
    
    print("MP3 size:", mp3_tmp.stat().st_size)
    ffmpeg = shutil.which("ffmpeg")
    print("FFMPEG:", ffmpeg)
    
    out_path = Path("test_out.wav")
    if ffmpeg:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(mp3_tmp), "-ar", "22050", "-ac", "1", str(out_path)],
            capture_output=True, timeout=60
        )
        print("FFMPEG OUT:", proc.stdout)
        print("FFMPEG ERR:", proc.stderr)
        print("RETURN CODE:", proc.returncode)
    else:
        mp3_tmp.rename(out_path)
    
    print("EXISTS:", out_path.exists())
except Exception as e:
    import traceback
    traceback.print_exc()
