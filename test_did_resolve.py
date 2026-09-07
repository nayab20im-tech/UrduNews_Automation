import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

import video_production.config as vc
from video_production.avatar_generator import AvatarGenerator
from video_production.lip_sync import LipSyncEngine

print("DID_API_KEY from vc:", repr(vc.DID_API_KEY[:10] if vc.DID_API_KEY else ""))

ag = AvatarGenerator(backend="auto")
print("Avatar backend:", ag.resolve_backend())

ls = LipSyncEngine(backend="auto")
print("LipSync backend:", ls.resolve_backend())
