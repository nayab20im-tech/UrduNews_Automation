"""
video_production/synchronizer.py
================================
Calculates precise timing for visual cuts based on the script text length 
and total audio duration, allowing dynamic PIP image changes during a single combined video.
"""

from typing import List, Dict, Any
from data_acquisition.utils.logger import get_logger

logger = get_logger("video_production.synchronizer")

class TimelineSynchronizer:
    def __init__(self):
        pass
        
    def generate_timeline(
        self, 
        full_script: str, 
        segments: List[Dict[str, Any]], 
        total_audio_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Maps each segment's text to a start and end time in the audio.
        Assumes uniform speaking rate.
        
        segments format: [{"text": "...", "image_path": "..."}]
        Returns list of segments with added "start_time" and "end_time"
        """
        if not full_script or not total_audio_duration:
            return []
            
        # Clean up full script to match character count actually spoken
        # TTS speaking rate is roughly uniform per character in Urdu
        total_chars = len(full_script.replace(" ", ""))
        if total_chars == 0:
            return []
            
        time_per_char = total_audio_duration / total_chars
        
        timeline = []
        current_time = 0.0
        
        for seg in segments:
            seg_text = seg.get("text", "")
            seg_chars = len(seg_text.replace(" ", ""))
            
            duration = seg_chars * time_per_char
            end_time = current_time + duration
            
            timeline.append({
                "text": seg_text,
                "image_path": seg.get("image_path"),
                "start_time": current_time,
                "end_time": end_time
            })
            
            current_time = end_time
            
        # Ensure the last segment stretches to the end just in case of rounding errors
        if timeline:
            timeline[-1]["end_time"] = max(timeline[-1]["end_time"], total_audio_duration)
            
        return timeline

    def build_ffmpeg_filter_for_slideshow(self, timeline: List[Dict[str, Any]], panel_w: int, panel_h: int) -> tuple[List[str], str]:
        """
        Builds the FFmpeg inputs and filter complex string to create a video stream 
        that changes images according to the timeline.
        Returns: (ffmpeg_inputs, filter_complex_string, output_label)
        """
        if not timeline:
            return [], "", ""
            
        inputs = []
        filter_parts = []
        
        # Filter out segments without images (or use a transparent/black placeholder)
        # Actually, we can just hold the previous image or show black if None.
        
        valid_segments = [s for s in timeline if s.get("image_path")]
        if not valid_segments:
            return [], "", ""
            
        input_idx_start = 0 # This will be offset by the caller
        
        overlay_chain = ""
        
        # We need a base canvas for the slideshow, same size as the panel
        filter_parts.append(f"color=c=black@0:s={panel_w}x{panel_h}:r=30[bg_slides]")
        last_out = "[bg_slides]"
        
        for i, seg in enumerate(valid_segments):
            img = seg["image_path"]
            inputs.extend(["-loop", "1", "-i", img])
            
            # Scale and pad image
            filter_parts.append(
                f"[{i}:v]"
                f"scale={panel_w}:{panel_h}:force_original_aspect_ratio=decrease,"
                f"pad={panel_w}:{panel_h}:(ow-iw)/2:(oh-ih)/2:color=0x12163700,"
                f"format=yuva420p[img{i}]"
            )
            
            start = seg["start_time"]
            end = seg["end_time"]
            
            # Enable the overlay only between start and end
            next_out = f"[out{i}]"
            filter_parts.append(
                f"{last_out}[img{i}]overlay=0:0:enable='between(t,{start},{end})'{next_out}"
            )
            last_out = next_out
            
        return inputs, ";".join(filter_parts), last_out
