import numpy as np

def generate_visemes(audio_data):
    """
    Generate viseme sequence from audio data with word-level timing.
    Returns list of (time_ms, viseme_type, mouth_open, word) tuples.
    
    mouth_open: float 0.0 (closed) to 1.0 (fully open)
    viseme_type: 'A' (closed), 'B' (medium), 'C' (wide) for backward compatibility
    """
    if not audio_data or len(audio_data) == 0:
        return []
    
    try:
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # If audio is stereo, take one channel
        if len(audio_array.shape) > 1:
            audio_array = audio_array[:, 0]
        
        # Calculate volume envelope
        volume = np.abs(audio_array).astype(np.float32)
        
        # Get sample rate (assume 24000 for edge-tts)
        sample_rate = 24000
        
        # Calculate frame positions (every 10ms for smoother animation)
        frame_duration = 10  # ms
        frame_samples = sample_rate * frame_duration // 1000
        
        # Calculate volume statistics for normalization
        max_volume = np.max(volume) if np.max(volume) > 0 else 1
        min_volume = np.percentile(volume[volume > 0], 5) if np.any(volume > 0) else 100
        volume_range = max_volume - min_volume
        
        # Smoothing window (50ms)
        smooth_window = 5
        
        visemes = []
        smoothed_volume = []
        
        # Apply smoothing to volume envelope
        for i in range(0, len(volume), frame_samples):
            frame = volume[i:i+frame_samples]
            if len(frame) == 0:
                break
            smoothed_volume.append(np.mean(frame))
        
        # Apply moving average smoothing
        if smoothed_volume:
            smoothed_volume = np.convolve(smoothed_volume, 
                                         np.ones(smooth_window)/smooth_window, 
                                         mode='same')
        
        for i, avg_volume in enumerate(smoothed_volume):
            time_ms = i * frame_duration
            
            # Normalize volume to 0-1 range with smooth curve
            if volume_range > 0:
                normalized = (avg_volume - min_volume) / volume_range
                normalized = max(0.0, min(1.0, normalized))
            else:
                normalized = 0.0
            
            # Apply non-linear curve for more natural movement
            # This makes quiet sounds slightly visible and loud sounds more pronounced
            mouth_open = normalized ** 0.7  # Compress the curve
            
            # Map to discrete viseme for backward compatibility
            if mouth_open < 0.15:
                viseme = "A"  # Closed/slight
            elif mouth_open < 0.6:
                viseme = "B"  # Medium
            else:
                viseme = "C"  # Wide open
            
            # Include mouth_open value for smooth animation
            visemes.append((time_ms, viseme, mouth_open))
        
        return visemes
        
    except Exception as e:
        print(f"Viseme generation error: {e}")
        return []

def estimate_word_timing(text, audio_duration_ms):
    """
    Estimate timing for each word in the text.
    Returns list of (start_ms, end_ms, word) tuples.
    """
    if not text or not audio_duration_ms:
        return []
    
    # Split text into words (keep Vietnamese characters)
    import re
    words = re.findall(r'[\w\u00C0-\u024F]+', text)
    words = [w.strip() for w in words if w.strip()]
    
    if not words:
        return []
    
    # Estimate average time per word (including pauses)
    total_chars = sum(len(w) for w in words)
    avg_word_duration = audio_duration_ms / len(words) * 1.2  # Add some padding
    
    word_timing = []
    current_time = 0
    
    for word in words:
        word_duration = len(word) / total_chars * audio_duration_ms * 0.9
        # Add small pause between words
        word_timing.append((current_time, current_time + word_duration, word))
        current_time += word_duration + 30  # 30ms pause between words
    
    return word_timing
