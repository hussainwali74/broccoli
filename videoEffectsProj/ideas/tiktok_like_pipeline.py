import cv2
import numpy as np
from filters.ar_filters import ARFilters
from filters.lens_flares import LensFlares
from filters.text_overlay import TextOverlay, GraphicsOverlay
from filters.style_transfer import StyleTransfer
from filters.particle_effects import ParticleEffects

def tiktok_like_filter_pipeline(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize filters
    ar_filter = ARFilters("sunglasses.png", scale=1.0)
    lens_flare = LensFlares("lens_flare.png")
    text_overlay = TextOverlay("Vibe Check!", (50, 50))
    graphics_overlay = GraphicsOverlay("logo.png", (width - 150, height - 100))
    style_transfer = StyleTransfer("starry_night.jpg")
    particle_effect = ParticleEffects()

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        progress = frame_number / total_frames

        # Apply Style Transfer with progressive intensity
        styled_frame = style_transfer.apply_style_transfer(frame)
        alpha = 0.5 * np.sin(progress * 2 * np.pi) + 0.5  # Oscillate between 0 and 1
        frame = cv2.addWeighted(frame, 1 - alpha, styled_frame, alpha, 0)

        # Apply AR filter
        frame = ar_filter.apply_filter(frame)

        # Apply lens flare moving across the screen
        x = int(width * progress)
        y = int(height * 0.2)
        frame = lens_flare.apply_flare(frame, position=(x, y), scale=0.5, opacity=0.7)

        # Apply particle effects on detected objects (e.g., faces)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = ar_filter.face_cascade.detectMultiScale(gray, 1.1, 4)
        frame = particle_effect.apply_effect(frame, faces)

        # Apply text overlay
        frame = text_overlay.apply_text(frame, frame_number, total_frames)

        # Apply graphics overlay
        if graphics_overlay:
            frame = graphics_overlay.apply_graphic(frame, frame_number, total_frames)

        # Write the frame
        out.write(frame)
        frame_number += 1

        # Optional: Display progress
        if frame_number % 30 == 0:
            print(f"Processing: {frame_number}/{total_frames} frames ({(frame_number / total_frames)*100:.1f}%)")

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_tiktok_like.mp4"

    tiktok_like_filter_pipeline(input_video, output_video)