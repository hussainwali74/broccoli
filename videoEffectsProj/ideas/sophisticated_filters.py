# pip install opencv-python moviepy numpy tensorflow torch torchvision
# some advanced filters you can implement:
# 1. Real-Time Face Filters (AR Filters): Overlay virtual objects like masks, hats, or sunglasses on detected faces.
# 2. Style Transfer: Apply artistic styles to video frames.
# 3. Particle Effects: Add sparkles, fire, or smoke effects.
# 4. Lens Flares and Light Leaks: Simulate camera lens effects.
# 5. Dynamic Text and Graphics Overlay: Add animated text or graphics.
# 6. 3D Transformations: Apply 3D rotations and transformations.
# 7. Color Grading and LUTs: Adjust colors for cinematic effects.
# 8. Motion Tracking: Track objects and apply effects that follow their movement.

import cv2
import numpy as np

class ARFilters:
    def __init__(self, overlay_image_path, scale=1.0):
        # Load overlay image with alpha channel
        self.overlay = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
        self.scale = scale

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                  'haarcascade_frontalface_default.xml')

    def apply_filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Calculate overlay size and position
            overlay_width = int(w * self.scale)
            overlay_height = int(self.overlay.shape[0] * (overlay_width / self.overlay.shape[1]))
            overlay_resized = cv2.resize(self.overlay, (overlay_width, overlay_height), 
                                         interpolation=cv2.INTER_AREA)

            # Calculate position: Adjust y for better placement (e.g., eyes level)
            y_offset = y + int(h * 0.3)
            x_offset = x + int((w - overlay_width) / 2)

            # Ensure the overlay fits within the frame
            if y_offset + overlay_height > frame.shape[0] or x_offset + overlay_width > frame.shape[1]:
                continue

            # Split overlay into channels
            overlay_image = overlay_resized[:, :, :3]
            mask = overlay_resized[:, :, 3] / 255.0
            mask_inv = 1.0 - mask

            # Blend the overlay with the frame
            for c in range(0, 3):
                frame[y_offset:y_offset+overlay_height, 
                      x_offset:x_offset+overlay_width, c] = (
                    mask * overlay_image[:, :, c] + 
                    mask_inv * frame[y_offset:y_offset+overlay_height, 
                                     x_offset:x_offset+overlay_width, c]
                )
        return frame

def apply_ar_filter_to_video(input_path, output_path, overlay_image_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ar_filter = ARFilters(overlay_image_path, scale=1.0)  # Adjust scale as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply AR filter
        filtered_frame = ar_filter.apply_filter(frame)

        # Write the frame
        out.write(filtered_frame)

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_ar.mp4"
    overlay_image = "sunglasses.png"  # Ensure this has an alpha channel

    apply_ar_filter_to_video(input_video, output_video, overlay_image)
    
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
import matplotlib.pyplot as plt

class StyleTransfer:
    def __init__(self, style_image_path):
        # Load and preprocess style image
        self.style_image = cv2.imread(style_image_path)
        self.style_image = cv2.cvtColor(self.style_image, cv2.COLOR_BGR2RGB)
        self.style_image = cv2.resize(self.style_image, (512, 512))
        self.style_image = tf.convert_to_tensor(self.style_image, dtype=tf.float32)
        self.style_image = tf.expand_dims(self.style_image, axis=0)
        self.style_image = vgg19.preprocess_input(self.style_image)

        # Load VGG19 model
        self.vgg = vgg19.VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False

        # Define layers to extract style features
        self.style_layers = [
            'block1_conv1', 
            'block2_conv1', 
            'block3_conv1', 
            'block4_conv1', 
            'block5_conv1'
        ]
        self.style_outputs = [self.vgg.get_layer(name).output for name in self.style_layers]
        self.model = tf.keras.Model([self.vgg.input], self.style_outputs)

        # Compute style features
        self.style_features = self.get_style_features()

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result / num_locations

    def get_style_features(self):
        style_output = self.model(self.style_image)
        style_gram = [self.gram_matrix(style_layer) for style_layer in style_output]
        return style_gram

    def apply_style_transfer(self, frame):
        # Preprocess frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)

        # Get style features of frame
        frame_style_output = self.model(img)
        frame_style_gram = [self.gram_matrix(style_layer) for style_layer in frame_style_output]

        # Compute style loss
        style_loss = 0
        for gs, gf in zip(self.style_features, frame_style_gram):
            style_loss += tf.reduce_mean((gs - gf)**2)
        style_loss /= len(self.style_features)

        # Simple way: blend original frame with style image based on style loss
        alpha = 0.5  # Weight for style
        styled_frame = cv2.addWeighted(frame, 1 - alpha, 
                                       cv2.imread("style_transfer.png"), alpha, 0)

        return styled_frame

def apply_style_transfer_to_video(input_path, output_path, style_image_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    style_transfer = StyleTransfer(style_image_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply style transfer
        styled_frame = style_transfer.apply_style_transfer(frame)

        # Write the frame
        out.write(styled_frame)

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_style.mp4"
    style_image = "starry_night.jpg"  # Your style image

    apply_style_transfer_to_video(input_video, output_video, style_image)

import cv2
import numpy as np
import random

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.randint(2, 5)
        self.color = (random.randint(200,255), random.randint(200,255), random.randint(200,255))
        self.lifetime = random.randint(20, 40)

    def update(self):
        self.y -= 2  # Move upwards
        self.lifetime -= 1

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)

class ParticleEffects:
    def __init__(self, max_particles=100):
        self.particles = []
        self.max_particles = max_particles

    def emit(self, x, y, num=5):
        for _ in range(num):
            self.particles.append(Particle(x, y))
            if len(self.particles) > self.max_particles:
                self.particles.pop(0)

    def apply_effect(self, frame, detections=None):
        # If detections are provided, emit particles at detection centers
        if detections:
            for (x, y, w, h) in detections:
                center_x = x + w // 2
                center_y = y
                self.emit(center_x, center_y, num=3)

        # Update and draw particles
        for particle in self.particles[:]:
            particle.update()
            if particle.lifetime <= 0:
                self.particles.remove(particle)
            else:
                particle.draw(frame)
        return frame

def apply_particle_effects_to_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    particle_effect = ParticleEffects()

    # Initialize object detector (simple face detector for demonstration)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects (faces)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Apply particle effects
        frame = particle_effect.apply_effect(frame, detections)

        # Write the frame
        out.write(frame)

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_particles.mp4"

    apply_particle_effects_to_video(input_video, output_video)

import cv2
import numpy as np

class LensFlares:
    def __init__(self, flare_image_path):
        self.flare = cv2.imread(flare_image_path, cv2.IMREAD_UNCHANGED)
        self.flare = cv2.cvtColor(self.flare, cv2.COLOR_BGRA2RGBA)  # Ensure alpha channel

    def apply_flare(self, frame, position=(0,0), scale=1.0, opacity=0.5):
        # Resize flare image
        flare_w = int(self.flare.shape[1] * scale)
        flare_h = int(self.flare.shape[0] * scale)
        flare_resized = cv2.resize(self.flare, (flare_w, flare_h), interpolation=cv2.INTER_AREA)

        # Split flare into channels
        b, g, r, a = cv2.split(flare_resized)
        overlay_color = cv2.merge((b, g, r))

        mask = cv2.merge((a, a, a)) / 255.0

        # Calculate region of interest
        x, y = position
        y1, y2 = max(0, y), min(frame.shape[0], y + flare_h)
        x1, x2 = max(0, x), min(frame.shape[1], x + flare_w)

        # Adjust flare overlay to fit within frame
        overlay_crop = overlay_color[0:y2 - y1, 0:x2 - x1]
        mask_crop = mask[0:y2 - y1, 0:x2 - x1]

        # Blend the flare with the frame
        frame[y1:y2, x1:x2] = (1.0 - mask_crop * opacity) * frame[y1:y2, x1:x2] + \
                                mask_crop * opacity * overlay_crop

        return frame

def apply_lens_flare_to_video(input_path, output_path, flare_image_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    lens_flare = LensFlares(flare_image_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Example: Position flare based on frame progression
        # Here, moving from left to right
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = frame_count / total_frames
        x = int(width * progress)
        y = int(height * 0.2)  # Fixed vertical position

        # Apply lens flare
        frame = lens_flare.apply_flare(frame, position=(x, y), scale=0.5, opacity=0.6)

        # Write the frame
        out.write(frame)

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_flare.mp4"
    flare_image = "lens_flare.png"  # Ensure this has an alpha channel

    apply_lens_flare_to_video(input_video, output_video, flare_image)

import cv2
import numpy as np

class TextOverlay:
    def __init__(self, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                 font_scale=1, color=(255, 255, 255), thickness=2):
        self.text = text
        self.position = position
        self.font = font
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness

    def apply_text(self, frame, frame_number, total_frames):
        # Example: Fade in and out effect
        fade_in_duration = total_frames * 0.1
        fade_out_duration = total_frames * 0.1
        alpha = 1.0

        if frame_number < fade_in_duration:
            alpha = frame_number / fade_in_duration
        elif frame_number > total_frames - fade_out_duration:
            alpha = (total_frames - frame_number) / fade_out_duration
        else:
            alpha = 1.0

        # Create overlay for transparency
        overlay = frame.copy()
        cv2.putText(overlay, self.text, self.position, self.font, 
                    self.font_scale, self.color, self.thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

class GraphicsOverlay:
    def __init__(self, graphic_image_path, position, scale=1.0):
        self.graphic = cv2.imread(graphic_image_path, cv2.IMREAD_UNCHANGED)
        self.position = position
        self.scale = scale

    def apply_graphic(self, frame, frame_number, total_frames):
        # Simple scaling animation: grow and shrink
        max_scale = self.scale * 1.5
        min_scale = self.scale * 0.5
        scale = min_scale + (max_scale - min_scale) * abs(
            np.sin((frame_number / total_frames) * 2 * np.pi)
        )

        # Resize graphic
        graphic_resized = cv2.resize(self.graphic, 
                                     (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = graphic_resized.shape[:2]
        x, y = self.position

        # Ensure the graphic fits within the frame
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return frame

        # Split graphic into channels
        b, g, r, a = cv2.split(graphic_resized)
        overlay_color = cv2.merge((b, g, r))
        mask = a / 255.0

        # Blend the graphic with the frame
        frame[y:y+h, x:x+w] = (1.0 - mask[..., None]) * frame[y:y+h, x:x+w] + \
                                mask[..., None] * overlay_color

        return frame

def apply_text_and_graphics_overlay(input_path, output_path, 
                                    text, text_position, graphic_image_path=None, 
                                    graphic_position=(0,0)):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    text_overlay = TextOverlay(text, text_position)
    graphics_overlay = GraphicsOverlay(graphic_image_path, graphic_position) if graphic_image_path else None

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply text overlay
        frame = text_overlay.apply_text(frame, frame_number, total_frames)

        # Apply graphics overlay if exists
        if graphics_overlay:
            frame = graphics_overlay.apply_graphic(frame, frame_number, total_frames)

        # Write the frame
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_text_graphics.mp4"
    text = "Welcome to My Video!"
    text_position = (50, 50)  # (x, y) coordinates
    graphic_image = "logo.png"  # Ensure this has an alpha channel

    apply_text_and_graphics_overlay(input_video, output_video, text, text_position, graphic_image, (width//2, height//2))
# ----------------------------
# Filter 6: Style-Based Lens Effects Using OpenGL

import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

# Note: Implementing full OpenGL in Python for video processing is complex.
# This example uses a simple color inversion shader.

VERTEX_SHADER = """
#version 330
layout(location = 0) in vec3 position
layout(location = 1) in vec2 texCoords
out vec2 TexCoords
void main()
{
    gl_Position = vec4(position, 1.0);
    TexCoords = texCoords;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 TexCoords;
out vec4 color;
uniform sampler2D screenTexture;

void main()
{
    vec3 texColor = texture(screenTexture, TexCoords).rgb;
    // Simple inversion effect
    color = vec4(vec3(1.0) - texColor, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    # Check compilation
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        info = glGetShaderInfoLog(shader)
        shader_type_name = 'vertex' if shader_type == GL_VERTEX_SHADER else 'fragment'
        raise RuntimeError(f"ERROR::SHADER_COMPILATION_ERROR of type: {shader_type_name}\n{info.decode()}")
    return shader

def create_shader_program():
    vertex = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    # Check linking
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        info = glGetProgramInfoLog(program)
        raise RuntimeError(f"ERROR::PROGRAM_LINKING_ERROR:\n{info.decode()}")
    glDeleteShader(vertex)
    glDeleteShader(fragment)
    return program

def apply_opengl_shader_to_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize OpenGL
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    window = glutCreateWindow(b"OpenGL Shader")

    # Create shader program
    shader_program = create_shader_program()

    # Define quad vertices and texture coordinates
    vertices = np.array([
        # Positions        # Texture Coords
        -1.0,  1.0, 0.0,   0.0, 1.0,
        -1.0, -1.0, 0.0,   0.0, 0.0,
         1.0, -1.0, 0.0,   1.0, 0.0,

        -1.0,  1.0, 0.0,   0.0, 1.0,
         1.0, -1.0, 0.0,   1.0, 0.0,
         1.0,  1.0, 0.0,   1.0, 1.0
    ], dtype=np.float32)

    # Setup VAO and VBO
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Position attribute
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(0))
    # Texture coord attribute
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, 
                          ctypes.c_void_p(12))

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Create texture
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare image for OpenGL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_flipped = cv2.flip(frame_rgb, 0)  # Flip vertically
        frame_data = frame_flipped.flatten()

        # Upload texture
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, frame_data)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader_program)
        glBindVertexArray(VAO)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glUniform1i(glGetUniformLocation(shader_program, "screenTexture"), 0)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glutSwapBuffers()

        # Read pixels back
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        frame_opengl = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
        frame_opengl = cv2.cvtColor(frame_opengl, cv2.COLOR_RGBA2BGR)
        frame_opengl = cv2.flip(frame_opengl, 0)  # Flip back

        # Write the frame
        out.write(frame_opengl)

    cap.release()
    out.release()
    glutDestroyWindow(window)

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_shader.mp4"
    flare_image = "lens_flare.png"  # Example shader overlay

    apply_opengl_shader_to_video(input_video, output_video)

# Filter 7: Combining Multiple Effects for a TikTok-Like Filter
import cv2
import numpy as np
from filters.ar_filters import ARFilters
from filters.lens_flares import LensFlares
from filters.text_overlay import TextOverlay, GraphicsOverlay

def apply_combined_filters(input_path, output_path, 
                           overlay_image_path, flare_image_path, 
                           text, text_position, graphic_image_path=None, 
                           graphic_position=(0,0)):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize individual filters
    ar_filter = ARFilters(overlay_image_path, scale=1.0)
    lens_flare = LensFlares(flare_image_path)
    text_overlay = TextOverlay(text, text_position)
    graphics_overlay = GraphicsOverlay(graphic_image_path, graphic_position) if graphic_image_path else None

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply AR filter
        frame = ar_filter.apply_filter(frame)

        # Apply lens flare based on frame progression
        progress = frame_number / total_frames
        x = int(width * progress)  # Move flare from left to right
        y = int(height * 0.2)      # Fixed vertical position
        frame = lens_flare.apply_flare(frame, position=(x, y), scale=0.5, opacity=0.6)

        # Apply text overlay
        frame = text_overlay.apply_text(frame, frame_number, total_frames)

        # Apply graphics overlay if exists
        if graphics_overlay:
            frame = graphics_overlay.apply_graphic(frame, frame_number, total_frames)

        # Optionally, add more effects here (e.g., style transfer)

        # Write the frame
        out.write(frame)
        frame_number += 1

        # Optional: Progress display
        if frame_number % 30 == 0:
            print(f"Processing: {frame_number}/{total_frames} frames ({(frame_number / total_frames) * 100:.1f}%)")

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_combined.mp4"
    overlay_image = "sunglasses.png"    # AR filter image with alpha channel
    flare_image = "lens_flare.png"      # Lens flare image with alpha channel
    text = "Enjoy the Show!"
    text_position = (50, 50)             # Top-left corner
    graphic_image = "logo.png"           # Optional graphic overlay
    graphic_position = (width - 150, height - 100)  # Bottom-right corner

    apply_combined_filters(input_video, output_video, 
                           overlay_image, flare_image, 
                           text, text_position, 
                           graphic_image, graphic_position)

## **Additional Advanced Effects**
# . Motion Tracking and Effect Application
import cv2
import numpy as np

class MotionTracker:
    def __init__(self):
        # Initialize object tracker (e.g., CSRT)
        self.tracker = cv2.TrackerCSRT_create()
        self.initialized = False

    def initialize(self, frame, bbox):
        self.tracker.init(frame, bbox)
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            return None
        success, bbox = self.tracker.update(frame)
        if success:
            return bbox
        else:
            return None

class MotionEffect:
    def __init__(self):
        self.tracker = MotionTracker()
        self.trails = []

    def apply_motion_effect(self, frame):
        # Add trail effect based on tracked object
        for trail in self.trails:
            for i in range(1, len(trail)):
                if trail[i - 1] is None or trail[i] is None:
                    continue
                thickness = int(np.sqrt(len(trail) / float(i + 1)) * 2)
                cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), thickness)
        return frame

def apply_motion_tracking_effect(input_path, output_path, initial_bbox):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    motion_effect = MotionEffect()

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number == 0:
            # Initialize tracker with first frame and bounding box
            motion_effect.tracker.initialize(frame, initial_bbox)

        # Update tracker
        bbox = motion_effect.tracker.update(frame)
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            center = (x + w // 2, y + h // 2)
            motion_effect.trails.append([center])

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Apply motion trails
        frame = motion_effect.apply_motion_effect(frame)

        # Write the frame
        out.write(frame)
        frame_number += 1

        # Optional: Progress display
        if frame_number % 30 == 0:
            print(f"Processing: {frame_number}/{total_frames} frames ({(frame_number / total_frames) * 100:.1f}%)")

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_motion.mp4"
    # Define initial bounding box (x, y, w, h)
    initial_bbox = (100, 100, 200, 200)

    apply_motion_tracking_effect(input_video, output_video, initial_bbox)
#  3D Rotations and Transformations
import cv2
import numpy as np

class ThreeDTransform:
    def __init__(self, angle_x=0, angle_y=0, angle_z=0, scale=1.0):
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.scale = scale

    def rotate_frame(self, frame):
        rows, cols, _ = frame.shape
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(self.angle_x), -np.sin(self.angle_x)],
                       [0, np.sin(self.angle_x), np.cos(self.angle_x)]])
        
        Ry = np.array([[np.cos(self.angle_y), 0, np.sin(self.angle_y)],
                       [0, 1, 0],
                       [-np.sin(self.angle_y), 0, np.cos(self.angle_y)]])
        
        Rz = np.array([[np.cos(self.angle_z), -np.sin(self.angle_z), 0],
                       [np.sin(self.angle_z), np.cos(self.angle_z), 0],
                       [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        
        # Projection matrix
        proj = np.array([[1, 0, 0],
                         [0, 1, 0]])
        
        transformed = []
        for y in range(rows):
            for x in range(cols):
                vec = np.array([x - cols/2, y - rows/2, 1]) * self.scale
                rotated = R @ vec
                projected = proj @ rotated
                new_x, new_y = projected + np.array([cols/2, rows/2])
                transformed.append([new_x, new_y])

        transformed = np.array(transformed).reshape(rows, cols, 2).astype(np.float32)
        transformed_frame = cv2.remap(frame, transformed[:,:,0], transformed[:,:,1], 
                                     cv2.INTER_LINEAR)
        return transformed_frame

def apply_3d_transform_to_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    three_d_transform = ThreeDTransform()

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update rotation angles dynamically
        progress = frame_number / total_frames
        three_d_transform.angle_x = progress * np.pi / 6  # Rotate up to 30 degrees
        three_d_transform.angle_y = progress * np.pi / 6
        three_d_transform.angle_z = progress * np.pi / 6

        # Apply 3D rotation
        transformed_frame = three_d_transform.rotate_frame(frame)

        # Write the frame
        out.write(transformed_frame)
        frame_number += 1

        # Optional: Progress display
        if frame_number % 30 == 0:
            print(f"Processing: {frame_number}/{total_frames} frames ({(frame_number / total_frames) * 100:.1f}%)")

    cap.release()
    out.release()

# Usage Example
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_3d.mp4"

    apply_3d_transform_to_video(input_video, output_video)