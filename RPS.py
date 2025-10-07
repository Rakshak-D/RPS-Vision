#!/usr/bin/env python3
"""
Rock Paper Scissors - Vision Edition

An advanced, single-file Rock-Paper-Scissors game that uses your webcam and hand gestures.
This version features a redesigned UI, dynamic full-screen scaling, and improved visual feedback.

Dependencies:
  pip install opencv-python mediapipe numpy pygame tkinter

Features:
- Modern, full-screen UI that adapts to any monitor resolution.
- Enhanced visual assets and animations for game elements.
- MediaPipe Hands for robust, landmark-based gesture recognition.
- Mirrored player preview inside a dedicated UI panel.
- Dynamic game states: Splash Screen, Countdown, and Results.
- Clear audio feedback for all game events.
- On-screen stats: score, win streak, and live FPS counter.
- Gesture smoothing to prevent flickering choices.
"""
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import pygame
import tkinter as tk
from collections import deque

# ---------------- INITIAL SETUP ----------------

def get_screen_resolution():
    """Uses tkinter to get the user's screen resolution."""
    try:
        root = tk.Tk()
        root.withdraw()
        return root.winfo_screenwidth(), root.winfo_screenheight()
    except tk.TclError:
        # Fallback for environments without a display
        return 1280, 720

# ---------------- CONFIG & THEME ----------------
WIDTH, HEIGHT = get_screen_resolution()
FPS_TARGET = 30
MIN_CONFIDENCE = 0.6  # MediaPipe min detection confidence

# Dynamic UI Scaling
ROI_SCALE = 0.35  # ROI size as a percentage of screen height
ROI_SIZE = int(HEIGHT * ROI_SCALE)
ROI_MARGIN = int(WIDTH * 0.03)

# Player ROI (right side of screen)
PLAYER_ROI_TL = (WIDTH - ROI_SIZE - ROI_MARGIN, (HEIGHT - ROI_SIZE) // 2)
PLAYER_ROI_BR = (WIDTH - ROI_MARGIN, (HEIGHT + ROI_SIZE) // 2)

# Computer ROI (left side)
COMP_ROI_TL = (ROI_MARGIN, (HEIGHT - ROI_SIZE) // 2)
COMP_ROI_BR = (ROI_MARGIN + ROI_SIZE, (HEIGHT + ROI_SIZE) // 2)

# Game Timings
COUNTDOWN_SECONDS = 3
RESULT_DISPLAY_SECONDS = 2.5

# Gesture Smoothing
HISTORY_LEN = 8  # Number of frames to average gesture over

# Theme Colors
class Color:
    BACKGROUND = (28, 28, 28)
    TEXT = (255, 255, 255)
    ACCENT = (0, 191, 255)  # Deep Sky Blue
    PLAYER_GLOW = (0, 255, 0)
    COMP_GLOW = (255, 69, 0)
    WIN = (50, 205, 50)
    LOSE = (255, 69, 0)
    DRAW = (255, 165, 0)
    PANEL = (45, 45, 45)

# ---------------- MEDIA PIPE SETUP ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ---------------- AUDIO SETUP ----------------
pygame.mixer.init()
def make_beep(freq, length_ms, decay_ms=100):
    """Creates a simple tone with a decay for a softer sound."""
    sample_rate = 44100
    n_samples = int(sample_rate * (length_ms / 1000.0))
    buf = np.sin(2.0 * np.pi * np.arange(n_samples) * freq / sample_rate)
    
    # Apply a linear decay for a less harsh sound
    decay_samples = int(sample_rate * (decay_ms / 1000.0))
    if n_samples > decay_samples:
        decay = np.linspace(1, 0, decay_samples)
        buf[n_samples-decay_samples:] *= decay

    sound = np.int16(buf * 32767 * 0.7)  # Reduce volume slightly
    sound = np.asfortranarray([sound, sound]) # Make it stereo
    return pygame.mixer.Sound(sound.T)

SOUNDS = {
    "WIN": make_beep(880, 250),
    "LOSE": make_beep(330, 300),
    "DRAW": make_beep(550, 250),
    "INVALID": make_beep(220, 400),
    "COUNTDOWN": make_beep(660, 150),
    "GO": make_beep(990, 150),
    "START": make_beep(770, 100)
}

# ---------------- GESTURE RECOGNITION ----------------
def get_finger_status(hand_landmarks):
    """
    Returns a list of booleans indicating if each finger is extended.
    [thumb, index, middle, ring, pinky]
    """
    fingers = []
    # Thumb: Check if tip is farther from the wrist on the x-axis than the pip joint.
    # This works for both hands because we check relative to the palm's orientation.
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    thumb_ip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    
    # Determine hand orientation (left vs right hand in mirrored view)
    index_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
    pinky_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x

    # In a mirrored view, a right hand will have index_mcp_x > pinky_mcp_x
    if index_mcp_x > pinky_mcp_x: # Right hand
        fingers.append(thumb_tip_x > thumb_ip_x)
    else: # Left hand
        fingers.append(thumb_tip_x < thumb_ip_x)

    # Other 4 fingers: Check if tip is higher (smaller y-value) than the pip joint.
    tip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP
    ]
    pip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP
    ]
    for i in range(4):
        tip_y = hand_landmarks.landmark[tip_ids[i]].y
        pip_y = hand_landmarks.landmark[pip_ids[i]].y
        fingers.append(tip_y < pip_y)
        
    return fingers

def get_gesture_from_fingers(finger_status):
    """Maps a list of finger statuses to a Rock/Paper/Scissors gesture."""
    extended_fingers = sum(finger_status)
    
    if extended_fingers == 0:
        return "Rock"
    if extended_fingers == 5:
        return "Paper"
    if extended_fingers == 2 and finger_status[1] and finger_status[2]:
        return "Scissors"
    return "Invalid"

# ---------------- UI & DRAWING HELPERS ----------------
def draw_text(frame, text, pos, font_scale=1.0, color=Color.TEXT, thickness=2, align="left"):
    """Draws text with options for alignment."""
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    if align == "center":
        pos = (pos[0] - text_size[0] // 2, pos[1] + text_size[1] // 2)
    elif align == "right":
        pos = (pos[0] - text_size[0], pos[1])
    
    cv2.putText(frame, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

def draw_rounded_rect(frame, top_left, bottom_right, color, radius=0.1, thickness=-1):
    """Draws a rounded rectangle."""
    h = bottom_right[1] - top_left[1]
    w = bottom_right[0] - top_left[0]
    r = int(min(h, w) * radius)
    
    # Draw four corners
    cv2.circle(frame, (top_left[0] + r, top_left[1] + r), r, color, thickness)
    cv2.circle(frame, (bottom_right[0] - r, top_left[1] + r), r, color, thickness)
    cv2.circle(frame, (top_left[0] + r, bottom_right[1] - r), r, color, thickness)
    cv2.circle(frame, (bottom_right[0] - r, bottom_right[1] - r), r, color, thickness)
    
    # Draw rectangles to fill the space
    cv2.rectangle(frame, (top_left[0] + r, top_left[1]), (bottom_right[0] - r, bottom_right[1]), color, thickness)
    cv2.rectangle(frame, (top_left[0], top_left[1] + r), (bottom_right[0], bottom_right[1] - r), color, thickness)

def draw_choice_icon(frame, choice, center_pos, size):
    """Draws a visually appealing icon for Rock, Paper, or Scissors."""
    if choice is None: return

    half_size = size // 2
    if choice == "Rock":
        cv2.circle(frame, center_pos, half_size, (110, 110, 110), -1)
        cv2.circle(frame, center_pos, half_size, (140, 140, 140), size // 20)
    elif choice == "Paper":
        top_left = (center_pos[0] - half_size, center_pos[1] - half_size)
        bottom_right = (center_pos[0] + half_size, center_pos[1] + half_size)
        draw_rounded_rect(frame, top_left, bottom_right, (240, 240, 240), radius=0.2)
    elif choice == "Scissors":
        angle = -45
        length = half_size * 1.2
        p1 = (int(center_pos[0] - length * np.cos(np.radians(angle))), 
              int(center_pos[1] - length * np.sin(np.radians(angle))))
        p2 = (int(center_pos[0] + length * np.cos(np.radians(angle))), 
              int(center_pos[1] + length * np.sin(np.radians(angle))))
        
        angle2 = 45
        p3 = (int(center_pos[0] - length * np.cos(np.radians(angle2))), 
              int(center_pos[1] - length * np.sin(np.radians(angle2))))
        p4 = (int(center_pos[0] + length * np.cos(np.radians(angle2))), 
              int(center_pos[1] + length * np.sin(np.radians(angle2))))

        cv2.line(frame, p1, p2, (180, 180, 180), size // 10)
        cv2.line(frame, p3, p4, (180, 180, 180), size // 10)
        cv2.circle(frame, (center_pos[0] - int(half_size*0.3), center_pos[1] + int(half_size*0.3)), size // 8, Color.ACCENT, -1)
        cv2.circle(frame, (center_pos[0] + int(half_size*0.3), center_pos[1] + int(half_size*0.3)), size // 8, Color.ACCENT, -1)

# ---------------- GAME LOGIC CLASS ----------------
class RPSGame:
    def __init__(self):
        self.player_score = 0
        self.comp_score = 0
        self.win_streak = 0
        self.state = "SPLASH"  # SPLASH -> COUNTDOWN -> RESULT
        self.comp_choice = None
        self.player_choice = None
        self.result_text = ""
        self.result_color = Color.TEXT
        self.last_state_time = time.time()
        self.countdown_val = COUNTDOWN_SECONDS
        self.gesture_history = deque(maxlen=HISTORY_LEN)
        self.fps = 0.0

    def start_round(self):
        self.comp_choice = random.choice(["Rock", "Paper", "Scissors"])
        self.player_choice = None
        self.result_text = ""
        self.state = "COUNTDOWN"
        self.last_state_time = time.time()
        self.gesture_history.clear()
        SOUNDS["COUNTDOWN"].play()

    def evaluate_player_choice(self):
        if not self.gesture_history:
            return "Invalid"
        # Return the most common valid gesture in history
        valid_gestures = [g for g in self.gesture_history if g != "Invalid"]
        if not valid_gestures:
            return "Invalid"
        return max(set(valid_gestures), key=valid_gestures.count)

    def finish_round(self):
        self.player_choice = self.evaluate_player_choice()

        if self.player_choice == "Invalid":
            self.result_text = "Invalid Move!"
            self.result_color = Color.DRAW
            SOUNDS["INVALID"].play()
        elif self.player_choice == self.comp_choice:
            self.result_text = "It's a DRAW!"
            self.result_color = Color.DRAW
            self.win_streak = 0
            SOUNDS["DRAW"].play()
        elif (self.player_choice == "Rock" and self.comp_choice == "Scissors") or \
             (self.player_choice == "Scissors" and self.comp_choice == "Paper") or \
             (self.player_choice == "Paper" and self.comp_choice == "Rock"):
            self.result_text = "YOU WIN!"
            self.result_color = Color.WIN
            self.player_score += 1
            self.win_streak += 1
            SOUNDS["WIN"].play()
        else:
            self.result_text = "YOU LOSE!"
            self.result_color = Color.LOSE
            self.comp_score += 1
            self.win_streak = 0
            SOUNDS["LOSE"].play()

        self.state = "RESULT"
        self.last_state_time = time.time()

# ---------------- MAIN APPLICATION ----------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Request HD, but will be scaled down
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    
    cv2.namedWindow("Rock Paper Scissors - Vision", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Rock Paper Scissors - Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    game = RPSGame()
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=0.5
    )

    frame_count = 0
    fps_calc_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1) # Mirror for intuitive interaction
            
            # --- Main Game Canvas ---
            out_frame = np.full((HEIGHT, WIDTH, 3), Color.BACKGROUND, dtype=np.uint8)

            # --- Scoreboard & Info ---
            draw_text(out_frame, "ROCK PAPER SCISSORS", (WIDTH // 2, 50), 1.2, Color.ACCENT, 3, "center")
            draw_text(out_frame, f"PLAYER: {game.player_score}", (int(WIDTH*0.7), 120), 1.0, Color.PLAYER_GLOW, 2)
            draw_text(out_frame, f"COMPUTER: {game.comp_score}", (int(WIDTH*0.3), 120), 1.0, Color.COMP_GLOW, 2, align="right")
            draw_text(out_frame, f"STREAK: {game.win_streak}", (WIDTH // 2, 160), 0.8, Color.DRAW, 2, "center")
            draw_text(out_frame, f"FPS: {int(game.fps)}", (WIDTH - 40, 40), 0.7, Color.TEXT, 1, "right")

            # --- UI Panels for Player and Computer ---
            draw_rounded_rect(out_frame, COMP_ROI_TL, COMP_ROI_BR, Color.PANEL)
            draw_rounded_rect(out_frame, PLAYER_ROI_TL, PLAYER_ROI_BR, Color.PANEL)
            draw_text(out_frame, "COMPUTER", (COMP_ROI_TL[0] + ROI_SIZE // 2, COMP_ROI_TL[1] - 20), 0.8, Color.TEXT, 2, "center")
            draw_text(out_frame, "PLAYER", (PLAYER_ROI_TL[0] + ROI_SIZE // 2, PLAYER_ROI_TL[1] - 20), 0.8, Color.TEXT, 2, "center")

            # --- Player Camera Feed Processing ---
            x1, y1 = PLAYER_ROI_TL
            x2, y2 = PLAYER_ROI_BR
            
            # Fix: Resize the raw camera frame to match the ROI dimensions exactly.
            # This prevents shape mismatch errors if the camera resolution is unexpected.
            roi = cv2.resize(frame, (ROI_SIZE, ROI_SIZE))

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = hands.process(roi_rgb)

            current_gesture = "Invalid"
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    roi, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                finger_status = get_finger_status(hand_landmarks)
                current_gesture = get_gesture_from_fingers(finger_status)
            
            # Place processed ROI back into the main frame
            out_frame[y1:y2, x1:x2] = roi
            
            # --- Game State Logic & Rendering ---
            if game.state == "SPLASH":
                draw_text(out_frame, "Show your hand in the box", (WIDTH // 2, HEIGHT // 2 - 50), 1.2, Color.TEXT, 3, "center")
                draw_text(out_frame, "Press 'S' to Start", (WIDTH // 2, HEIGHT // 2 + 20), 1.0, Color.ACCENT, 2, "center")
                draw_text(out_frame, "Press 'Q' to Quit", (WIDTH // 2, HEIGHT - 50), 0.8, Color.TEXT, 2, "center")

            elif game.state == "COUNTDOWN":
                elapsed = time.time() - game.last_state_time
                countdown_left = COUNTDOWN_SECONDS - int(elapsed)
                
                if countdown_left != game.countdown_val:
                    game.countdown_val = countdown_left
                    if countdown_left > 0:
                        SOUNDS["COUNTDOWN"].play()
                    else:
                        SOUNDS["GO"].play()

                if elapsed >= COUNTDOWN_SECONDS:
                    game.finish_round()
                else:
                    draw_text(out_frame, str(countdown_left), (WIDTH//2, HEIGHT//2), 6.0, Color.ACCENT, 15, "center")
                    game.gesture_history.append(current_gesture)
                
                # Show computer's "thinking" animation
                if int(time.time() * 2) % 2 == 0:
                    draw_choice_icon(out_frame, random.choice(["Rock", "Paper", "Scissors"]), 
                                     (COMP_ROI_TL[0] + ROI_SIZE // 2, COMP_ROI_TL[1] + ROI_SIZE // 2), 
                                     int(ROI_SIZE * 0.7))

            elif game.state == "RESULT":
                draw_text(out_frame, game.result_text, (WIDTH//2, HEIGHT//2 - 50), 2.0, game.result_color, 5, "center")
                draw_choice_icon(out_frame, game.player_choice, 
                                 (int(WIDTH/2) + 150, HEIGHT//2 + 80), 
                                 int(ROI_SIZE * 0.5))
                draw_choice_icon(out_frame, game.comp_choice, 
                                 (int(WIDTH/2) - 150, HEIGHT//2 + 80), 
                                 int(ROI_SIZE * 0.5))
                
                if time.time() - game.last_state_time >= RESULT_DISPLAY_SECONDS:
                    game.start_round()

            # --- Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # q or ESC
                break
            if game.state == "SPLASH" and key == ord('s'):
                SOUNDS["START"].play()
                game.start_round()
            if key == ord('r'):
                game.__init__() # Reset game

            # --- FPS Calculation ---
            frame_count += 1
            if time.time() - fps_calc_time >= 1.0:
                game.fps = frame_count
                frame_count = 0
                fps_calc_time = time.time()

            cv2.imshow("Rock Paper Scissors - Vision", out_frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()

