import gradio as gr
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO

KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def check_backsquat_form(kpts, state_tracker):
    feedback = []
    if state_tracker['state'] == "down":
        lk, rk = kpts[KEYPOINT_DICT['left_knee']], kpts[KEYPOINT_DICT['right_knee']]
        la, ra = kpts[KEYPOINT_DICT['left_ankle']], kpts[KEYPOINT_DICT['right_ankle']]
        if abs(lk[0] - rk[0]) < abs(la[0] - ra[0]) * 0.9: feedback.append("KNEES IN! PUSH THEM OUT.")
    rs, rh, rk = kpts[KEYPOINT_DICT['right_shoulder']], kpts[KEYPOINT_DICT['right_hip']], kpts[KEYPOINT_DICT['right_knee']]
    if calculate_angle(rs, rh, rk) < 70 and state_tracker['state'] == "down": feedback.append("CHEST FALLING! KEEP IT UP.")
    return feedback

def check_barbellrow_form(kpts, state_tracker):
    feedback = []
    if state_tracker['state'] == "up" and state_tracker.get('initial_torso_angle'):
        rs, rh, ra = kpts[KEYPOINT_DICT['right_shoulder']], kpts[KEYPOINT_DICT['right_hip']], kpts[KEYPOINT_DICT['right_ankle']]
        current_angle = calculate_angle(rs, rh, ra)
        if current_angle > state_tracker['initial_torso_angle'] + 15: feedback.append("DON'T USE YOUR BACK! KEEP TORSO STILL.")
    return feedback

def check_overheadpress_form(kpts, state_tracker):
    feedback = []
    if state_tracker['state'] == "up":
        current_hip_y = kpts[KEYPOINT_DICT['right_hip']][1]
        if current_hip_y > state_tracker.get('start_hip_y', 0) + 10: feedback.append("DON'T USE YOUR LEGS! STRICT PRESS.")
    return feedback

def check_jumpingjack_form(kpts, state_tracker):
    feedback = []
    if state_tracker['state'] == "out":
        ls, le, lw = kpts[KEYPOINT_DICT['left_shoulder']], kpts[KEYPOINT_DICT['left_elbow']], kpts[KEYPOINT_DICT['left_wrist']]
        elbow_angle = calculate_angle(ls, le, lw)
        if elbow_angle < 150:
            feedback.append("EXTEND ARMS FULLY!")
    return feedback

def update_backsquat_state(kpts, tracker):
    lh, rh, lk, rk, la, ra = (kpts[KEYPOINT_DICT[name]] for name in ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'])
    angle = (calculate_angle(lh, lk, la) + calculate_angle(rh, rk, ra)) / 2
    if tracker['state'] == "up" and angle < 160:
        tracker['state'] = "down"; tracker['min_angle_rep'] = angle
    elif tracker['state'] == "down":
        tracker['min_angle_rep'] = min(tracker['min_angle_rep'], angle)
        if angle > 165:
            tracker['state'] = "up"; tracker['reps'] += 1
            if tracker['min_angle_rep'] > 100: tracker['last_rep_feedback'] = "GO DEEPER!"
    return tracker

def update_barbellrow_state(kpts, tracker):
    ls, rs, le, re, lw, rw = (kpts[KEYPOINT_DICT[name]] for name in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'])
    angle = (calculate_angle(ls, le, lw) + calculate_angle(rs, re, rw)) / 2
    if tracker['state'] == "down" and angle < 100:
        tracker['state'] = "up"
        rs, rh, ra = kpts[KEYPOINT_DICT['right_shoulder']], kpts[KEYPOINT_DICT['right_hip']], kpts[KEYPOINT_DICT['right_ankle']]
        tracker['initial_torso_angle'] = calculate_angle(rs, rh, ra)
    elif tracker['state'] == "up" and angle > 160:
        tracker['state'] = "down"; tracker['reps'] += 1; tracker['initial_torso_angle'] = None
    return tracker

def update_overheadpress_state(kpts, tracker):
    ls, rs, le, re, lw, rw = (kpts[KEYPOINT_DICT[name]] for name in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'])
    angle = (calculate_angle(ls, le, lw) + calculate_angle(rs, re, rw)) / 2
    if tracker['state'] == "down" and angle < 100:
        tracker['state'] = "up"; tracker['start_hip_y'] = kpts[KEYPOINT_DICT['right_hip']][1]
    elif tracker['state'] == "up" and angle > 160:
        tracker['state'] = "down"; tracker['reps'] += 1
    return tracker

def update_jumpingjack_state(kpts, tracker):
    lw, rw = kpts[KEYPOINT_DICT['left_wrist']], kpts[KEYPOINT_DICT['right_wrist']]
    ls, rs = kpts[KEYPOINT_DICT['left_shoulder']], kpts[KEYPOINT_DICT['right_shoulder']]
    avg_wrist_y = (lw[1] + rw[1]) / 2
    avg_shoulder_y = (ls[1] + rs[1]) / 2
    if tracker['state'] == "in" and avg_wrist_y < avg_shoulder_y: tracker['state'] = "out"
    elif tracker['state'] == "out" and avg_wrist_y > avg_shoulder_y:
        tracker['state'] = "in"; tracker['reps'] += 1
    return tracker

def analyze_fitness(video_path, exercise):
    print("analyze_fitness called")
    try:
        if isinstance(video_path, tuple):  # when input is (name, bytes)
            print("Saving uploaded file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_path[1])
                video_path = tmp.name
            print(f"Saved video to {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file!")
            return None, "Failed to open video."
        frame_w, frame_h, fps = (int(cap.get(p)) for p in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
        print(f"Video props: {frame_w}x{frame_h} at {fps} fps")
        if fps == 0:
            print("Error: Video FPS is 0!")
            return None, "Invalid video FPS (0)."
        out_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
        model = YOLO('yolov8s-pose.pt')
        initial_state = 'in' if exercise == 'JumpingJack' else ('down' if exercise in ['BarbellRow', 'OverheadPress'] else 'up')
        state_tracker = {'state': initial_state, 'reps': 0}
        update_state_func = globals()[f"update_{exercise.lower()}_state"]
        check_form_func = globals()[f"check_{exercise.lower()}_form"]

        feedback_timeline = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("No more frames or read failed.")
                break
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}")
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            form_feedback = []
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                kpts = results[0].keypoints.xy[0].cpu().numpy().tolist()
                state_tracker = update_state_func(kpts, state_tracker)
                form_feedback = check_form_func(kpts, state_tracker)
            else:
                form_feedback = ["No person detected"]
            if state_tracker.get('last_rep_feedback'): form_feedback.append(state_tracker['last_rep_feedback'])

            if form_feedback and not (len(form_feedback) == 1 and "No person detected" in form_feedback):
                feedback_timeline.append({
                    "frame": frame_idx,
                    "timestamp": frame_idx / fps,
                    "feedback": form_feedback.copy()
                })
            if state_tracker.get('last_rep_feedback'): state_tracker['last_rep_feedback'] = None
            out.write(annotated_frame)
            frame_idx += 1

        cap.release()
        out.release()
        print("Video processing complete.")

        # Prepare feedback for display
        feedback_display = ""
        last_time = -100
        for fb in feedback_timeline:
            if fb["timestamp"] - last_time > 1.0:
                t = fb["timestamp"]
                m, s = int(t // 60), int(t % 60)
                feedback_display += f"At {m:02d}:{s:02d} → " + " | ".join(fb["feedback"]) + "\n"
                last_time = t

        print("Returning output video and feedback.")
        return out_video_path, feedback_display.strip()
    except Exception as e:
        print(f"Exception in analyze_fitness: {e}")
        return None, f"Processing failed: {e}"
   

title = "🤖 AI Fitness Coach"
description = "Upload your exercise video and get real-time form feedback! Feedback is shown with time markers. (YOLOv8 pose, works best with clear videos and single person in frame.)"

exercises = ["BackSquat", "BarbellRow", "OverheadPress", "JumpingJack"]
demo = gr.Interface(
    fn=analyze_fitness,
    inputs=[
        gr.Video(label="Input Exercise Video (.mp4 preferred)"),
        gr.Dropdown(exercises, label="Exercise", value="BackSquat"),
    ],
    outputs=[
        gr.Video(label="Annotated Video With Feedback"),
        gr.Textbox(label="Feedback Timeline (Paused points)", lines=10),
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
