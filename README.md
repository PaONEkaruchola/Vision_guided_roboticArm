# Vision-Guided 3-DOF Robotic Arm

A real-time system that combines **YOLO object detection** with a **3-degree-of-freedom robotic arm simulation**. The camera feed locates a target object in the frame, converts its pixel location into physical robot-space coordinates, and drives a kinematic arm model to reach toward it — all visualized live in a side-by-side 2D/3D matplotlib window.

---

## Overview

The script (`guide_robot.py`) runs two things in lockstep, updated every animation frame:

1. **Computer vision pipeline** — reads video frames, runs a YOLO model, filters for specific object classes, and converts the detected object's pixel position into centimeters in the robot's coordinate frame.
2. **Robot arm kinematics** — a 3-DOF arm model that takes the target position and computes the joint angles needed to reach it (inverse kinematics), then computes the resulting arm pose for rendering (forward kinematics).

Both halves are tied together in `IntegratedSystem`, which owns the matplotlib figure, the video capture, the YOLO model, and the arm instance.

---
##Demo Video

## Features

- Real-time object detection and localization using a custom-trained YOLO model (`v11nb.pt`)
- Class filtering — only tracks specific object categories (sunglasses, book, watch, remote)
- Pixel-to-centimeter coordinate mapping from camera frame to robot workspace
- Exponential smoothing of the target position to reduce detection jitter
- Analytical (closed-form) forward and inverse kinematics for a 3-DOF arm
- Reach-limiting so unreachable targets are clamped to the arm's maximum radius
- Smoothed joint-angle interpolation for fluid simulated motion
- Live dual-pane visualization: annotated camera feed (left) + 3D arm pose and workspace boundary (right)
- Frame skipping to reduce inference load and improve throughput
- Looping video playback (restarts automatically when the source video ends)

---

## Robot Arm Model

The arm is modeled as three rigid links connected by three rotational joints, defined by `RobotArm3DOF`:

| Link | Length | Description |
|------|--------|-------------|
| `l1` (Base) | 15 cm | Vertical base riser, rotates about the Z axis |
| `l2` (Shoulder) | 10 cm | First arm segment, pitches up/down |
| `l3` (Elbow) | 20 cm | Second arm segment, pitches relative to the shoulder |

Joint angles are:
- **θ1** — base rotation (yaw, around Z)
- **θ2** — shoulder elevation angle
- **θ3** — elbow angle relative to the shoulder segment

### Forward Kinematics

`forward_kinematics(theta1, theta2, theta3)` computes the 3D position of every joint given the current joint angles, so the arm can be drawn.

- The base joint `j0` sits at the origin `(0, 0, 0)`.
- The shoulder joint `j1` sits directly above the base at height `l1`.
- The elbow joint `j2` is found by extending link `l2` from the shoulder, first computing its horizontal reach `r2 = l2·cos(θ2)`, then projecting that reach into X/Y using the base rotation `θ1`, and adding `l2·sin(θ2)` to the height.
- The end effector `j3` is found the same way, extending link `l3` from the elbow using the **combined** angle `θ2 + θ3` (since the elbow angle is relative to the shoulder segment, not the world frame).

This is a standard forward-kinematics chain: each joint's position is the previous joint's position plus a rotated/projected link vector.

### Inverse Kinematics

`inverse_kinematics(x, y, z)` computes the joint angles needed to place the end effector at a target point. This uses a classic 2-link planar IK solution combined with a base rotation:

1. **Base angle (θ1):** found directly from the target's X/Y position with `atan2(y, x)` — this points the whole arm toward the target regardless of height.
2. **Reduce to a 2D problem:** since θ1 already handles azimuth, the shoulder/elbow solve happens in a vertical plane. The target is expressed as a radial distance `r_target` (horizontal distance from the base axis) and a height `z_target` (target height minus the base link length `l1`).
3. **Reach clamping:** if the straight-line distance `D` to the target exceeds the arm's maximum reach (`l2 + l3`), the target is scaled down proportionally so the arm reaches as far as it can without producing an invalid (out-of-domain) triangle.
4. **Law of cosines (triangle solve):** with the base-to-target distance `D` and link lengths `l2`/`l3` forming a triangle, the law of cosines gives:
   - `β` — the angle between the shoulder link and the line to the target, used together with the elevation angle `α = atan2(z_target, r_target)` to get the **shoulder angle θ2 = α + β**.
   - `γ` — the interior angle at the elbow, which is converted to the **elbow angle θ3 = γ − π** (since a fully extended arm corresponds to γ = π, this reframes it as 0 offset).
5. All cosine arguments are clipped to `[-1, 1]` before `arccos` to guard against floating-point rounding pushing values slightly outside the valid domain.

If any part of the solve fails, the method falls back to `θ2 = θ3 = 0` rather than crashing.

---

## Computer Vision Pipeline

Implemented in `IntegratedSystem.process_vision()`, run once per animation frame (with frame skipping applied).

### 1. Frame acquisition
- Reads a frame from the video source (`cv2.VideoCapture`).
- If the video ends, it automatically seeks back to frame 0 and continues — creating a looping feed.

### 2. Frame skipping
- Detection only runs every `skip_frames + 1` frames (currently every 3rd frame) to reduce YOLO inference load and keep the simulation responsive.

### 3. Object detection
- The frame is passed to the YOLO model (`self.model(frame, stream=True)`).
- Detections are filtered by:
  - **Confidence threshold:** only boxes with confidence > 0.5 are kept.
  - **Class filter (`ALLOWED_CLASSES`):** only specific object classes are tracked (indices `[0, 1, 3, 4]`, corresponding to sunglasses, book, watch, and remote in the custom model's class list).
- The first qualifying detection in the frame is selected as the target; a green bounding box and a text label with its computed location are drawn onto the frame.

### 4. Pixel → centimeter → robot-space conversion
This is the key step that bridges vision and robotics:

1. **Pixel-to-cm ratio:** the physical workspace is defined as a `FIELD_WIDTH_CM × FIELD_HEIGHT_CM` square (30×30 cm) mapped onto the frame's pixel dimensions, giving `ratio_x` and `ratio_y` (pixels per cm).
2. **Offset from image center:** the object's bounding-box center `(cx, cy)` is converted to a signed pixel offset from the image center `(img_cx, img_cy)`. The Y offset is flipped (`img_cy - cy`) so that "up" in the image corresponds to a positive direction, matching a standard right-handed coordinate convention rather than image-space's downward Y axis.
3. **Convert offset to cm:** the pixel offsets are divided by the pixel-per-cm ratios to get `cam_dx_cm` / `cam_dy_cm`.
4. **Map into robot coordinates:** the robot's origin is at a corner of the workspace, not its center, so the camera-space offset is added to the workspace center (`FRAME_CENTER_X_ROBOT`, `FRAME_CENTER_Y_ROBOT`) to produce the final robot-frame target `(rob_x, rob_y)`. Note the axes are swapped between camera and robot space (camera's vertical offset maps to robot X, camera's horizontal offset maps to robot Y), reflecting the arm's mounting orientation relative to the camera.
5. A fixed height `rob_z = 2.0` cm is assumed for all detected objects (the system doesn't estimate height from vision).

### 5. Target smoothing
- Rather than snapping directly to each new detection, the target position is updated as an exponential moving average: `target_pos = 0.5 * target_pos + 0.5 * new_target`. This reduces jitter from frame-to-frame detection noise while still tracking a moving object reasonably quickly.

---

## Simulation Loop

`update_simulation()` runs on every animation tick (`FuncAnimation`, ~1 ms interval):

1. Calls `process_vision()` to update the detected frame and target position.
2. Updates the 2D camera-feed subplot with the latest annotated frame.
3. Solves inverse kinematics for the current `target_pos`.
4. Smooths the joint angles toward the IK solution using a proportional step (`current_angles += error * 0.5`) instead of jumping instantly — this produces visibly smooth arm motion rather than teleporting.
5. Runs forward kinematics on the smoothed angles to get joint positions for rendering.
6. Updates the 3D arm plot (link positions) and the target marker.
7. Re-applies fixed axis limits each frame (required because `FuncAnimation` with `blit=False` redraws the full axes).

---

## Visualization

- **Left pane (2D):** live camera feed with detection bounding box and coordinate label overlaid.
- **Right pane (3D):** 
  - The arm rendered as connected joints (`'o-'` markers/lines).
  - A red star marking the current smoothed target position.
  - A green dashed square showing the physical workspace boundary the camera can see, positioned in robot coordinates.

---

## Requirements

```
opencv-python
numpy
matplotlib
ultralytics
```

A trained YOLO weights file (`v11nb.pt`) and a video source (`cv2.mp4` by default, or a live camera index) must be available in the working directory / accessible path.

## Usage

```bash
python guide_robot.py
```

By default this loads `cv2.mp4` as the video source and opens the visualization window. To use a live webcam instead, change `video_source` to a camera index (e.g. `0`) when constructing `IntegratedSystem`.

## Configuration

Key parameters can be adjusted in `IntegratedSystem.__init__`:

| Parameter | Purpose |
|---|---|
| `FIELD_WIDTH_CM` / `FIELD_HEIGHT_CM` | Physical size of the camera's field of view, in cm |
| `ALLOWED_CLASSES` | YOLO class indices to track |
| `skip_frames` | Number of frames to skip between detection runs |
| `link_lengths` | Arm segment lengths passed to `RobotArm3DOF` |

## Known Limitations

- Assumes a fixed, hardcoded object height (`z = 2.0 cm`) since depth isn't estimated from a single camera.
- Only the first qualifying detection per frame is tracked; multiple simultaneous objects aren't handled.
- The arm's IK is a purely geometric solve with no collision checking or joint-limit enforcement beyond the reach clamp.
- `interval=1` in `FuncAnimation` requests a ~1 ms redraw, but actual frame rate is bounded by YOLO inference and rendering time.
