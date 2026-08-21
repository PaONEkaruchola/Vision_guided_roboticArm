import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ultralytics import YOLO
import os

class RobotArm3DOF:
    #link lengths: Base=15, Shoulder=10, Elbow=20  
    def __init__(self, link_lengths=[15, 10, 20]):
        self.l1 = link_lengths[0]
        self.l2 = link_lengths[1]
        self.l3 = link_lengths[2]

    def forward_kinematics(self, theta1, theta2, theta3):
        j0 = np.array([0, 0, 0])
        j1 = np.array([0, 0, self.l1])

        r2 = self.l2 * np.cos(theta2)
        x2 = r2 * np.cos(theta1)
        y2 = r2 * np.sin(theta1)
        z2 = self.l1 + self.l2 * np.sin(theta2)
        j2 = np.array([x2, y2, z2])

        theta_sum = theta2 + theta3
        r3 = self.l3 * np.cos(theta_sum)
        x3 = x2 + r3 * np.cos(theta1)
        y3 = y2 + r3 * np.sin(theta1)
        z3 = z2 + self.l3 * np.sin(theta_sum)
        j3 = np.array([x3, y3, z3])

        return np.array([j0, j1, j2, j3])

    def inverse_kinematics(self, target_x, target_y, target_z):
        theta1 = np.arctan2(target_y, target_x)

        r_target = np.sqrt(target_x ** 2 + target_y ** 2)
        z_target = target_z - self.l1

        D = np.sqrt(r_target ** 2 + z_target ** 2)
        max_reach = self.l2 + self.l3

        if D > max_reach:
            scale = max_reach / D
            r_target *= scale
            z_target *= scale
            D = max_reach

        try:
            alpha = np.arctan2(z_target, r_target)
            cos_beta = (self.l2 ** 2 + D ** 2 - self.l3 ** 2) / (2 * self.l2 * D)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            beta = np.arccos(cos_beta)
            theta2 = alpha + beta

            cos_gamma = (self.l2 ** 2 + self.l3 ** 2 - D ** 2) / (2 * self.l2 * self.l3)
            cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
            gamma = np.arccos(cos_gamma)
            theta3 = gamma - np.pi
        except Exception:
            theta2, theta3 = 0, 0

        return theta1, theta2, theta3


class IntegratedSystem:
    def __init__(self, video_source="cv2.mp4"):
        #frame size in metric units cm
        self.FIELD_WIDTH_CM = 30.0
        self.FIELD_HEIGHT_CM = 30.0
        
        #robot is at (0,0). Workspace Center is offset so (0,0) is the corner.
        self.FRAME_CENTER_X_ROBOT = self.FIELD_HEIGHT_CM / 2.0
        self.FRAME_CENTER_Y_ROBOT = self.FIELD_WIDTH_CM / 2.0
        #object classes focused on sunglasses, book, watch, remote
        self.ALLOWED_CLASSES = [0,1,3,4] 

        print("loading custume yolo model...")
        self.model = YOLO("v11nb.pt")
        self.robot = RobotArm3DOF(link_lengths=[15, 10, 20])
        self.cap = cv2.VideoCapture(video_source)
        #decereasing fps for increasing model efficieny
        self.frame_count = 0
        self.skip_frames = 2
        
        #intial position and parameters
        self.target_pos = np.array([15.0, 14.5, 5.0])
        self.current_angles = np.array([0.0, np.pi / 2, -np.pi / 2])
        self.detected_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.object_info = "Searching..."

        self.fig = plt.figure(figsize=(15, 7))

        #2d view
        self.ax_cam = self.fig.add_subplot(1, 2, 1)
        self.cam_display = self.ax_cam.imshow(self.detected_frame)
        self.ax_cam.set_title(f"2D Feed")
        self.ax_cam.axis('off')

        #3d robot simulation view
        self.ax_3d = self.fig.add_subplot(1, 2, 2, projection='3d')
        self.line, = self.ax_3d.plot([], [], [], 'o-', lw=6, markersize=10, color="#191818", zorder=10)
        self.target_marker, = self.ax_3d.plot([], [], [], 'r*', markersize=7, label='Target', zorder=10)

        #these green square represent the camera frame
        x_min = self.FRAME_CENTER_X_ROBOT - (self.FIELD_HEIGHT_CM / 2)
        x_max = self.FRAME_CENTER_X_ROBOT + (self.FIELD_HEIGHT_CM / 2)
        y_min = self.FRAME_CENTER_Y_ROBOT - (self.FIELD_WIDTH_CM / 2)
        y_max = self.FRAME_CENTER_Y_ROBOT + (self.FIELD_WIDTH_CM / 2)
        #vertex
        rect_x = [x_min, x_max, x_max, x_min, x_min]
        rect_y = [y_min, y_min, y_max, y_max, y_min]
        rect_z = [0, 0, 0, 0, 0]
        
        self.ax_3d.plot(rect_x, rect_y, rect_z, 'g--', lw=2, label="Workspace")

    def process_vision(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret or frame is None: return

        self.frame_count += 1
        
        if self.frame_count % (self.skip_frames + 1) == 0:
            h_px, w_px, _ = frame.shape
            #getting pixel to cm ratio
            img_cx, img_cy = w_px // 2, h_px // 2
            ratio_x = w_px / self.FIELD_WIDTH_CM
            ratio_y = h_px / self.FIELD_HEIGHT_CM
            #getting bounding box from model
            results = self.model(frame, verbose=False, stream=True)
            new_target = None

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if float(box.conf[0]) > 0.5 and int(box.cls[0]) in self.ALLOWED_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        #making bounding box around the detected object
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        #converting center of the object to cm
                        px_dx = cx - img_cx
                        px_dy = img_cy - cy 
                        cam_dx_cm = px_dx / ratio_x
                        cam_dy_cm = px_dy / ratio_y
                        
                        rob_x = self.FRAME_CENTER_X_ROBOT + cam_dy_cm
                        rob_y = self.FRAME_CENTER_Y_ROBOT + cam_dx_cm
                        rob_z = 2.0

                        new_target = np.array([rob_x, rob_y, rob_z])
                        self.object_info = f"Loc: ({rob_x:.1f}, {rob_y:.1f})"
                        cv2.putText(frame, self.object_info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break
                if new_target is not None: break

            if new_target is not None:
                # Fast movement gain (0.5)
                self.target_pos = self.target_pos * 0.5 + new_target * 0.5
        
        self.detected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def update_simulation(self, frame):
        self.process_vision()

        #update 2D Display
        if self.detected_frame is not None and self.detected_frame.shape[0] > 0:
            self.cam_display.set_data(self.detected_frame)

        #update robot
        target_angles = self.robot.inverse_kinematics(*self.target_pos)
        error = np.array(target_angles) - self.current_angles
        
        #high gain for the speed of the simulation
        self.current_angles += error * 0.5

        joints = self.robot.forward_kinematics(*self.current_angles)
        xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]

        self.line.set_data(xs, ys)
        self.line.set_3d_properties(zs)
        self.target_marker.set_data([self.target_pos[0]], [self.target_pos[1]])
        self.target_marker.set_3d_properties([self.target_pos[2]])

        #limiting the axis 
        self.ax_3d.set_xlim(-10, 35)
        self.ax_3d.set_ylim(-10, 35)
        self.ax_3d.set_zlim(0, 40)
        self.ax_3d.set_xlabel("X (cm)")
        self.ax_3d.set_ylabel("Y (cm)")

        return self.line, self.target_marker, self.cam_display

    def run(self):
        #update for every 1ms
        anim = FuncAnimation(self.fig, self.update_simulation, interval=1, blit=False)
        plt.show()
        self.cap.release()

if __name__ == "__main__":
    video_file = "cv2.mp4" 
    try:
        app = IntegratedSystem(video_source=video_file)
        app.run()
    except Exception as e:
        print(f"Detailed Error: {e}")