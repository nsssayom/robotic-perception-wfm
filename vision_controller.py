from controller import Robot
import cv2
import numpy as np
import csv
import os

# --- CONFIGURATION ---
VIDEO_PATH = "path_glare.mp4" 
LOG_FILE = "data_glare.csv"
DISPLAY_SCALE = 2.0  # Window size multiplier

# Perception Constants
YELLOW_LOWER = np.array([15, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])
# ---------------------

class ResearchRobot(Robot):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # Hardware Setup
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        self.ps_left = self.getDevice('left wheel sensor')
        self.ps_right = self.getDevice('right wheel sensor')
        self.ps_left.enable(self.timestep)
        self.ps_right.enable(self.timestep)
        
        # Telemetry Setup
        # Print where the file is being saved so you can find it
        cwd = os.getcwd()
        print(f"--- WRITING DATA TO: {cwd}/{LOG_FILE} ---")
        
        self.csv_file = open(LOG_FILE, 'w', newline='')
        self.logger = csv.writer(self.csv_file)
        self.logger.writerow(["Time", "Frame", "Error", "Motor_L", "Motor_R", "Odom_L", "Odom_R"])
        
        # Video Input
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.frame_count = 0
        self.WHEEL_RADIUS = 0.02

    def run(self):
        while self.step(self.timestep) != -1:
            # 1. READ SENSORS
            sim_time = self.getTime()
            l_enc = self.ps_left.getValue() * self.WHEEL_RADIUS
            r_enc = self.ps_right.getValue() * self.WHEEL_RADIUS
            
            # 2. READ VIDEO
            ret, frame = self.cap.read()
            if not ret:
                print("--- END OF SCENARIO ---")
                self.close_and_exit()
                break
            
            self.frame_count += 1

            # 3. PERCEPTION
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
            
            moments = cv2.moments(mask)
            height, width = mask.shape
            
            steering_error = 0.0
            turn_cmd = 0.0
            base_speed = 3.0
            
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                steering_error = (cx - width/2) / (width/2)
                turn_cmd = steering_error * 2.5 
                cv2.line(frame, (cx, 0), (cx, height), (0, 255, 0), 2)
            else:
                steering_error = 0.0 # Keep straight if lost (for now)
                turn_cmd = 0.0
            
            # 4. ACTUATION
            v_left = base_speed + turn_cmd
            v_right = base_speed - turn_cmd
            
            self.left_motor.setVelocity(v_left)
            self.right_motor.setVelocity(v_right)

            # 5. LOGGING
            self.logger.writerow([
                f"{sim_time:.4f}", self.frame_count, f"{steering_error:.4f}",
                f"{v_left:.2f}", f"{v_right:.2f}", f"{l_enc:.3f}", f"{r_enc:.3f}"
            ])

            # 6. VISUALIZATION
            display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("Cosmos Verification", display_frame)
            
            if cv2.waitKey(1) == ord('q'):
                self.close_and_exit()
                break
        
    def close_and_exit(self):
        # 1. Stop the robot
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        
        # 2. Save the data
        self.csv_file.flush()
        self.csv_file.close()
        
        # 3. Cleanup Video
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"Data saved to {LOG_FILE}. Controller exiting.")
        # REMOVED: self.simulationSetMode(self.MODE_PAUSE) 
        # (This caused the crash because standard Robots cannot pause the simulator)

# --- RUN ---
bot = ResearchRobot()
bot.run()