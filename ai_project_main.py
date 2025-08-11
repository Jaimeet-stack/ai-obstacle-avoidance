#!/usr/bin/env python3
"""
ai_indoor_avoidance_pro.py

Upgraded advanced single-file AI-powered indoor obstacle avoidance prototype.

New features added:
 - Simulated 2D LiDAR scan fused into occupancy grid
 - Hungarian assignment (if scipy available) for better multi-object association
 - Flask dashboard (MJPEG stream + telemetry JSON)
 - Voice alerts (pyttsx3) for critical events
 - Map snapshot export (press 'm')
 - Clean shutdown of Flask + detector threads

Run:
    python ai_indoor_avoidance_pro.py
"""

import time, math, os, threading, argparse, queue, csv
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import pyttsx3
    TTS_OK = True
except Exception:
    TTS_OK = False

try:
    from flask import Flask, Response, render_template_string, jsonify
    FLASK_OK = True
except Exception:
    FLASK_OK = False

REALSR = False
try:
    import pyrealsense2 as rs
    REALSR = True
except Exception:
    REALSR = False
CAM_W, CAM_H = 640, 480
GRID_SIZE_M = 8.0
GRID_RES = 0.05
GRID_DIM = int(GRID_SIZE_M / GRID_RES)
ROBOT_RADIUS = 0.25
YOLO_CONF = 0.35
MAX_DETECTIONS = 12
FPS_TARGET = 15.0

MAX_LIN_V = 0.6
MAX_ANG_V = 1.2
PID_KP, PID_KI, PID_KD = 1.0, 0.0, 0.08

SAFE_DIST = 0.5
LOG_CSV = "events_log.csv"
LIDAR_RAYS = 180  

def clamp(x,a,b): return max(a, min(b, x))
def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

def world_to_grid(x, y, world_origin=(-GRID_SIZE_M/2, -GRID_SIZE_M/2), res=GRID_RES):
    gx = int((x - world_origin[0]) / res)
    gy = int((y - world_origin[1]) / res)
    return gx, gy

def grid_to_world(gx, gy, world_origin=(-GRID_SIZE_M/2, -GRID_SIZE_M/2), res=GRID_RES):
    x = world_origin[0] + gx * res + res/2.0
    y = world_origin[1] + gy * res + res/2.0
    return x, y

def log_event(kind, details):
    header = ["timestamp","kind","details"]
    write_header = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow([now(), kind, details])


class Sensor:
    def __init__(self, mode='AUTO'):
        self.mode = mode
        self.realsense_running = False
        self.cap = None
        self.pipeline = None
        self.align = None
        self.fx = 600; self.fy = 600; self.cx = CAM_W/2; self.cy = CAM_H/2
        if mode == 'AUTO':
            self.mode = 'REAL' if REALSR else 'SIM'
        if self.mode == 'REAL' and REALSR:
            self._start_realsense()
        else:
            self._start_webcam()

    def _start_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.realsense_running = True
        cs = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy, self.cx, self.cy = cs.fx, cs.fy, cs.ppx, cs.ppy
        print("[Sensor] RealSense started.")

    def _start_webcam(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        if not self.cap.isOpened(): raise RuntimeError("Webcam not available")
        print("[Sensor] Webcam started (SIM mode).")

    def read(self):
        if self.realsense_running:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame: return None, None, None
            color_img = np.asanyarray(color_frame.get_data()); depth_img = np.asanyarray(depth_frame.get_data())
            depth_m = (depth_img.astype(np.float32) * depth_frame.get_units()) if hasattr(depth_frame,'get_units') else (depth_img.astype(np.float32)/1000.0)
            return color_img, depth_m, depth_frame
        else:
            ret, frame = self.cap.read()
            if not ret: return None, None, None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_m = 2.0 + (255 - gray).astype(np.float32)/255.0 * 4.0
            return frame, depth_m, None

    def stop(self):
        try:
            if self.realsense_running: self.pipeline.stop()
            if self.cap: self.cap.release()
        except Exception: pass


import threading, queue
class DetectorThread(threading.Thread):
    def __init__(self, model_name='yolov8n.pt', conf=YOLO_CONF):
        super().__init__(daemon=True)
        self.model = YOLO(model_name)
        self.conf = conf
        self.frame_q = queue.Queue(maxsize=1)
        self.results_lock = threading.Lock()
        self.latest = []
        self.running = True
        print("[Detector] YOLO loaded.")

    def submit(self, frame):
        try:
            if not self.frame_q.full(): self.frame_q.put(frame.copy(), block=False)
        except Exception: pass

    def get_latest(self):
        with self.results_lock: return list(self.latest)

    def run(self):
        while self.running:
            try:
                frame = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            res = self.model.predict(source=frame, conf=self.conf, imgsz=640, max_det=MAX_DETECTIONS, verbose=False)
            dets = []
            if len(res) > 0:
                r = res[0]; boxes = getattr(r, 'boxes', None)
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    for (x1,y1,x2,y2), c, cl in zip(xyxy, confs, cls):
                        dets.append((int(x1),int(y1),int(x2),int(y2), float(c), int(cl)))
            with self.results_lock: self.latest = dets

    def stop(self): self.running = False


class Perception:
    def __init__(self, detector: DetectorThread, intrinsics=(600,600,CAM_W/2,CAM_H/2)):
        self.detector = detector
        self.fx, self.fy, self.cx, self.cy = intrinsics

    def get_detections(self): return self.detector.get_latest()

    def detections_to_points(self, detections, depth_m):
        pts = []
        for (x1,y1,x2,y2,conf,cls) in detections:
            cx_pix = int((x1+x2)/2); cy_pix = int((y1+y2)/2)
            if depth_m is None:
                d=3.0
            else:
                cxp = clamp(cx_pix, 0, depth_m.shape[1]-1); cyp = clamp(cy_pix, 0, depth_m.shape[0]-1)
                d = float(depth_m[cyp, cxp])
                if d<=0 or np.isnan(d) or d>20:
                    patch = depth_m[max(0,cyp-3):min(depth_m.shape[0],cyp+4), max(0,cxp-3):min(depth_m.shape[1],cxp+4)]
                    if patch.size==0: d=3.0
                    else:
                        med = np.median(patch); d = float(med) if med>0 else 3.0
            X = (cx_pix - self.cx) * d / self.fx
            forward = d; left = -X
            pts.append((forward, left, cls, conf))
        return pts

    def simulate_lidar_from_depth(self, depth_m, max_range=6.0, rays=LIDAR_RAYS):
        """
        Create a 2D LiDAR-like scan in robot frame by sampling depth_m along rays
        Returns list of (range, angle) in robot frame (angles in radians, forward=0)
        """
        if depth_m is None:
            return []
        h, w = depth_m.shape
        # sample angles from -pi/2 to +pi/2 (front hemisphere); you can expand to full 360 if desired
        angles = np.linspace(-math.pi/1.5, math.pi/1.5, rays)
        ranges = []
        for a in angles:
            # sample points along ray in image coordinates using pinhole approx
            # For simplicity, sample along depth image row center
            best = max_range
            for r in np.linspace(0.2, max_range, 120):
                # project point (forward r, left r*tan(a)) -> pixel coords
                x_cam = - (r * math.tan(a))  # camera x to right
                z_cam = r
                u = int((x_cam * self.fx / z_cam) + self.cx)
                v = int(self.cy)  # sample around center row
                if 0 <= u < w and 0 <= v < h:
                    d = float(depth_m[v, u])
                    if d > 0 and d < best:
                        best = d
                        break
            ranges.append((best, a))
        return ranges


class OccupancyGridMapper:
    def __init__(self, grid_dim=GRID_DIM, res=GRID_RES, world_size=GRID_SIZE_M):
        self.grid_dim = grid_dim; self.res = res
        self.world_origin = (-world_size/2.0, -world_size/2.0)
        self.grid = np.zeros((grid_dim, grid_dim), dtype=np.uint8)
        self.logodds = np.zeros((grid_dim, grid_dim), dtype=np.float32)
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.grid.fill(0); self.logodds.fill(0.0)

    def integrate_points(self, points, robot_pose=(0,0,0)):
        rx, ry, rtheta = robot_pose; cos_t = math.cos(rtheta); sin_t = math.sin(rtheta)
        with self.lock:
            for (fwd, left, *_rest) in points:
                wx = rx + fwd * cos_t - left * sin_t
                wy = ry + fwd * sin_t + left * cos_t
                gx, gy = world_to_grid(wx, wy, self.world_origin, self.res)
                if 0 <= gx < self.grid_dim and 0 <= gy < self.grid_dim:
                    self.logodds[gx, gy] += 1.0
                    inflation = int(math.ceil(ROBOT_RADIUS / self.res))
                    for dx in range(-inflation, inflation+1):
                        for dy in range(-inflation, inflation+1):
                            nx, ny = gx+dx, gy+dy
                            if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim:
                                self.logodds[nx, ny] += 0.25
            self.logodds *= 0.998
            self.logodds = np.clip(self.logodds, 0.0, 50.0)
            self.grid = (self.logodds > 0.5).astype(np.uint8)

    def integrate_lidar(self, lidar_scan, robot_pose=(0,0,0), max_range=6.0):
        rx, ry, rtheta = robot_pose
        with self.lock:
            for r,a in lidar_scan:
                if r <= 0 or r > max_range: continue
                # convert to world
                ang = rtheta + a
                wx = rx + r * math.cos(ang); wy = ry + r * math.sin(ang)
                gx, gy = world_to_grid(wx, wy, self.world_origin, self.res)
                if 0 <= gx < self.grid_dim and 0 <= gy < self.grid_dim:
                    self.logodds[gx, gy] += 0.8
                    inflation = int(math.ceil(ROBOT_RADIUS / self.res))
                    for dx in range(-inflation, inflation+1):
                        for dy in range(-inflation, inflation+1):
                            nx, ny = gx+dx, gy+dy
                            if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim:
                                self.logodds[nx, ny] += 0.2
            self.logodds *= 0.997
            self.logodds = np.clip(self.logodds, 0.0, 50.0)
            self.grid = (self.logodds > 0.5).astype(np.uint8)

    def get_grid_copy(self):
        with self.lock:
            return self.grid.copy(), self.logodds.copy()


class AStarPlanner:
    def __init__(self, risk_weight=4.0):
        self.risk_weight = risk_weight
    def astar(self, grid, risk, start_cell, goal_cell):
        rows, cols = grid.shape
        start = tuple(start_cell); goal = tuple(goal_cell)
        if grid[start] == 1 or grid[goal] == 1: return None
        def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
        import heapq
        open_set = [(h(start,goal), 0, start, None)]
        came_from = {}; gscore = {start:0}; visited = set()
        while open_set:
            f,g,node,parent = heapq.heappop(open_set)
            if node in visited: continue
            visited.add(node); came_from[node]=parent
            if node==goal:
                path=[]; cur=node
                while cur:
                    path.append(cur); cur=came_from[cur]
                return list(reversed(path))
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nb=(node[0]+dx,node[1]+dy)
                if not (0<=nb[0]<rows and 0<=nb[1]<cols): continue
                if grid[nb]==1: continue
                move_cost = math.hypot(dx,dy)
                risk_cost = (risk[nb] / (np.max(risk)+1e-6)) * self.risk_weight
                tentative_g = g + move_cost + risk_cost
                if tentative_g < gscore.get(nb, float('inf')):
                    gscore[nb]=tentative_g
                    heapq.heappush(open_set, (tentative_g + h(nb,goal), tentative_g, nb, node))
        return None


class LocalPlanner:
    def __init__(self, max_lin=MAX_LIN_V, max_ang=MAX_ANG_V):
        self.max_lin=max_lin; self.max_ang=max_ang
    def plan(self, robot_pose, grid, global_path, world_origin, res):
        if not global_path: return 0.0, 0.0
        rx,ry,rtheta = robot_pose
        best_idx=0; best_dist=1e9
        for i,(gx,gy) in enumerate(global_path):
            px,py = grid_to_world(gx,gy,world_origin,res)
            d = math.hypot(px-rx, py-ry)
            if d < best_dist: best_dist=d; best_idx=i
        lookahead = min(best_idx+6, len(global_path)-1)
        tgx,tgy = global_path[lookahead]
        tx,ty = grid_to_world(tgx,tgy,world_origin,res)
        angle_to_target = math.atan2(ty-ry, tx-rx)
        ang_err = (angle_to_target - rtheta + math.pi) % (2*math.pi) - math.pi
        v_des = self.max_lin * max(0.15, (1 - abs(ang_err)/(math.pi/2)))
        steps = int(1.0 / res)
        min_clear = 1.0
        for s in range(1, steps+1):
            fx = rx + s*res*math.cos(rtheta); fy = ry + s*res*math.sin(rtheta)
            gx,gy = world_to_grid(fx,fy,world_origin,res)
            if 0<=gx<grid.shape[0] and 0<=gy<grid.shape[1]:
                if grid[gx,gy]==1:
                    min_clear = s*res; break
        if min_clear < 0.6: v_des *= 0.2
        omega_des = clamp(ang_err*2.0, -self.max_ang, self.max_ang)
        return v_des, omega_des

class PIDController:
    def __init__(self,kp=PID_KP,ki=PID_KI,kd=PID_KD):
        self.kp=kp; self.ki=ki; self.kd=kd; self.last_err_v=0.0; self.last_err_w=0.0; self.int_v=0.0; self.int_w=0.0
    def step(self, tv, tw, cv, cw, dt):
        ev=tv-cv; ew=tw-cw
        self.int_v += ev*dt; self.int_w += ew*dt
        dv=(ev-self.last_err_v)/max(dt,1e-6); dw=(ew-self.last_err_w)/max(dt,1e-6)
        out_v=self.kp*ev + self.ki*self.int_v + self.kd*dv
        out_w=self.kp*ew + self.ki*self.int_w + self.kd*dw
        self.last_err_v=ev; self.last_err_w=ew
        out_v=clamp(out_v, -MAX_LIN_V, MAX_LIN_V); out_w=clamp(out_w, -MAX_ANG_V, MAX_ANG_V)
        return out_v, out_w

class RobotSim:
    def __init__(self,x=0.0,y=0.0,theta=0.0):
        self.x=x; self.y=y; self.theta=theta; self.v=0.0; self.w=0.0
    def apply(self, v_cmd, w_cmd, dt):
        self.v=v_cmd; self.w=w_cmd
        self.x += self.v*math.cos(self.theta)*dt; self.y += self.v*math.sin(self.theta)*dt; self.theta += self.w*dt
        self.theta = (self.theta + math.pi) % (2*math.pi) - math.pi


class KFTrack:
    _next_id = 0
    def __init__(self, x, y, dt=0.1):
        self.id = KFTrack._next_id; KFTrack._next_id += 1
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 1.0; self.dt = dt; self.last_update = time.time()
    def predict(self, dt=None):
        if dt is None: dt=self.dt
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        Q = np.eye(4) * 0.01
        self.x = F.dot(self.x); self.P = F.dot(self.P).dot(F.T) + Q
    def update(self, zx, zy):
        H = np.array([[1,0,0,0],[0,1,0,0]]); R = np.eye(2)*0.5
        z = np.array([zx, zy]); y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R; K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y); self.P = (np.eye(4) - K.dot(H)).dot(self.P); self.last_update = time.time()
    def pos(self): return (float(self.x[0]), float(self.x[1]))
    def vel(self): return (float(self.x[2]), float(self.x[3]))

class TrackerManager:
    def __init__(self):
        self.tracks=[]; self.max_age=2.0; self.dist_thresh=1.0
    def step(self, measurements, dt):
        for t in self.tracks: t.predict(dt)
        if len(self.tracks)==0:
            for (mx,my) in measurements: self.tracks.append(KFTrack(mx,my,dt=max(dt,0.05))); return
        
        if len(measurements)>0 and SCIPY_OK:
            cost = np.zeros((len(self.tracks), len(measurements)), dtype=float)
            for i,t in enumerate(self.tracks):
                for j,(mx,my) in enumerate(measurements):
                    cost[i,j] = math.hypot(t.x[0]-mx, t.x[1]-my)
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_meas = set()
            assigned_tracks = set()
            for r,c in zip(row_ind, col_ind):
                if cost[r,c] < self.dist_thresh:
                    self.tracks[r].update(measurements[c][0], measurements[c][1])
                    assigned_meas.add(c); assigned_tracks.add(r)
           
            for j,(mx,my) in enumerate(measurements):
                if j not in assigned_meas: self.tracks.append(KFTrack(mx,my, dt=max(dt,0.05)))
        else:
           
            assigned = set()
            for mx,my in measurements:
                best=None; bestd=1e9
                for t in self.tracks:
                    d = math.hypot(t.x[0]-mx, t.x[1]-my)
                    if d < bestd: bestd=d; best=t
                if best and bestd < self.dist_thresh: best.update(mx,my)
                else: self.tracks.append(KFTrack(mx,my, dt=max(dt,0.05)))
        nowt = time.time()
        self.tracks = [t for t in self.tracks if (nowt - t.last_update) <= self.max_age]
    def get_predictions(self, horizon=0.8):
        preds=[]
        for t in self.tracks:
            px = t.x[0] + t.x[2]*horizon; py = t.x[1] + t.x[3]*horizon
            preds.append((px,py,t.id))
        return preds


def cbf_filter(robot_state, cmd_v, cmd_w, obstacle_preds, safe_dist=SAFE_DIST):
    xs, ys, th = robot_state
    v_safe = cmd_v
    for ox, oy, _id in obstacle_preds:
        dx = ox - xs; dy = oy - ys; d = math.hypot(dx,dy)
        if d <= 0.001: continue
        bearing = math.atan2(dy, dx)
        rel = (bearing - th + math.pi) % (2*math.pi) - math.pi
        approach = math.cos(rel)
        if approach > 0.2:
            margin = 0.15
            max_v_allowed = math.sqrt(max(0.0, (d - safe_dist - margin) * 2.0))
            v_safe = min(v_safe, max_v_allowed)
    v_safe = clamp(v_safe, -MAX_LIN_V, MAX_LIN_V)
    return v_safe, cmd_w


tts_engine = None
if TTS_OK:
    try:
        tts_engine = pyttsx3.init(); tts_engine.setProperty("rate",160)
    except Exception:
        tts_engine = None

def speak(text):
    if tts_engine:
        try:
            tts_engine.say(text); tts_engine.runAndWait()
        except Exception:
            pass


app = None
def start_flask_thread(get_frame_func, telemetry_func, host="0.0.0.0", port=5000):
    if not FLASK_OK:
        print("[Dashboard] Flask not available. Install flask to use dashboard.")
        return None
    global app
    app = Flask(__name__)

    @app.route("/")
    def index():
        html = """
        <html><head><title>AI Obstacle Avoidance Dashboard</title></head>
        <body style="background:#111;color:#ddd;font-family:Arial">
        <h2>AI Obstacle Avoidance Dashboard</h2>
        <img src="/video_feed" width="720"><br>
        <pre id="tele"></pre>
        <script>
        async function fetchTele(){ let r=await fetch('/telemetry'); let j=await r.json(); document.getElementById('tele').innerText = JSON.stringify(j, null, 2); }
        setInterval(fetchTele, 500);
        </script>
        </body></html>
        """
        return render_template_string(html)

    def gen():
        while True:
            frame = get_frame_func()
            if frame is None:
                time.sleep(0.05); continue
            _, jpg = cv2.imencode('.jpg', frame)
            b = jpg.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b + b'\r\n')
    @app.route('/video_feed')
    def video_feed(): return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    @app.route('/telemetry')
    def telemetry(): return jsonify(telemetry_func())
    th = threading.Thread(target=lambda: app.run(host=host,port=port,debug=False,threaded=True), daemon=True)
    th.start()
    print(f"[Dashboard] started at http://{host}:{port}")
    return th


class System:
    def __init__(self, mode='AUTO', model='yolov8n.pt'):
        print("[System] starting")
        self.sensor = Sensor(mode)
        self.detector = DetectorThread(model, conf=YOLO_CONF)
        self.detector.start()
        self.perc = Perception(self.detector, intrinsics=(self.sensor.fx,self.sensor.fy,self.sensor.cx,self.sensor.cy))
        self.mapper = OccupancyGridMapper()
        self.astar = AStarPlanner(risk_weight=4.0)
        self.local = LocalPlanner()
        self.pid = PIDController()
        self.robot = RobotSim(x=0.0,y=0.0,theta=0.0)
        self.tracker = TrackerManager()
        self.goal_world = (2.0, 2.0)
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
      
        if FLASK_OK:
            start_flask_thread(self.get_dashboard_frame, self.get_telemetry, host="0.0.0.0", port=5000)

    def get_dashboard_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def get_telemetry(self):
        return {"pose":(self.robot.x, self.robot.y, self.robot.theta), "goal":self.goal_world,
                "tracks":[{"id":t.id,"pos":t.pos(),"vel":t.vel()} for t in self.tracker.tracks]}

    def set_goal(self, gx, gy):
        self.goal_world = (gx, gy); log_event("goal_set", f"{gx:.2f},{gy:.2f}")

    def shutdown(self):
        print("[System] shutdown requested")
        self.running = False
        self.detector.stop()
        try: self.sensor.stop()
        except: pass

    def run(self):
        last_t=time.time()
        while self.running:
            start = time.time()
            frame, depth_m, depth_frame = self.sensor.read()
            if frame is None:
                print("[System] no frame, exiting"); break
            
            self.detector.submit(frame)
            detections = self.perc.get_detections()
            pts = self.perc.detections_to_points(detections, depth_m)
            
            lidar = self.perc.simulate_lidar_from_depth(depth_m, max_range=6.0, rays=LIDAR_RAYS)
            
            meas_world=[]
            for (fwd,left,cls,conf) in pts:
                rx,ry,rt = self.robot.x, self.robot.y, self.robot.theta
                wx = rx + fwd*math.cos(rt) - left*math.sin(rt)
                wy = ry + fwd*math.sin(rt) + left*math.cos(rt)
                meas_world.append((wx,wy))
            dt = max(0.01, time.time()-last_t); last_t=time.time()
            self.tracker.step(meas_world, dt)
            
            self.mapper.integrate_points(pts, robot_pose=(self.robot.x,self.robot.y,self.robot.theta))
            self.mapper.integrate_lidar(lidar, robot_pose=(self.robot.x,self.robot.y,self.robot.theta))
            grid, risk = self.mapper.get_grid_copy()
            
            start_cell = world_to_grid(self.robot.x, self.robot.y, self.mapper.world_origin, self.mapper.res)
            goal_cell = world_to_grid(self.goal_world[0], self.goal_world[1], self.mapper.world_origin, self.mapper.res)
            start_cell = (clamp(start_cell[0],0,grid.shape[0]-1), clamp(start_cell[1],0,grid.shape[1]-1))
            goal_cell = (clamp(goal_cell[0],0,grid.shape[0]-1), clamp(goal_cell[1],0,grid.shape[1]-1))
            path_cells = self.astar.astar(grid, risk, start_cell, goal_cell)
            v_cmd, w_cmd = self.local.plan((self.robot.x,self.robot.y,self.robot.theta), grid, path_cells or [], self.mapper.world_origin, self.mapper.res)
            obstacle_preds = self.tracker.get_predictions(horizon=0.8)
            v_safe, w_safe = cbf_filter((self.robot.x,self.robot.y,self.robot.theta), v_cmd, w_cmd, obstacle_preds, safe_dist=SAFE_DIST)
            if v_safe < v_cmd - 1e-3:
                log_event("cbf_slow", f"{v_cmd:.2f}->{v_safe:.2f}")
                if v_cmd - v_safe > 0.2: threading.Thread(target=speak, args=("Slowing down for safety",), daemon=True).start()
            out_v, out_w = self.pid.step(v_safe, w_safe, self.robot.v, self.robot.w, dt)
            
            vis = frame.copy()
            for (x1,y1,x2,y2,conf,cl) in detections:
                label = self.detector.model.names[cl] if hasattr(self.detector.model,'names') else str(cl)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, f"{label} {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
           
            for t in self.tracker.tracks:
                px,py = t.pos()
                gx,gy = world_to_grid(px,py,self.mapper.world_origin,self.mapper.res)
                imgx = int((gy / grid.shape[1]) * CAM_W)
                imgy = int((gx / grid.shape[0]) * CAM_H)
                cv2.circle(vis, (imgx,imgy), 4, (0,165,255), -1)
                cv2.putText(vis, f"ID{t.id}", (imgx+5,imgy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255),1)
            
            grid_rgb = cv2.cvtColor((1-grid).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
            risk_norm = (risk / (np.max(risk)+1e-6)); risk_img = (np.clip(risk_norm,0,1)*255).astype(np.uint8)
            risk_rgb = cv2.applyColorMap(risk_img, cv2.COLORMAP_JET); risk_rgb = cv2.resize(risk_rgb, (grid_rgb.shape[1], grid_rgb.shape[0]))
            blended = cv2.addWeighted(grid_rgb, 0.6, risk_rgb, 0.4, 0)
            if path_cells:
                for (gx,gy) in path_cells:
                    if 0<=gx<grid.shape[0] and 0<=gy<grid.shape[1]:
                        blended[gx,gy] = (0,0,255)
            rxg, ryg = world_to_grid(self.robot.x, self.robot.y, self.mapper.world_origin, self.mapper.res)
            if 0<=rxg<grid.shape[0] and 0<=ryg<grid.shape[1]:
                cv2.circle(blended, (ryg, rxg), 3, (0,255,0), -1)
            grid_disp = cv2.resize(blended, (CAM_W, CAM_H))
            combo = np.hstack([vis, grid_disp])
            cv2.putText(combo, f"Pose: ({self.robot.x:.2f},{self.robot.y:.2f},{self.robot.theta:.2f})", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)
            cv2.putText(combo, f"Cmd v/w: {v_cmd:.2f}/{w_cmd:.2f} | Safe: {v_safe:.2f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)
            
            with self.lock: self.latest_frame = combo.copy()
            cv2.imshow("Perception | Grid (Pro)", combo)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.shutdown(); break
            elif key == ord('g'): self.set_goal(self.robot.x + 2.0, self.robot.y)
            elif key == ord('r'): self.mapper.reset(); log_event("map_reset","user pressed r")
            elif key == ord('m'):
                
                grid_img = (self.mapper.grid*255).astype(np.uint8); risk_norm = (self.mapper.logodds/(np.max(self.mapper.logodds)+1e-6)*255).astype(np.uint8)
                vis_save = cv2.resize(np.hstack([cv2.cvtColor(grid_img,cv2.COLOR_GRAY2BGR), cv2.applyColorMap(risk_norm, cv2.COLORMAP_JET)]), (800,400))
                cv2.imwrite("map_snapshot.png", vis_save); log_event("map_saved","map_snapshot.png"); print("[Map] saved map_snapshot.png")
            
            if cv2.getWindowProperty("Perception | Grid (Pro)", cv2.WND_PROP_VISIBLE) < 1:
                print("[System] window closed by user"); self.shutdown(); break
            
            elapsed = time.time() - start
            sleep = max(0.0, (1.0/FPS_TARGET) - elapsed)
            time.sleep(sleep)
        
        try:
            self.detector.stop(); self.sensor.stop(); cv2.destroyAllWindows()
        except:
            pass
        print("[System] exited")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['AUTO','REAL','SIM'], default='AUTO')
    parser.add_argument('--model', default='yolov8n.pt')
    args = parser.parse_args()
    system = System(mode=args.mode, model=args.model)
    try:
        system.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
        system.shutdown()
