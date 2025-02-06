import os
import sys
import cv2
import time
import random
import threading
import statistics
import numpy as np

from ultralytics import YOLO, RTDETR

try:
    # Pillow >= 9.1.0
    from PIL import Image, ImageTk, Resampling
    resample_method = Resampling.LANCZOS
except ImportError:
    # Pillow versi lama (sebelum 9.1.0) tidak kenal "Resampling"
    from PIL import Image, ImageTk
    resample_method = Image.LANCZOS  # Bisa juga Image.BICUBIC atau Image.ANTIALIAS pada versi lama

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import Meter

from tkinter import filedialog, BooleanVar, IntVar, Text, Scrollbar
import matplotlib
matplotlib.use("Agg")  # Gunakan backend "Agg" agar tidak muncul jendela tambahan
import matplotlib.pyplot as plt

# Coba import mplcursors untuk interaksi
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ======================= Fungsi Pendukung ======================= #
def get_color(label):
    rng = np.random.RandomState(abs(hash(label)) % 256)
    color = rng.randint(0, 256, 3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


def draw_rounded_rectangle(img, pt1, pt2, color, thickness, r=10):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    r = int(min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2))
    
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_crosshair(frame, center, color=(0, 255, 0), size=20, thickness=2):
    x, y = center
    cv2.circle(frame, (x, y), size, color, thickness)


def draw_boxes(frame, results):
    result = results[0]
    if hasattr(result, 'names'):
        names = result.names
    else:
        names = {i: str(i) for i in range(100)}
    
    boxes = result.boxes
    if boxes is None or boxes.data is None:
        return frame

    boxes_np = boxes.data.cpu().numpy()
    for box in boxes_np:
        if len(box) == 7:
            x1, y1, x2, y2, conf, cls, track_id = box
        else:
            x1, y1, x2, y2, conf, cls = box[:6]
            track_id = None

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        label = names[cls] if cls in names else str(cls)
        if track_id is not None:
            label = f"{label} ID:{int(track_id)}"

        if conf <= 1.0:
            conf_value = conf * 100
        else:
            conf_value = conf
        conf_value = min(max(conf_value, 0), 100)
        conf_text = f"{conf_value:.0f}%"

        color = get_color(label)
        draw_rounded_rectangle(frame, (x1, y1), (x2, y2), color, thickness=1, r=10)

        font = cv2.FONT_HERSHEY_SIMPLEX
        label_font_scale = 0.5
        conf_font_scale = 0.4
        label_thickness = 1
        conf_thickness = 1
        (label_w, label_h), _ = cv2.getTextSize(label, font, label_font_scale, label_thickness)
        if y1 - 10 > label_h:
            label_org = (x1, y1 - 10)
        else:
            label_org = (x1, y1 + label_h + 10)
        cv2.putText(frame, label, label_org, font, label_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
        conf_org = (x1, label_org[1] + label_h + 5)
        cv2.putText(frame, conf_text, conf_org, font, conf_font_scale, (255, 255, 255), conf_thickness, cv2.LINE_AA)
    return frame


# ======================= Worker Thread ======================= #
class VideoProcessingThread(threading.Thread):
    def __init__(self, model_choice, video_path, log_callback, progress_callback,
                 frame_callback, finish_callback, crosshair_size):
        super().__init__()
        self.model_choice = model_choice
        self.video_path = video_path
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback
        self.finish_callback = finish_callback
        self.crosshair_size = crosshair_size
        self.active_events = {}
        self.reaction_times = []
        self.dwell_times = []

    def generate_suggestion(self, reaction_time_ms):
        if reaction_time_ms < 200:
            return "Excellent reaction!"
        elif reaction_time_ms < 400:
            return "Good, but can be improved."
        elif reaction_time_ms < 600:
            return "Average reaction, practice more."
        else:
            return "Slow reaction, work on your aim."

    def analyze_crosshair_placement(self, detections, center, current_time):
        radius = 20
        for detection in detections:
            key = detection["key"]
            bbox = detection["bbox"]
            if key not in self.active_events:
                self.active_events[key] = {
                    'detection_time': current_time,
                    'crosshair_enter_time': None,
                    'crosshair_inside': False,
                    'last_seen_time': current_time
                }
            else:
                self.active_events[key]['last_seen_time'] = current_time

            x1, y1, x2, y2 = bbox
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = np.sqrt((center[0] - bbox_center[0])**2 + (center[1] - bbox_center[1])**2)

            if distance <= radius:
                if not self.active_events[key]['crosshair_inside']:
                    self.active_events[key]['crosshair_inside'] = True
                    self.active_events[key]['crosshair_enter_time'] = current_time
                    reaction_time_ms = (current_time - self.active_events[key]['detection_time']) * 1000
                    self.reaction_times.append(reaction_time_ms)
                    note = self.generate_suggestion(reaction_time_ms)
                    self.log_callback(
                        f"Deteksi {key}: Reaction time = {reaction_time_ms:.0f} ms. {note}"
                    )
                else:
                    dwell_time_ms = (current_time - self.active_events[key]['crosshair_enter_time']) * 1000
                    self.log_callback(
                        f"Deteksi {key}: Dwell time = {dwell_time_ms:.0f} ms."
                    )
            else:
                if self.active_events[key]['crosshair_inside']:
                    dwell_time_ms = (current_time - self.active_events[key]['crosshair_enter_time']) * 1000
                    self.dwell_times.append(dwell_time_ms)
                    self.log_callback(
                        f"Deteksi {key}: Final dwell time = {dwell_time_ms:.0f} ms."
                    )
                    self.active_events[key]['crosshair_inside'] = False

        # Hapus event yang tidak terupdate
        keys_to_remove = []
        for key, event in self.active_events.items():
            if current_time - event['last_seen_time'] > 0.5:
                if event['crosshair_inside']:
                    dwell_time_ms = (current_time - event['crosshair_enter_time']) * 1000
                    self.dwell_times.append(dwell_time_ms)
                    self.log_callback(
                        f"Deteksi {key}: Final dwell time = {dwell_time_ms:.0f} ms (detection lost)."
                    )
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.active_events[key]

    def run(self):
        self.log_callback("Memulai proses video...")
        random.seed(10)
        if self.model_choice == "YOLO":
            chosen_model = "weights/yolo.pt"
            self.log_callback("Model yang dipilih: YOLO")
            model = YOLO(chosen_model)
        elif self.model_choice == "FastSAM":
            chosen_model = "weights/fastsam.pt"
            self.log_callback("Model yang dipilih: FastSAM")
            model = YOLO(chosen_model)
        else:
            chosen_model = "weights/rtdetr.pt"
            self.log_callback("Model yang dipilih: RTDETR")
            model = RTDETR(chosen_model)

        self.log_callback(f"Memuat model dari: {chosen_model}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_callback("Error: Tidak dapat membuka video!")
            self.finish_callback()
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_folder = "output_video"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(output_folder, f"output_video_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        self.log_callback(f"Memproses video: {self.video_path}")
        self.log_callback(f"Total frame: {total_frames}")
        self.log_callback(f"Menyimpan video output di: {output_video_path}")

        inference_times = []
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            current_time = time.time()
            start_time = current_time

            # Inference dengan track
            results = model.track(frame, persist=True, conf=0.5)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            draw_crosshair(frame, center, size=self.crosshair_size)

            # Ekstrak boxes
            boxes = results[0].boxes
            detections = []
            if boxes is not None and boxes.data is not None:
                boxes_np = boxes.data.cpu().numpy()
                for box in boxes_np:
                    if len(box) == 7:
                        x1, y1, x2, y2, conf, cls, track_id = box
                        key = int(track_id) if track_id is not None else (int(x1)//10, int(y1)//10, int(x2)//10, int(y2)//10)
                    else:
                        x1, y1, x2, y2, conf, cls = box[:6]
                        key = (int(x1)//10, int(y1)//10, int(x2)//10, int(y2)//10)

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if conf <= 1.0:
                        conf_value = conf * 100
                    else:
                        conf_value = conf
                    conf_value = min(max(conf_value, 0), 100)
                    if conf_value >= 50:
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "conf": conf_value,
                            "key": key
                        })

            self.analyze_crosshair_placement(detections, center, current_time)
            annotated_frame = draw_boxes(frame.copy(), results)
            output_video.write(annotated_frame)
            self.frame_callback(annotated_frame)

            frame_count += 1
            progress = int(frame_count / total_frames * 100)
            self.progress_callback(progress)
            if frame_count % 10 == 0:
                self.log_callback(f"Frame {frame_count}/{total_frames} telah diproses.")

        cap.release()
        output_video.release()

        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        self.log_callback(f"Waktu inferensi rata-rata: {avg_inference_time:.4f} detik")
        self.log_callback("Proses video selesai.")

        stats = {
            "reaction_times": self.reaction_times,
            "dwell_times": self.dwell_times
        }
        self.finish_callback(stats)


# ======================= Main Application ======================= #
class MainApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Crosshair Placement Analyzer")

        # Set window size and prevent resizing
        self.geometry("1300x850")
        self.resizable(False, False)

        # Add background image, check for existence
        bg_img_path = "background.jpg"
        if os.path.exists(bg_img_path):
            bg_image = Image.open(bg_img_path)
            bg_image = bg_image.resize((1300, 850), resample_method)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.bg_label = ttk.Label(self, image=self.bg_photo)
            self.bg_label.grid(row=0, column=0, sticky="nsew", rowspan=3)  # This line is within the class now
            self.bg_label.lower()  # Make sure it's behind other widgets
        else:
            self.config(bg="#222222")  # If no background image, set a solid background color

        self.video_path = None
        self.worker_thread = None
        self.show_log = BooleanVar(value=True)
        self.progress_var = IntVar(value=0)
        self.crosshair_size = IntVar(value=20)

        self.latest_frame = None

        # Setup grid pada root
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # control panel
        self.rowconfigure(1, weight=1)  # content
        self.rowconfigure(2, weight=0)  # progress

        self.create_widgets()

    def create_widgets(self):
        # --- Control Panel (Top) ---
        control_frame = ttk.Frame(self, padding=10, bootstyle="dark")  # Frame lebih gelap
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        # Buat sub-grid agar lebih rapi
        for i in range(5):
            control_frame.columnconfigure(i, weight=1)

        # Baris 1: Pilih model + combobox + label info video + tombol
        row_idx = 0
        ttk.Label(control_frame, text="Pilih Model:", font=("Poppins", 10, "bold")).grid(
            row=row_idx, column=0, padx=5, pady=5, sticky="w"
        )
        self.model_combo = ttk.Combobox(
            control_frame,
            values=["YOLO", "FastSAM", "RTDETR"],
            state="readonly",
            width=15
        )
        self.model_combo.current(0)
        self.model_combo.grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")

        self.video_label = ttk.Label(
            control_frame,
            text="Belum ada video yang dipilih",
            width=30,
            anchor="center"
        )
        self.video_label.grid(row=row_idx, column=2, padx=5, pady=5, sticky="ew")

        select_video_btn = ttk.Button(
            control_frame, text="Pilih Video",
            command=self.select_video,
            bootstyle=PRIMARY
        )
        select_video_btn.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew")

        self.log_toggle_btn = ttk.Button(
            control_frame,
            text="Hide Log",
            command=self.toggle_log,
            bootstyle=INFO
        )
        self.log_toggle_btn.grid(row=row_idx, column=4, padx=5, pady=5, sticky="ew")

        # Baris 2: Slider crosshair size + label size + tombol Mulai
        row_idx += 1
        ttk.Label(control_frame, text="Ukuran Crosshair:", font=("Poppins", 10, "bold")).grid(
            row=row_idx, column=0, padx=5, pady=5, sticky="w"
        )
        self.crosshair_slider = ttk.Scale(
            control_frame,
            from_=10, to=50,
            orient="horizontal",
            variable=self.crosshair_size,
            bootstyle="info"
        )
        self.crosshair_slider.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")

        self.crosshair_size_label = ttk.Label(
            control_frame,
            text=f"Size: {self.crosshair_size.get()}",
            font=("Poppins", 10)
        )
        self.crosshair_size_label.grid(row=row_idx, column=2, padx=5, pady=5, sticky="w")

        self.start_btn = ttk.Button(
            control_frame,
            text="Mulai Proses",
            command=self.start_processing,
            bootstyle=SUCCESS
        )
        self.start_btn.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew", columnspan=2)

        # --- Main Content (Center) ---
        content_frame = ttk.Frame(self, padding=10, bootstyle="secondary")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.columnconfigure(0, weight=1, uniform="group1")
        content_frame.columnconfigure(1, weight=1, uniform="group1")
        content_frame.rowconfigure(0, weight=1)

        # Preview (Left)
        preview_frame = ttk.Frame(content_frame, bootstyle="dark")
        preview_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill="both", expand=True)

        # Notebook: Log & Stats (Right)
        self.notebook = ttk.Notebook(content_frame, bootstyle="info")
        self.notebook.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Tab Log
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Log")

        self.log_widget = Text(self.log_tab, wrap="word", height=20, font=("Poppins", 9))
        self.log_widget.pack(side="left", fill="both", expand=True)
        log_scroll = Scrollbar(self.log_tab, command=self.log_widget.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_widget.config(yscrollcommand=log_scroll.set)

        # Tab Stats
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Stats")
        self.stats_tab.columnconfigure(0, weight=1)
        self.stats_tab.rowconfigure(0, weight=3)
        self.stats_tab.rowconfigure(1, weight=1)

        # Frame Plot (Top in Stats Tab)
        self.stats_plot_frame = ttk.Frame(self.stats_tab)
        self.stats_plot_frame.grid(row=0, column=0, sticky="nsew")

        # Bottom (Scoreboard + Detail)
        bottom_stats_frame = ttk.Frame(self.stats_tab)
        bottom_stats_frame.grid(row=1, column=0, sticky="nsew", pady=10)

        scoreboard_frame = ttk.Frame(bottom_stats_frame)
        scoreboard_frame.pack(side="left", padx=20)

        self.score_meter = Meter(
            scoreboard_frame,
            bootstyle='success',
            subtext='Score',
            interactive=False,
            amountused=0,
            metertype='arc',
            stripethickness=4,
            metersize=150
        )
        self.score_meter.pack()

        self.score_label = ttk.Label(scoreboard_frame, text="Score: -", font=("Poppins", 10, "bold"))
        self.score_label.pack(pady=10)

        details_frame = ttk.Frame(bottom_stats_frame)
        details_frame.pack(side="left", fill="both", expand=True, padx=20)

        self.stats_display = Text(details_frame, wrap="word", height=8, font=("Poppins", 10))
        self.stats_display.pack(fill="both", expand=True, padx=10, pady=5)

        # Atur style warna dalam text
        self.stats_display.tag_config("good", foreground="green")
        self.stats_display.tag_config("medium", foreground="yellow")
        self.stats_display.tag_config("bad", foreground="red")
        self.stats_display.tag_config("title", foreground="cyan", font=("Poppins", 10, "bold"))
        
        self.stats_tree = ttk.Treeview(
            details_frame,
            columns=("reaction", "dwell"),
            show="headings",
            height=5
        )
        self.stats_tree.heading("reaction", text="Reaction (ms)")
        self.stats_tree.heading("dwell", text="Dwell (ms)")
        self.stats_tree.column("reaction", width=120, anchor="center")
        self.stats_tree.column("dwell", width=120, anchor="center")

        self.stats_tree.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Progress Bar (Bottom) ---
        progress_frame = ttk.Frame(self, padding=10, bootstyle="dark")
        progress_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            bootstyle="striped"
        )
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.progress_label = ttk.Label(progress_frame, text="0%", font=("Poppins", 10))
        self.progress_label.pack(side="left")

        # Binding
        self.preview_label.bind("<Configure>", self.on_resize_label)
        self.crosshair_slider.bind("<Motion>", self.update_crosshair_size_label)

    def update_crosshair_size_label(self, event):
        self.crosshair_size_label.config(text=f"Size: {self.crosshair_size.get()}")

    def on_resize_label(self, event):
        if self.latest_frame is not None:
            self.update_live_view(self.latest_frame)

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Pilih File Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv")]
        )
        if path:
            self.video_path = path
            self.video_label.config(text=os.path.basename(path))
            self.add_log(f"Video yang dipilih: {path}")

    def toggle_log(self):
        if self.show_log.get():
            self.notebook.forget(self.log_tab)
            self.log_toggle_btn.config(text="Show Log")
            self.show_log.set(False)
        else:
            self.notebook.add(self.log_tab, text="Log")
            self.log_toggle_btn.config(text="Hide Log")
            self.show_log.set(True)

    def start_processing(self):
        if not self.video_path:
            self.add_log("Harap pilih file video terlebih dahulu!")
            return
        model_choice = self.model_combo.get()
        self.add_log(f"Memulai proses dengan model: {model_choice}")
        self.start_btn.config(state="disabled")
        self.worker_thread = VideoProcessingThread(
            model_choice,
            self.video_path,
            log_callback=lambda msg: self.after(0, lambda: self.add_log(msg)),
            progress_callback=lambda val: self.after(0, lambda: self.update_progress(val)),
            frame_callback=lambda frame: self.after(0, lambda: self.update_live_view(frame)),
            finish_callback=lambda stats=None: self.after(0, lambda: self.processing_finished(stats)),
            crosshair_size=self.crosshair_size.get()
        )
        # Jalankan dalam thread terpisah
        threading.Thread(target=self.worker_thread.run, daemon=True).start()

    def update_progress(self, value):
        self.progress_var.set(value)
        self.progress_label.config(text=f"{value}%")

    def add_log(self, message):
        self.log_widget.insert("end", message + "\n")
        self.log_widget.see("end")

    def update_live_view(self, frame):
        if frame is None:
            return

        self.latest_frame = frame
        label_width = self.preview_label.winfo_width()
        label_height = self.preview_label.winfo_height()

        if label_width < 2 or label_height < 2:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        orig_w, orig_h = pil_img.size
        scale = min(label_width / orig_w, label_height / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        letterbox_img = Image.new("RGB", (label_width, label_height), (0, 0, 0))
        offset_x = (label_width - new_w) // 2
        offset_y = (label_height - new_h) // 2
        letterbox_img.paste(resized_img, (offset_x, offset_y))

        imgtk = ImageTk.PhotoImage(letterbox_img)
        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk)

    def processing_finished(self, stats=None):
        self.start_btn.config(state="normal")
        self.add_log("Proses selesai.")
        if stats:
            self.show_stats_plot(stats)
        else:
            self.stats_display.delete("1.0", "end")
            self.stats_display.insert("end", "Tidak ada event crosshair placement yang terekam.")

    def calculate_score(self, avg_reaction, avg_dwell):
        score = 100
        # Reaction time penalty
        if avg_reaction > 400:
            score -= 15
        elif avg_reaction > 300:
            score -= 10
        elif avg_reaction > 200:
            score -= 5

        # Dwell time penalty
        if avg_dwell > 1000:
            score -= 15
        elif avg_dwell > 600:
            score -= 10
        elif avg_dwell > 300:
            score -= 5

        return max(1, min(100, score))

    def show_stats_plot(self, stats):
        reaction_times = stats.get("reaction_times", [])
        dwell_times = stats.get("dwell_times", [])

        # Bersihkan frame plot
        for widget in self.stats_plot_frame.winfo_children():
            widget.destroy()

        if not reaction_times and not dwell_times:
            self.stats_display.delete("1.0", "end")
            self.stats_display.insert("end", "Tidak ada data untuk ditampilkan.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=100)

        # Plot Reaction Time
        bars1 = None
        if reaction_times:
            bars1 = axes[0].hist(reaction_times, bins=10, color='blue', alpha=0.7)
            axes[0].set_title("Reaction Times")
            axes[0].set_xlabel("Waktu (ms)")
            axes[0].set_ylabel("Frekuensi")
        else:
            axes[0].text(0.5, 0.5, "No Reaction Data", ha='center', va='center')

        # Plot Dwell Time
        bars2 = None
        if dwell_times:
            bars2 = axes[1].hist(dwell_times, bins=10, color='green', alpha=0.7)
            axes[1].set_title("Dwell Times")
            axes[1].set_xlabel("Waktu (ms)")
            axes[1].set_ylabel("Frekuensi")
        else:
            axes[1].text(0.5, 0.5, "No Dwell Data", ha='center', va='center')

        fig.tight_layout()

        # Interaktif dengan mplcursors (jika tersedia)
        if MPLCURSORS_AVAILABLE:
            # Reaction times hover
            if bars1 is not None:
                cursor1 = mplcursors.cursor(axes[0], hover=True)
                @cursor1.connect("add")
                def on_add_reaction(sel):
                    bin_edges = bars1[1]
                    xdata = sel.target[0]
                    bin_index = np.searchsorted(bin_edges, xdata) - 1
                    if bin_index < 0 or bin_index >= len(bars1[0]):
                        return
                    freq = bars1[0][bin_index]
                    left_edge = bin_edges[bin_index]
                    right_edge = bin_edges[bin_index + 1]
                    sel.annotation.set_text(f"{int(freq)} items in {int(left_edge)} - {int(right_edge)} ms")
                    sel.annotation.get_bbox_patch().set_alpha(0.7)

            # Dwell times hover
            if bars2 is not None:
                cursor2 = mplcursors.cursor(axes[1], hover=True)
                @cursor2.connect("add")
                def on_add_dwell(sel):
                    bin_edges = bars2[1]
                    xdata = sel.target[0]
                    bin_index = np.searchsorted(bin_edges, xdata) - 1
                    if bin_index < 0 or bin_index >= len(bars2[0]):
                        return
                    freq = bars2[0][bin_index]
                    left_edge = bin_edges[bin_index]
                    right_edge = bin_edges[bin_index + 1]
                    sel.annotation.set_text(f"{int(freq)} items in {int(left_edge)} - {int(right_edge)} ms")
                    sel.annotation.get_bbox_patch().set_alpha(0.7)

        canvas = FigureCanvasTkAgg(fig, master=self.stats_plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()

        # Update Summary (Text & Tree)
        self.update_stats_display(reaction_times, dwell_times)
        self.update_stats_tree(reaction_times, dwell_times)

    def update_stats_display(self, reaction_times, dwell_times):
        self.stats_display.delete("1.0", "end")
        self.stats_display.insert("end", "[ Statistical Summary ]\n", "title")

        # Reaction Times
        if reaction_times:
            avg_r = statistics.mean(reaction_times)
            mn_r = min(reaction_times)
            mx_r = max(reaction_times)
            sd_r = statistics.pstdev(reaction_times) if len(reaction_times) > 1 else 0

            if avg_r < 300:
                tag = "good"
                msg = "Reaction time is quite good!\n"
            elif avg_r < 600:
                tag = "medium"
                msg = "Reaction time is moderate.\n"
            else:
                tag = "bad"
                msg = "Reaction time is quite slow!\n"

            self.stats_display.insert("end", f"\nReaction Times:\n")
            self.stats_display.insert("end", msg, tag)
            self.stats_display.insert(
                "end",
                f" - Average : {avg_r:.0f} ms\n - Min     : {mn_r:.0f} ms\n - Max     : {mx_r:.0f} ms\n - StdDev  : {sd_r:.0f} ms\n",
            )
        else:
            self.stats_display.insert("end", "\nReaction Times:\nNo Data Available.\n", "medium")

        # Dwell Times
        if dwell_times:
            avg_d = statistics.mean(dwell_times)
            mn_d = min(dwell_times)
            mx_d = max(dwell_times)
            sd_d = statistics.pstdev(dwell_times) if len(dwell_times) > 1 else 0

            if avg_d < 300:
                tag = "good"
                msg = "Dwell time is very quick!\n"
            elif avg_d < 600:
                tag = "medium"
                msg = "Dwell time is moderate.\n"
            else:
                tag = "bad"
                msg = "Dwell time is quite long!\n"

            self.stats_display.insert("end", f"\nDwell Times:\n")
            self.stats_display.insert("end", msg, tag)
            self.stats_display.insert(
                "end",
                f" - Average : {avg_d:.0f} ms\n - Min     : {mn_d:.0f} ms\n - Max     : {mx_d:.0f} ms\n - StdDev  : {sd_d:.0f} ms\n",
            )
        else:
            self.stats_display.insert("end", "\nDwell Times:\nNo Data Available.\n", "medium")

        # Score
        final_score_text = self.build_score_text(reaction_times, dwell_times)
        self.stats_display.insert("end", f"\n{final_score_text}\n", "title")

    def build_score_text(self, reaction_times, dwell_times):
        if not reaction_times and not dwell_times:
            return "No scoring data."

        avg_r = statistics.mean(reaction_times) if reaction_times else 0
        avg_d = statistics.mean(dwell_times) if dwell_times else 0

        score = self.calculate_score(avg_r, avg_d)
        return f"Final Score: {score:.0f}/100"

    def update_stats_tree(self, reaction_times, dwell_times):
        self.stats_tree.delete(*self.stats_tree.get_children())

        if reaction_times:
            avg_r = statistics.mean(reaction_times)
            min_r = min(reaction_times)
            max_r = max(reaction_times)
            std_r = statistics.pstdev(reaction_times) if len(reaction_times) > 1 else 0
        else:
            avg_r = min_r = max_r = std_r = 0

        if dwell_times:
            avg_d = statistics.mean(dwell_times)
            min_d = min(dwell_times)
            max_d = max(dwell_times)
            std_d = statistics.pstdev(dwell_times) if len(dwell_times) > 1 else 0
        else:
            avg_d = min_d = max_d = std_d = 0

        # Tambahkan ringkasan ke Tree
        self.stats_tree.insert("", "end", values=(f"{avg_r:.0f} (avg)", f"{avg_d:.0f} (avg)"))
        self.stats_tree.insert("", "end", values=(f"{min_r:.0f} (min)", f"{min_d:.0f} (min)"))
        self.stats_tree.insert("", "end", values=(f"{max_r:.0f} (max)", f"{max_d:.0f} (max)"))
        self.stats_tree.insert("", "end", values=(f"{std_r:.0f} (std)", f"{std_d:.0f} (std)"))

        # Update scoreboard
        score = self.calculate_score(avg_r, avg_d)
        self.score_meter.configure(amountused=score)
        self.score_label.config(text=f"Score: {score:.0f}")


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
