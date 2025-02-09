import sys
import os
import time
import random
import statistics
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QSlider,
    QComboBox, QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QCheckBox
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCharts import QChart, QChartView, QPieSeries

# ---- Matplotlib: gunakan dark background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setting default font ke Poppins, ukuran default 12 (nantinya bisa di-override)
matplotlib.rcParams['font.family'] = 'Poppins'
matplotlib.rcParams['font.size'] = 12
plt.style.use("dark_background")  # tema gelap

try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Placeholder ultralytics
try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    YOLO = None
    RTDETR = None


def get_color(label: str):
    rng = np.random.RandomState(abs(hash(label)) % 256)
    color = rng.randint(0, 256, 3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


def draw_rounded_rectangle(img, pt1, pt2, color, thickness=1, r=10):
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


def draw_crosshair(frame: np.ndarray, center, color=(0, 255, 0), size=20, thickness=2):
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


# =================== Thread Pemrosesan ===================== #
class VideoProcessingThread(QThread):
    logSignal = Signal(str)
    progressSignal = Signal(int)
    frameSignal = Signal(np.ndarray)
    finishedSignal = Signal(dict)

    def __init__(self, model_choice, video_path, crosshair_size=20):
        super().__init__()
        self.model_choice = model_choice
        self.video_path = video_path
        self.crosshair_size = crosshair_size

        self.active_events = {}
        self.reaction_times = []
        self.dwell_times = []

        self.paused = False  # untuk Pause/Resume
        self.output_video_path = None  # Simpan hasil output

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

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
                    self.logSignal.emit(
                        f"Deteksi {key}: Reaction time = {reaction_time_ms:.0f} ms. {note}"
                    )
                else:
                    dwell_time_ms = (current_time - self.active_events[key]['crosshair_enter_time']) * 1000
                    self.logSignal.emit(
                        f"Deteksi {key}: Dwell time = {dwell_time_ms:.0f} ms."
                    )
            else:
                if self.active_events[key]['crosshair_inside']:
                    dwell_time_ms = (current_time - self.active_events[key]['crosshair_enter_time']) * 1000
                    self.dwell_times.append(dwell_time_ms)
                    self.logSignal.emit(
                        f"Deteksi {key}: Final dwell time = {dwell_time_ms:.0f} ms."
                    )
                    self.active_events[key]['crosshair_inside'] = False

        # Hapus object yang tidak terdeteksi lagi > 0.5s
        keys_to_remove = []
        for key, event in self.active_events.items():
            if current_time - event['last_seen_time'] > 0.5:
                if event['crosshair_inside']:
                    dwell_time_ms = (current_time - event['crosshair_enter_time']) * 1000
                    self.dwell_times.append(dwell_time_ms)
                    self.logSignal.emit(
                        f"Deteksi {key}: Final dwell time = {dwell_time_ms:.0f} ms (detection lost)."
                    )
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.active_events[key]

    def run(self):
        self.logSignal.emit("Memulai proses video...")
        random.seed(10)

        # Pilih model
        if self.model_choice == "YOLO":
            chosen_model = "weights/yolo.pt"
            self.logSignal.emit("Model yang dipilih: YOLO")
            model = YOLO(chosen_model) if YOLO else None
        elif self.model_choice == "FastSAM":
            chosen_model = "weights/fastsam.pt"
            self.logSignal.emit("Model yang dipilih: FastSAM")
            model = YOLO(chosen_model) if YOLO else None
        else:
            chosen_model = "weights/rtdetr.pt"
            self.logSignal.emit("Model yang dipilih: RTDETR")
            model = RTDETR(chosen_model) if RTDETR else None

        self.logSignal.emit(f"Memuat model dari: {chosen_model}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.logSignal.emit("Error: Tidak dapat membuka video!")
            self.finishedSignal.emit({})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logSignal.emit(f"Total frame: {total_frames}")

        output_folder = "output_video"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_video_path = os.path.join(output_folder, f"output_video_{timestamp}.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        self.logSignal.emit(f"Menyimpan video output di: {self.output_video_path}")

        inference_times = []
        frame_count = 0

        while True:
            # -- Check Pause/Resume
            if self.paused:
                time.sleep(0.1)
                continue

            success, frame = cap.read()
            if not success:
                break

            start_time = time.time()
            if model is not None:
                results = model.track(frame, persist=True, conf=0.5)
            else:
                # Dummy object jika model tidak tersedia
                class DummyResult:
                    def __init__(self):
                        self.boxes = None
                        self.names = {}
                results = [DummyResult()]

            end_time = time.time()
            inference_times.append(end_time - start_time)

            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            draw_crosshair(frame, center, size=self.crosshair_size)

            # Kumpulkan box dengan confidence >= 50
            boxes = results[0].boxes if hasattr(results[0], 'boxes') else None
            detections = []
            if boxes is not None and boxes.data is not None:
                boxes_np = boxes.data.cpu().numpy()
                for boxdata in boxes_np:
                    if len(boxdata) == 7:
                        x1, y1, x2, y2, conf, cls, track_id = boxdata
                        key = int(track_id) if track_id is not None else (
                            int(x1)//10, int(y1)//10, int(x2)//10, int(y2)//10
                        )
                    else:
                        x1, y1, x2, y2, conf, cls = boxdata[:6]
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

            current_time = time.time()
            self.analyze_crosshair_placement(detections, center, current_time)

            annotated_frame = draw_boxes(frame.copy(), results) if model else frame
            output_writer.write(annotated_frame)

            self.frameSignal.emit(annotated_frame)

            frame_count += 1
            progress = int(frame_count / total_frames * 100)
            self.progressSignal.emit(progress)
            if frame_count % 10 == 0:
                self.logSignal.emit(f"Frame {frame_count}/{total_frames} telah diproses.")

        cap.release()
        output_writer.release()

        avg_inference_time = np.mean(inference_times) if inference_times else 0
        self.logSignal.emit(f"Waktu inferensi rata-rata: {avg_inference_time:.4f} detik")
        self.logSignal.emit("Proses video selesai.")

        stats = {
            "reaction_times": self.reaction_times,
            "dwell_times": self.dwell_times,
            "output_video_path": self.output_video_path
        }
        self.finishedSignal.emit(stats)


# =================== Main Window =========================== #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fixed Size + Donut Score Chart")
        self.setFixedSize(1300, 850)

        # Set default font di aplikasi agar pakai Poppins
        self.setFont(QFont("Poppins", 10))

        self.video_path = None
        self.worker_thread = None
        self.is_paused = False  # state Pause/Resume

        bg_img = "background.jpg"
        if os.path.exists(bg_img):
            # Mempertebal progress bar, dll.
            self.setStyleSheet(f"""
                QMainWindow {{
                    background-image: url({bg_img});
                    background-repeat: no-repeat;
                    background-position: center;
                    font-family: 'Poppins';
                }}
                QWidget#centralwidget {{
                    background: transparent;
                    font-family: 'Poppins';
                }}
                QTabWidget::pane {{
                    background: transparent;
                    border: none;
                    font-family: 'Poppins';
                }}
                QTableWidget {{
                    background: rgba(0,0,0,200); 
                    color: white;
                    font-family: 'Poppins';
                }}
                QTextEdit {{
                    background: rgba(0,0,0,100);
                    color: white;
                    font-family: 'Poppins';
                }}
                QLabel {{
                    background: transparent;
                    color: white;
                    font-family: 'Poppins';
                }}
                QPushButton {{
                    background-color: rgba(50, 150, 250, 180);
                    color: white;
                    font-weight: bold;
                    border-radius: 8px;
                    padding: 8px;
                    font-family: 'Poppins';
                }}
                /* Progressbar lebih "keren dan tebal" */
                QProgressBar {{
                    background-color: rgba(0,0,0,80);
                    color: white;
                    border: 1px solid #aaa;
                    font-family: 'Poppins';
                    font-weight: bold;
                    font-size: 12pt;
                    height: 25px;
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background-color: rgba(50, 200, 50, 1);
                    border-radius: 6px;
                }}
                QSlider::groove:horizontal {{
                    height: 6px;
                    background: rgba(255,255,255,120);
                }}
                QSlider::handle:horizontal {{
                    background: rgba(80,80,255,180);
                    width: 16px;
                    border-radius: 8px;
                }}
                QComboBox {{
                    background: rgba(255,255,255,160);
                    font-family: 'Poppins';
                }}
            """)

        self.initUI()

    def initUI(self):
        self.main_widget = QWidget()
        self.main_widget.setObjectName("centralwidget")
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)

        # Kontrol di atas
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        control_layout.addWidget(QLabel("Pilih Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLO", "FastSAM", "RTDETR"])
        control_layout.addWidget(self.model_combo)

        self.video_label = QLabel("Belum ada video yang dipilih")
        control_layout.addWidget(self.video_label)

        pick_btn = QPushButton("Pilih Video")
        pick_btn.clicked.connect(self.select_video)
        control_layout.addWidget(pick_btn)

        control_layout.addWidget(QLabel("Ukuran Crosshair:"))
        self.crosshair_slider = QSlider(Qt.Horizontal)
        self.crosshair_slider.setRange(10, 50)
        self.crosshair_slider.setValue(20)
        self.crosshair_slider.valueChanged.connect(self.on_crosshair_size_changed)
        control_layout.addWidget(self.crosshair_slider)

        self.crosshair_size_label = QLabel(str(self.crosshair_slider.value()))
        control_layout.addWidget(self.crosshair_size_label)

        self.start_btn = QPushButton("Mulai Proses")
        self.start_btn.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_btn)

        # Tombol Pause/Resume
        self.pause_resume_btn = QPushButton("Pause")
        self.pause_resume_btn.clicked.connect(self.pause_resume_processing)
        self.pause_resume_btn.setEnabled(False)
        control_layout.addWidget(self.pause_resume_btn)

        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab Preview
        self.tab_preview = QWidget()
        self.tabs.addTab(self.tab_preview, "Preview")
        preview_layout = QVBoxLayout(self.tab_preview)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)

        # Tab Log
        self.tab_log = QWidget()
        self.tabs.addTab(self.tab_log, "Log")
        log_layout = QVBoxLayout(self.tab_log)
        self.log_widget = QTextEdit()
        log_layout.addWidget(self.log_widget)

        # Tab Stats
        self.tab_stats = QWidget()
        self.tabs.addTab(self.tab_stats, "Stats")
        stats_layout = QVBoxLayout(self.tab_stats)

        self.stats_text = QTextEdit()
        stats_layout.addWidget(self.stats_text)

        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["Reaction (ms)", "Dwell (ms)"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(self.stats_table)

        self.chart_layout = QHBoxLayout()
        stats_layout.addLayout(self.chart_layout)

        self.figure_canvas = None
        self.score_chart_view = None

        # Tab Player (Media Player) untuk hasil akhir
        self.tab_player = QWidget()
        self.tabs.addTab(self.tab_player, "Player")
        player_layout = QVBoxLayout(self.tab_player)

        self.video_widget = QVideoWidget()
        player_layout.addWidget(self.video_widget)

        # QMediaPlayer + Audio
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Tambahkan slider untuk "seek" video
        self.media_slider = QSlider(Qt.Horizontal)
        player_layout.addWidget(self.media_slider)

        # Kontrol Media Player
        player_control_layout = QHBoxLayout()
        player_layout.addLayout(player_control_layout)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.media_player.play)
        player_control_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.media_player.pause)
        player_control_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.media_player.stop)
        player_control_layout.addWidget(self.stop_button)

        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(False)
        player_control_layout.addWidget(self.loop_checkbox)

        # Signal-signal media player
        self.loop_checkbox.stateChanged.connect(self.on_loop_checkbox_changed)
        self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_slider.sliderMoved.connect(self.on_slider_moved)

        self.is_loop = False

    @Slot(int)
    def on_crosshair_size_changed(self, val):
        self.crosshair_size_label.setText(str(val))

    def select_video(self):
        file_dialog = QFileDialog.getOpenFileName(
            self, "Pilih File Video", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        path = file_dialog[0]
        if path:
            self.video_path = path
            self.video_label.setText(os.path.basename(path))
            self.add_log(f"Video yang dipilih: {path}")

    def start_processing(self):
        if not self.video_path:
            # Pesan box warna teks hitam
            msgBox = QMessageBox(self)
            msgBox.setStyleSheet("QLabel { color: black; }")
            msgBox.setWindowTitle("Peringatan")
            msgBox.setText("Harap pilih file video terlebih dahulu!")
            msgBox.exec()
            return

        model_choice = self.model_combo.currentText()
        crosshair_size = self.crosshair_slider.value()
        self.add_log(f"Memulai proses: Model={model_choice}, Crosshair={crosshair_size}")

        self.start_btn.setEnabled(False)
        self.pause_resume_btn.setEnabled(True)

        self.worker_thread = VideoProcessingThread(
            model_choice=model_choice,
            video_path=self.video_path,
            crosshair_size=crosshair_size
        )
        self.worker_thread.logSignal.connect(self.add_log)
        self.worker_thread.progressSignal.connect(self.on_progress)
        self.worker_thread.frameSignal.connect(self.on_new_frame)
        self.worker_thread.finishedSignal.connect(self.on_finished)

        self.worker_thread.start()

    def pause_resume_processing(self):
        if not self.worker_thread:
            return

        if not self.is_paused:
            self.worker_thread.pause()
            self.is_paused = True
            self.pause_resume_btn.setText("Resume")
            self.add_log("Preview di-pause.")
        else:
            self.worker_thread.resume()
            self.is_paused = False
            self.pause_resume_btn.setText("Pause")
            self.add_log("Preview dilanjutkan.")

    @Slot(str)
    def add_log(self, text):
        self.log_widget.append(text)

    @Slot(int)
    def on_progress(self, val):
        self.progress_bar.setValue(val)

    @Slot(np.ndarray)
    def on_new_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)

    @Slot(dict)
    def on_finished(self, stats):
        self.add_log("Proses video selesai.")
        self.start_btn.setEnabled(True)
        self.pause_resume_btn.setEnabled(False)

        self.show_stats(stats)

        # Atur media player ke output video
        output_path = stats.get("output_video_path", "")
        if os.path.exists(output_path):
            self.media_player.setSource(QUrl.fromLocalFile(output_path))
            self.add_log(f"Player disiapkan untuk file: {output_path}")
        else:
            self.add_log("Tidak ada file output yang ditemukan.")

    def calculate_score(self, avg_reaction, avg_dwell):
        score = 100
        if avg_reaction > 400:
            score -= 15
        elif avg_reaction > 300:
            score -= 10
        elif avg_reaction > 200:
            score -= 5

        if avg_dwell > 1000:
            score -= 15
        elif avg_dwell > 600:
            score -= 10
        elif avg_dwell > 300:
            score -= 5

        return max(1, min(100, score))

    def show_stats(self, stats):
        reaction_times = stats.get("reaction_times", [])
        dwell_times = stats.get("dwell_times", [])

        self.stats_text.clear()
        self.stats_table.setRowCount(0)

        if self.figure_canvas:
            self.figure_canvas.setParent(None)
            self.figure_canvas = None
        if self.score_chart_view:
            self.score_chart_view.setParent(None)
            self.score_chart_view = None

        avg_r = statistics.mean(reaction_times) if reaction_times else 0
        avg_d = statistics.mean(dwell_times) if dwell_times else 0
        score = self.calculate_score(avg_r, avg_d) if (reaction_times or dwell_times) else 0

        summary_html = f"""
        <html>
        <body style="font-family:'Poppins', sans-serif; font-size:14pt; color:#ffffff;">
            <h2 style="color:#00FFFF; margin: 0 0 10px 0;">[ Statistical Summary ]</h2>
            <h2 style="color:#FFD700; margin: 0 0 8px 0;">Reaction Times</h2>
            <p style="margin:4px 0 4px 0;">
            {"No Data" if not reaction_times else f"Avg: {avg_r:.0f}, Min: {min(reaction_times):.0f}, Max: {max(reaction_times):.0f}"}
            </p>

            <h2 style="color:#7FFFD4; margin: 0 0 8px 0;">Dwell Times</h2>
            <p style="margin:4px 0 4px 0;">
            {"No Data" if not dwell_times else f"Avg: {avg_d:.0f}, Min: {min(dwell_times):.0f}, Max: {max(dwell_times):.0f}"}
            </p>

            <h3 style="color:#32CD32; margin: 8px 0 0 0;">Final Score: {score:.0f}/100</h3>
        </body>
        </html>
        """
        self.stats_text.setHtml(summary_html)

        # Tampilkan data di table
        row_idx = self.stats_table.rowCount()
        self.stats_table.insertRow(row_idx)
        self.stats_table.setItem(row_idx, 0, QTableWidgetItem(f"{avg_r:.0f} (avg)"))
        self.stats_table.setItem(row_idx, 1, QTableWidgetItem(f"{avg_d:.0f} (avg)"))

        if reaction_times:
            row_idx = self.stats_table.rowCount()
            self.stats_table.insertRow(row_idx)
            self.stats_table.setItem(row_idx, 0, QTableWidgetItem(f"{min(reaction_times):.0f} (min)"))
            if dwell_times:
                self.stats_table.setItem(row_idx, 1, QTableWidgetItem(f"{min(dwell_times):.0f} (min)"))
            else:
                self.stats_table.setItem(row_idx, 1, QTableWidgetItem("-"))

            row_idx = self.stats_table.rowCount()
            self.stats_table.insertRow(row_idx)
            self.stats_table.setItem(row_idx, 0, QTableWidgetItem(f"{max(reaction_times):.0f} (max)"))
            if dwell_times:
                self.stats_table.setItem(row_idx, 1, QTableWidgetItem(f"{max(dwell_times):.0f} (max)"))
            else:
                self.stats_table.setItem(row_idx, 1, QTableWidgetItem("-"))

            row_idx = self.stats_table.rowCount()
            self.stats_table.insertRow(row_idx)
            std_r = statistics.pstdev(reaction_times) if len(reaction_times) > 1 else 0
            self.stats_table.setItem(row_idx, 0, QTableWidgetItem(f"{std_r:.0f} (std)"))
            if dwell_times:
                std_d = statistics.pstdev(dwell_times) if len(dwell_times) > 1 else 0
                self.stats_table.setItem(row_idx, 1, QTableWidgetItem(f"{std_d:.0f} (std)"))
            else:
                self.stats_table.setItem(row_idx, 1, QTableWidgetItem("-"))

        # Plot histogram Reaction & Dwell
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
        bars1 = None
        if reaction_times:
            bars1 = axes[0].hist(reaction_times, bins=10, color='blue', alpha=0.7)
            axes[0].set_title("Reaction Times", color='white', fontsize=12)
        else:
            axes[0].text(0.5, 0.5, "No Reaction Data", ha='center', va='center', color='white')

        bars2 = None
        if dwell_times:
            bars2 = axes[1].hist(dwell_times, bins=10, color='green', alpha=0.7)
            axes[1].set_title("Dwell Times", color='white', fontsize=12)
        else:
            axes[1].text(0.5, 0.5, "No Dwell Data", ha='center', va='center', color='white')

        for ax in axes:
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')

        fig.tight_layout()

        if MPLCURSORS_AVAILABLE:
            import mplcursors
            if bars1:
                cursor1 = mplcursors.cursor(axes[0], hover=True)
                @cursor1.connect("add")
                def on_add_reaction(sel):
                    bin_edges = bars1[1]
                    xdata = sel.target[0]
                    bin_index = np.searchsorted(bin_edges, xdata) - 1
                    if 0 <= bin_index < len(bars1[0]):
                        freq = bars1[0][bin_index]
                        left_edge = bin_edges[bin_index]
                        right_edge = bin_edges[bin_index + 1]
                        sel.annotation.set_text(
                            f"{int(freq)} items in {int(left_edge)} - {int(right_edge)} ms"
                        )
                        sel.annotation.get_bbox_patch().set_alpha(0.7)

            if bars2:
                cursor2 = mplcursors.cursor(axes[1], hover=True)
                @cursor2.connect("add")
                def on_add_dwell(sel):
                    bin_edges = bars2[1]
                    xdata = sel.target[0]
                    bin_index = np.searchsorted(bin_edges, xdata) - 1
                    if 0 <= bin_index < len(bars2[0]):
                        freq = bars2[0][bin_index]
                        left_edge = bin_edges[bin_index]
                        right_edge = bin_edges[bin_index + 1]
                        sel.annotation.set_text(
                            f"{int(freq)} items in {int(left_edge)} - {int(right_edge)} ms"
                        )
                        sel.annotation.get_bbox_patch().set_alpha(0.7)

        self.figure_canvas = FigureCanvas(fig)
        self.chart_layout.addWidget(self.figure_canvas)
        self.figure_canvas.draw()

        # Donut Chart Score (QtCharts)
        score_series = QPieSeries()
        score_series.setHoleSize(0.4)  # donut
        score_series.append("Score", score)
        score_series.append("Remaining", 100 - score)

        slice_score = score_series.slices()[0]
        slice_score.setLabelVisible(True)
        slice_score.setLabelColor(Qt.white)
        slice_score.setLabel(f"{score:.0f}")

        # Set label bold + lebih besar (size 14)
        label_font = QFont("Poppins", 14, QFont.Bold)
        slice_score.setLabelFont(label_font)
        slice_score.setBrush(QColor("#32CD32"))

        slice_remain = score_series.slices()[1]
        slice_remain.setBrush(QColor("gray"))
        slice_remain.setLabelVisible(False)

        score_chart = QChart()
        score_chart.addSeries(score_series)
        score_chart.setTitle(f"Final Score: {score:.0f}/100")

        # Perbesar dan tebalkan judul "Final Score"
        title_font = QFont("Poppins", 16, QFont.Bold)
        score_chart.setTitleFont(title_font)

        score_chart.legend().hide()
        score_chart.setBackgroundVisible(False)
        score_chart.setTitleBrush(QColor("white"))

        self.score_chart_view = QChartView(score_chart)
        self.score_chart_view.setRenderHint(QPainter.Antialiasing)
        self.score_chart_view.setStyleSheet("background: transparent;")
        self.score_chart_view.setMinimumSize(300, 300)

        self.chart_layout.addWidget(self.score_chart_view)

    # ======== Bagian Media Player Loop & Slider ========
    def on_loop_checkbox_changed(self, state):
        self.is_loop = (state == Qt.Checked)

    def on_media_status_changed(self, status):
        # Jika video sudah selesai dan checkbox "Loop" tercentang => ulangi
        if status == QMediaPlayer.MediaStatus.EndOfMedia and self.is_loop:
            self.media_player.setPosition(0)
            self.media_player.play()

    @Slot(int)
    def on_duration_changed(self, duration):
        self.media_slider.setRange(0, duration)

    @Slot(int)
    def on_position_changed(self, position):
        # Kalau slider tidak sedang di-drag user, sinkronkan slider
        if not self.media_slider.isSliderDown():
            self.media_slider.setValue(position)

    @Slot(int)
    def on_slider_moved(self, position):
        self.media_player.setPosition(position)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Poppins", 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
