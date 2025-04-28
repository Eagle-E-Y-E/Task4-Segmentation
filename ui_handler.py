import sys
from PyQt5 import uic
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
import cv2
from utils import load_pixmap_to_label, display_image_Graphics_scene, enforce_slider_step, show_histogram_on_label
from Kmeans import kmeans_segment_image
from Agglomerative import apply_agglomerative_clustering
from regionGrowing import segment_image, smooth_histogram
from scipy.signal import find_peaks
from MeanShift import mean_shift_filter
from Thresholding import Thresholder


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.thresholder = Thresholder()
        uic.loadUi('ui.ui', self)
        self.input_image = None

        self.input_img1.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img1)

        # sliders for K-means
        self.num_clusters_slider.valueChanged.connect(
            lambda: self.num_clusters_label.setText(f"{self.num_clusters_slider.value()}"))
        self.max_iterations_slider.valueChanged.connect(
            lambda: self.max_iterations_label.setText(f"{self.max_iterations_slider.value()}"))
        # sliders for Agglomerative
        self.init_num_clusters_slider.valueChanged.connect(
            lambda: self.init_num_clusters_label.setText(f"{self.init_num_clusters_slider.value()}"))
        self.agg_num_clusters_slider.valueChanged.connect(
            lambda: self.agg_num_clusters_label.setText(f"{self.agg_num_clusters_slider.value()}"))
        # sliders for mean shift
        self.color_radius_slider.valueChanged.connect(
            lambda: self.color_radius_label.setText(f"{self.color_radius_slider.value()}"))
        self.spatial_radius_slider.valueChanged.connect(
            lambda: self.spatial_radius_label.setText(f"{self.spatial_radius_slider.value()}"))
        

        self.mode_combo.currentTextChanged.connect(self.handle_mode_change)
        self.handle_mode_change()

        # region growing_______________________________
        # sliders for region growing
        self.region_growing_threshold_slider.valueChanged.connect(
            lambda: self.region_growing_threshold_label.setText(f"{self.region_growing_threshold_slider.value()}"))
        # prominence_spinbox
        # distance_spinbox

        # how_histogram_on_label(self.histogram_label, self.input_image)

        # connect buttons
        self.apply_btn.clicked.connect(self.handle_apply)

        # thresholding tab____________________________________________________________
        self.thresholding_image = None
        self.input_img_thresholding.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img_thresholding)
        self.thersholding_combo.currentTextChanged.connect(
            self.handle_thresholding_mode_change)
        self.handle_thresholding_mode_change()
        # radio btns
        # self.global_radio
        # self.local_radio

        # sliders for spectral
        self.num_modes_slider.valueChanged.connect(
            lambda: self.num_modes_label.setText(f"{self.num_modes_slider.value()}"))
        self.smoothing_window_slider.valueChanged.connect(
            lambda: self.smoothing_window_label.setText(f"{self.smoothing_window_slider.value()}"))

        # sliders for local
        self.block_size_slider.valueChanged.connect(
            lambda: self.block_size_label.setText(f"{self.block_size_slider.value()}"))
        self.c_slider.valueChanged.connect(
            lambda: self.c_label.setText(f"{self.c_slider.value()}"))
        self.apply_btn_thresholding.clicked.connect(self.update_thresholding_image)
        self.num_modes_slider.valueChanged.connect(self.update_thresholding_image)
        self.smoothing_window_slider.valueChanged.connect(self.update_thresholding_image)
        self.block_size_slider.valueChanged.connect(self.update_thresholding_image)
        self.c_slider.valueChanged.connect(self.update_thresholding_image)


    def handle_mode_change(self):
        self.mode = self.mode_combo.currentText()
        if self.mode == "agglomerative":
            self.agglomerative_widget.show()
            self.kmeans_widget.hide()
            self.region_growing_widget.hide()
            self.region_growing_hist_widget.hide()
            self.mean_shift_widget.hide()
        elif self.mode == "K-means":
            self.agglomerative_widget.hide()
            self.kmeans_widget.show()
            self.region_growing_widget.hide()
            self.region_growing_hist_widget.hide()
            self.mean_shift_widget.hide()
        elif self.mode == "region growing":
            self.agglomerative_widget.hide()
            self.kmeans_widget.hide()
            self.region_growing_widget.show()
            self.region_growing_hist_widget.show()
            self.mean_shift_widget.hide()
        elif self.mode == "mean shift":
            self.agglomerative_widget.hide()
            self.kmeans_widget.hide()
            self.region_growing_widget.hide()
            self.region_growing_hist_widget.hide()
            self.mean_shift_widget.show()

    def handle_thresholding_mode_change(self):
        self.thresholding_mode = self.thersholding_combo.currentText()
        if self.thresholding_mode == "Spectral":
            self.spectral_widget.show()
            self.local_thresholding_widget.hide()
        elif self.thresholding_mode == "Local":
            self.spectral_widget.hide()
            self.local_thresholding_widget.show()
        else:
            self.spectral_widget.hide()
            self.local_thresholding_widget.hide()

    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)
        if widget == self.input_img1:
            self.input_image = cv2.imread(self.img_path)
        if widget == self.input_img_thresholding:
            self.thresholding_image = cv2.imread(self.img_path)

    def handle_apply(self):
        if self.mode == "K-means":
            self.apply_kmeans()
        elif self.mode == "agglomerative":
            self.apply_agglomerative()
        elif self.mode == "region growing":
            self.apply_region_growing()
        elif self.mode == "mean shift":
            self.apply_mean_shift()

    def apply_kmeans(self):
        if self.input_image is not None:
            num_clusters = self.num_clusters_slider.value()
            max_iterations = self.max_iterations_slider.value()
            segmented_image = kmeans_segment_image(
                self.input_image, k=num_clusters, max_iters=max_iterations)
            display_image_Graphics_scene(self.output_img1_GV, segmented_image)

    def apply_agglomerative(self):
        if self.input_image is not None:
            initial_num_clusters = self.init_num_clusters_slider.value()
            num_clusters = self.agg_num_clusters_slider.value()
            segmented_image = apply_agglomerative_clustering(
                self.input_image, number_of_clusters=num_clusters, initial_number_of_clusters=initial_num_clusters)
            display_image_Graphics_scene(self.output_img1_GV, segmented_image)
    
    def apply_region_growing(self):
        if self.input_image is not None:
            tol = self.region_growing_threshold_slider.value()
            prominence = self.prominence_spinbox.value()
            distance = self.distance_spinbox.value()
            img = self.input_image.copy()
            process_as_color = True
            if len(img.shape) == 3 and img.shape[2] == 3:
                if np.allclose(img[..., 0], img[..., 1], atol=1) and np.allclose(img[..., 0], img[..., 2], atol=1):
                    print("Detected as Grayscale image (stored in 3 channels).")
                    img = img[..., 0]
                    process_as_color = False
                else:
                    print("Detected as Color image.")
            elif len(img.shape) == 2:
                print("Detected as Grayscale image.")
                process_as_color = False
            else:
                raise ValueError("Unsupported image shape.")
            
            if not process_as_color:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
                smoothed_hist = smooth_histogram(hist, sigma=2)
                global_max_peak = np.argmax(smoothed_hist)
                peaks, _ = find_peaks(smoothed_hist, prominence=prominence, distance=distance)
                peaks = np.append(peaks, global_max_peak)
                peaks = np.unique(peaks)
                print("Detected grayscale peaks (intensity):", peaks)
                show_histogram_on_label(self.histogram_label, smoothed_hist, peaks=peaks)

            else:
                # For color segmentation, perform analysis using the L-channel.
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                L_channel = img_lab[:, :, 0]
                hist = cv2.calcHist([L_channel], [0], None, [256], [0, 256]).flatten()
                smoothed_hist = smooth_histogram(hist, sigma=2)
                global_max_peak = np.argmax(smoothed_hist)
                peaks, _ = find_peaks(smoothed_hist, prominence=prominence, distance=distance)
                peaks = np.append(peaks, global_max_peak)
                peaks = np.unique(peaks)
                print("Detected peaks in L-channel:", peaks)
                show_histogram_on_label(self.histogram_label, smoothed_hist, peaks=peaks)

            segmented_image = segment_image(img=img, peaks=peaks, tol=tol, peak_tol=2, process_as_color=process_as_color)
            display_image_Graphics_scene(self.output_img1_GV, segmented_image)

    def apply_mean_shift(self):
        if self.input_image is None: return
        segmented_image = mean_shift_filter(self.input_image, spatial_radius=8, color_radius=16, max_level=1)
        display_image_Graphics_scene(self.output_img1_GV, segmented_image)

    def update_thresholding_image(self):
        if self.thresholding_image is not None:
            gray_image = cv2.cvtColor(self.thresholding_image, cv2.COLOR_BGR2GRAY)
            method = self.thersholding_combo.currentText()
            num_modes = self.num_modes_slider.value()
            smooth_window = self.smoothing_window_slider.value()
            block_size = self.block_size_slider.value()
            C = self.c_slider.value()

            output_image, thresholds = self.thresholder.update_output(gray_image, method, num_modes,
                                                                      smooth_window, block_size, C)
            display_image_Graphics_scene(self.output_img_thresholding_GV, output_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
