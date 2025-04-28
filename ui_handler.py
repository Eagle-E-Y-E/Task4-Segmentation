import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
import cv2
from utils import load_pixmap_to_label, display_image_Graphics_scene, enforce_slider_step, show_histogram_on_label
from Kmeans import kmeans_segment_image
from Agglomerative import apply_agglomerative_clustering
from regionGrowing import segment_image
import io


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
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

        self.mode_combo.currentTextChanged.connect(self.handle_mode_change)
        self.handle_mode_change()
        

        ## region growing_______________________________
        # sliders for region growing
        self.region_growing_threshold_slider.valueChanged.connect(
            lambda: self.region_growing_threshold_label.setText(f"{self.region_growing_threshold_slider.value()}"))
        #prominence_spinbox
        #distance_spinbox

        ##  how_histogram_on_label(self.histogram_label, self.input_image)

        # connect buttons
        self.apply_btn.clicked.connect(self.handle_apply)
        

    def handle_mode_change(self):
        self.mode = self.mode_combo.currentText()
        if self.mode == "agglomerative":
            self.agglomerative_widget.show()
            self.kmeans_widget.hide()
            self.region_growing_widget.hide()
            self.region_growing_hist_widget.hide()
        elif self.mode == "K-means":
            self.agglomerative_widget.hide()
            self.kmeans_widget.show()
            self.region_growing_widget.hide()
            self.region_growing_hist_widget.hide()
        elif self.mode == "region growing":
            self.agglomerative_widget.hide()
            self.kmeans_widget.hide()
            self.region_growing_widget.show()
            self.region_growing_hist_widget.show()


    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)
        if widget == self.input_img1:
            self.input_image = cv2.imread(self.img_path)
            s
    
    def handle_apply(self):
        if self.mode == "K-means":
            self.apply_kmeans()
        elif self.mode == "agglomerative":
            self.apply_agglomerative()
        elif self.mode == "region growing":
            self.apply_region_growing()
        elif self.mode == "mean shift":
            """"""
    def apply_kmeans(self):
        if self.input_image is not None:
            num_clusters = self.num_clusters_slider.value()
            max_iterations = self.max_iterations_slider.value()
            segmented_image = kmeans_segment_image(self.input_image, k=num_clusters, max_iters=max_iterations)
            display_image_Graphics_scene(self.output_img1_GV, segmented_image)

    def apply_agglomerative(self):
        if self.input_image is not None:
            initial_num_clusters = self.init_num_clusters_slider.value()
            num_clusters = self.agg_num_clusters_slider.value()
            segmented_image = apply_agglomerative_clustering(self.input_image, number_of_clusters=num_clusters, initial_number_of_clusters=initial_num_clusters)
            display_image_Graphics_scene(self.output_img1_GV, segmented_image)
    
    def apply_region_growing(self):
        if self.input_image is not None:
            tol = self.region_growing_threshold_slider.value()
            prominence = self.prominence_spinbox.value()
            distance = self.distance_spinbox.value()
            segmented_image = segment_image(self.input_image, prominence=prominence, distance=distance, tol=tol, mode='color')
            display_image_Graphics_scene(self.output_img1_GV, segmented_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
