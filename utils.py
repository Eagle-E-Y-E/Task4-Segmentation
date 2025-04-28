from PyQt5.QtWidgets import QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLabel, QVBoxLayout, \
    QWidget, QFileDialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer
import io


def load_pixmap_to_label(label: QLabel):
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                               options=options)

    if file_path:
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter)
    return file_path


def display_image_Graphics_scene(view, image):
    pixmap = convert_cv_to_pixmap(image)
    scene = QGraphicsScene()
    scene.addPixmap(pixmap)
    view.setScene(scene)
    view.fitInView(
        scene.itemsBoundingRect(), Qt.KeepAspectRatio)


def convert_cv_to_pixmap(cv_img):
    """Convert an OpenCV image to QPixmap"""
    if len(cv_img.shape) == 2:  # Grayscale image
        height, width = cv_img.shape
        bytesPerLine = width
        qImg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
    else:  # Color image
        height, width, channels = cv_img.shape
        bytesPerLine = channels * width
        qImg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

    return QPixmap.fromImage(qImg)

## usage

## # Display the edge image in filteroutput1
##    self.display_image_to_graphics_view(self.filteroutput1, edges)
##    edges should be the image returned from the canny function



def enforce_slider_step(slider, step, min_value):
    """
    Enforce a slider to snap to specific steps.

    Args:
        slider (QSlider): The slider to enforce steps on.
        step (int): The step size (e.g., 2 for increments of 2).
        min_value (int): The minimum value of the slider.
    """
    value = slider.value()
    if (value - min_value) % step != 0:
        corrected_value = round((value - min_value) / step) * step + min_value
        slider.setValue(corrected_value)

def show_histogram_on_label(label, data):
    # Create the histogram with matplotlib
    plt.figure(figsize=(4, 3))
    plt.hist(data.ravel(), bins=256, color='gray')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    # Convert to QPixmap
    qimg = QImage.fromData(buf.getvalue())
    pixmap = QPixmap.fromImage(qimg)
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)
