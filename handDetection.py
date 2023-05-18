import cv2
import numpy as np
import konvolusi
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.uic import loadUi
from keras.models import load_model



dataset_path = 'asl_alphabet_test/'
model_path = 'model/keras_model.h5'  # Path to your trained model file


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('signGUI.ui', self)
        self.setStyleSheet("background-color: #9FE2BF;")
        self.image = None
        self.model = load_model(model_path)  # Load the trained model
        self.class_labels = ['A', 'B', 'C']  # Replace with your ASL alphabet class labels
        # load gambar
        self.load_button.clicked.connect(self.load)
        self.load_button.setStyleSheet("background-color: #FFFFFF;")
        # translate
        self.translate_button.setStyleSheet("background-color: #FFFFFF")
        self.translate_button.clicked.connect(self.translate)

    def load(self):
        file_dialog = QFileDialog()
        image_path_tuple, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png)")
        image_path = image_path_tuple if image_path_tuple else ""
        print(image_path)
        if image_path:
            self.image = cv2.imread(image_path)
            if self.image is not None:
                self.displayImage(1)
            else:
                print("Gagal untuk memuat gambar.")
        else:
            print("Tidak ada file yang dipilih.")

    # Grayscale
    def grayscale(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                     0.587 * self.image[i, j, 1] +
                                     0.114 * self.image[i, j, 2], 0, 255)
        return gray

    # Gaussian blur
    def gaussianBlur(self):
        kernelGaussian = (1.0 / 345) * np.array([
            [1, 5, 7, 5, 1],
            [5, 20, 33, 20, 5],
            [7, 33, 55, 33, 7],
            [5, 20, 33, 20, 5],
            [1, 5, 7, 5, 1]
        ])
        img = self.image
        hasil = konvolusi.konvolusi(img, kernelGaussian)
        return hasil

    def preprocess(self):
        gray = self.grayscale()

        blur = self.gaussianBlur()

        _, threshold = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

        return threshold

    def translate(self):
        if self.image is not None:
            preprocessed_image = self.preprocess()
            # Resize the preprocessed image to match the input size expected by the model
            resized_image = cv2.resize(preprocessed_image, (224, 224))
            # Normalize the image values between 0 and 1
            normalized_image = resized_image / 255.0
            # Add an extra dimension to match the model's input shape
            input_image = np.expand_dims(normalized_image, axis=0)
            # Predict the class label using the loaded model
            predictions = self.model.predict(input_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = self.class_labels[predicted_class_index]
            self.label.setText(f"Predicted ASL Alphabet: {predicted_class_label}")

            self.displayImage(2, preprocessed_image)
        else:
            print("No image loaded.")

    def displayImage(self, windows=1, image=None):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:  # row[0],col[1],channel[2]
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0],
                     self.image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Hand Sign Detection Program')
window.show()
sys.exit(app.exec_())
