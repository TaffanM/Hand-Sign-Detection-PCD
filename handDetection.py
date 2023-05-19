import cv2
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.uic import loadUi
from keras.models import load_model


model_path = 'model/keras_model.h5'  # Path to your trained model file


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('signGUI.ui', self)
        self.setStyleSheet("background-color: #9FE2BF;")
        self.image = None
        self.model = load_model(model_path)  # Load the trained model
        self.class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # Replace with your ASL alphabet class labels
        # self.class_labels = self.load_class_labels('model/labels.txt')

        # load gambar
        self.load_button.clicked.connect(self.load)
        self.load_button.setStyleSheet("background-color: #FFFFFF;")
        # translate
        self.translate_button.setStyleSheet("background-color: #FFFFFF")
        self.translate_button.clicked.connect(self.translate)


    # def load_class_labels(self, label_path):
    #     class_labels = np.loadtxt(str)
    #     return list(class_labels)
    
    def load(self):
        file_dialog = QFileDialog()
        image_path_tuple, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png)")
        image_path = image_path_tuple if image_path_tuple else ""
        print(image_path)
        if image_path:
            self.image = cv2.imread(image_path)
            if self.image is not None:
                self.displayImage()
                self.label_5.setText("")
            else:
                print("Gagal untuk memuat gambar.")
        else:
            print("Tidak ada file yang dipilih.")

    # Grayscale
    def grayscale(self, hand_image):
        H, W = self.image.shape[:2]
        print(f"Image shape: {self.image.shape}")
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                     0.587 * self.image[i, j, 1] +
                                     0.114 * self.image[i, j, 2], 0, 255)
        return gray

    # Histogram Equalization
    def histogramEqualization(self, grayscale):
        equalized = cv2.equalizeHist(grayscale)
        return equalized

    # Fast Fourier Transform
    def fft(self, grayscale):
        f = np.fft.fft2(grayscale)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        return magnitude_spectrum

    # Image Binarization
    def binarization(self, grayscale):
        _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
        return binary

    # Edge Detection
    def edgeDetection(self, grayscale):
        edges = cv2.Canny(grayscale, 100, 200)
        return edges

    def preprocess(self, hand_image):
        # Grayscale
        gray = self.grayscale(hand_image)
        # cv2.imshow("Grayscale", gray)

        # Histogram Equalization
        equalized = self.histogramEqualization(gray)
        # cv2.imshow("Histogram Equalization", equalized)

        # Fast Fourier Transform
        magnitude_spectrum = self.fft(equalized)
        # cv2.imshow("FFT", magnitude_spectrum)

        # Image Binarization
        binary = self.binarization(magnitude_spectrum)
        # cv2.imshow("Binarization", binary)
        # Convert to uint8
        binary = binary.astype(np.uint8)

        # Edge Detection
        edges = self.edgeDetection(binary)
        # cv2.imshow("Edge Detection", edges)

        # Resize to (224, 224)
        resized_image = cv2.resize(edges, (224, 224))

        # Add third channel dimension
        input_image = np.expand_dims(resized_image, axis=2)
        input_image = np.repeat(input_image, 3, axis=2)

        return input_image

        
    

    def translate(self):
        if self.image is not None:
            
            preprocessed_image = self.preprocess(self.image)
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
            self.label_4.setText(f"Predicted ASL Alphabet: {predicted_class_label}")
        else:
            self.label_5.setText("Tidak ada gambar yang dimuat")

    def displayImage(self):
        if self.image is None:
            print("No image to display.")
            return

        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:  # row[0], col[1], channel[2]
            if self.image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Hand Sign Detection Program')
window.show()
sys.exit(app.exec_())
