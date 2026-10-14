Handwritten Digit Recognition Using CNN

A web-based handwritten digit recognition application powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can draw digits from 0 to 9 directly in their browser and receive a prediction with confidence and probability scores.

🚀 Features

- Interactive drawing canvas for mouse and touch input
- Recognition of handwritten digits from 0 to 9
- CNN model built with TensorFlow and Keras
- Prediction confidence and class probability visualization
- Clear and redraw functionality
- Responsive frontend using HTML, CSS, and JavaScript
- Flask backend for connecting the frontend with the trained model

🛠️ Technologies Used

- **Language:** Python
- **Machine Learning:** TensorFlow, Keras, CNN
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Image Processing:** NumPy, Pillow
- **Dataset:** MNIST handwritten digits dataset

## 📊 Model Performance

The trained CNN was evaluated on the MNIST test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | 98.42% |
| Test Loss | 0.0537 |

These results describe performance on the MNIST test dataset. Accuracy on user-drawn images may vary because handwriting and image preprocessing can differ.

## 📁 Project Structure

MNIST_CNN/
├── app.py
├── mnist_cnn.keras
├── requirements.txt
├── README.md
├── .gitignore
└── templates/
    └── index.html

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/HemCharanKalavakuri/handwritten-digit-recognition-cnn.git
cd handwritten-digit-recognition-cnn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the website

Visit:

`http://127.0.0.1:5000`

Draw a digit on the canvas and click **Predict Digit** to view the result.

## 🧠 How It Works

1. The user draws a digit on the browser canvas.
2. JavaScript sends the drawing to the Flask backend.
3. The backend converts the image to grayscale, resizes it to 28 × 28 pixels, and normalizes its pixel values.
4. The trained CNN predicts the digit.
5. The frontend displays the predicted digit, confidence score, and class probabilities.

## 🎥 Demo Video

**Video:** [Watch the project demonstration](ADD_YOUR_VIDEO_URL_HERE)

Replace the placeholder above with your screen-recorded demo video URL.

## 🌐 Live Demo

**Application URL:** ADD_YOUR_RENDER_URL_HERE

Replace this placeholder after your Render deployment succeeds.

## 🔮 Future Improvements

- Improve digit centering and image preprocessing
- Enhance recognition across different handwriting styles
- Add prediction history
- Improve the user interface
- Deploy additional model versions for comparison

## 👨‍💻 Author

**Hem Charan Kalavakuri**

GitHub: [HemCharanKalavakuri](https://github.com/HemCharanKalavakuri)

## 📄 License

This project is intended for educational and learning purposes. Add a suitable open-source license if you intend to distribute it under specific reuse terms.
