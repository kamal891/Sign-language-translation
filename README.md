# 🤟 Sign Language Alphabet Translator (A–Z) – Real-Time

A simple real-time Sign Language Alphabet Recognition system using Python, OpenCV, and MediaPipe.  
It captures hand landmarks via webcam and predicts **A–Z letters only** using a trained machine learning model (Random Forest).

## 💡 Features

- 🖐️ Hand tracking using MediaPipe
- 📷 Real-time webcam support
- 📊 Train your own dataset (A–Z signs)
- 🧠 ML model trained on landmarks
- 💾 Data saved as CSV, model saved as .pkl


## 🧑‍💻 Tech Stack

- Python 3.10+
- OpenCV
- MediaPipe
- Scikit-learn
- Pandas
- NumPy


## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/SignLanguageTranslator.git
cd SignLanguageTranslator


SignLanguageTranslator/
├── collect_data.py           # Collect sign data (A–Z)
├── train_model.py            # Train Random Forest model
├── predict.py                # Real-time sign prediction
├── webcam_test.py            # Test webcam and hand tracking
├── hand_landmark_test.py     # Display hand landmarks
├── data/
│   └── sign_data.csv         # Dataset generated from webcam
├── sign_model.pkl            # Trained model (output)
├── requirements.txt          # Python dependencies
└── README.md                 # This file

