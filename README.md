# 🩺 Skin Lesion Classifier

An AI model that looks at a photo of a skin lesion and predicts which of **9 common
skin conditions** it most likely shows — built with YOLOv8, trained on real
dermatology images, and wrapped in a simple web app anyone can use.

> ⚠️ **This is an educational project, not a medical device.** It is not approved
> for diagnosis and must never replace seeing a real doctor. See
> [Limitations](#-known-limitations) below for why.

---

## 📸 What it does

Upload a photo of a skin lesion → the app shows its top predictions, ranked by
confidence, with a plain-English description of what each condition is.

| Input                    | Output                                             |
| ------------------------ | -------------------------------------------------- |
| A photo of a skin lesion | Top 5 predicted conditions + confidence % for each |

---

## 🎯 Results

The model was tested on 176 images it had never seen during training, and got the
**overall diagnosis right 87.5% of the time.**

| Condition                             | How often it's identified correctly (Recall) |
| ------------------------------------- | -------------------------------------------- |
| Vascular lesion                       | 100%                                         |
| Melanocytic nevus (common mole)       | 100%                                         |
| Atopic Dermatitis (eczema)            | 100%                                         |
| Benign keratosis                      | 95%                                          |
| Tinea Ringworm (fungal infection)     | 90%                                          |
| Dermatofibroma                        | 85%                                          |
| Melanoma (skin cancer)                | 70%                                          |
| Squamous cell carcinoma (skin cancer) | 75%                                          |
| Actinic keratosis (pre-cancerous)     | 75%                                          |

**In plain terms:** the model is very reliable on 6 of the 9 conditions, and
reasonably good — but not perfect — on the three that are hardest even for trained
dermatologists to tell apart from photos alone (melanoma vs. moles, and the two
pre-cancerous/cancerous conditions that sit on the same disease spectrum).

---

## 🧠 How it was built

1. **Started with a small dataset** (~80-100 images per condition) covering 9 skin
   lesion types.
2. **Balanced the dataset** so no single condition dominated training, using image
   augmentation (flips, rotation, brightness changes) to create realistic variety
   instead of just duplicating the same photos.
3. **Trained a YOLOv8 image classification model**, starting from a version
   pre-trained on millions of general images, then specialized it for skin lesions.
4. **Measured accuracy per condition**, not just overall — because a model can look
   great on paper while quietly failing on the conditions that matter most (like
   skin cancer).
5. **Found the two hardest conditions kept getting confused for each other** no
   matter how the model was tuned — so the fix wasn't a smarter model, it was
   **more real training photos** for exactly those conditions, pulled from the
   [ISIC Archive](https://www.isic-archive.com/), a public dermatology image
   database. That single change produced the biggest accuracy jump in the whole
   project (83% → 87.5%).
6. **Wrapped the final model in a Streamlit web app** so it's usable without
   needing to write any code.

---

## 🖥️ Try it yourself

### 1. Install the requirements

```bash
pip install -r requirements.txt
```

### 2. Make sure the model file is in place

Put `best.pt` inside a folder named `models/` in the project root:

```
skin-lesion-classifier/
└── models/
    └── best.pt
```

### 3. Run the app

```bash
streamlit run streamlit_app.py
```

This opens the app in your browser at `http://localhost:8501`. Upload any skin
lesion photo and see the predictions.

### Command-line option (no browser needed)

```bash
python inference/predict.py --image path/to/photo.jpg --model models/best.pt
```

---

## 📁 Project structure

```
skin-lesion-classifier/
├── README.md                  # This file
├── requirements.txt           # Python packages needed
├── .gitignore                 # Files git should ignore (model weights, cache, etc.)
├── streamlit_app.py           # The web app
├── .streamlit/
│   └── config.toml            # App color theme
├── notebooks/
│   └── train_yolov8_skin_classifier.ipynb   # Full training process, step by step
├── models/
│   └── best.pt                 # The trained model (not stored in git — see note below)
├── inference/
│   └── predict.py              # Command-line prediction script
└── results/
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── training_curves.png
```

**Note on the model file:** `best.pt` is ~32MB, too large for a typical git repo,
so it's excluded via `.gitignore`. If you're sharing this project, host the model
separately (a GitHub Release, Hugging Face, or Google Drive link) and download it
before running the app.

---

## ⚠️ Known Limitations

**1. It only works well on close-up, dermatoscope-style images.**
The model was trained entirely on images from the ISIC Archive — a medical
database of photos taken with a special magnifying device (a dermatoscope) held
directly against the skin, under controlled lighting. A normal photo taken with a
phone camera from a distance looks like a *completely different kind of image* to
the model, even though it's still a picture of skin to a human. This is a known,
expected limitation of every model trained this way — not a bug. If you test the
app with a random photo from Google Images, don't be surprised if it gets it wrong;
if you test it with an image from the actual dataset, it will typically be very
confident and correct.

**2. Some skin conditions are hard even for the model to separate — because they're
hard for humans too.**
Melanoma (skin cancer) and melanocytic nevus (a normal mole) can look very similar
in early stages — this is a genuinely difficult distinction in real dermatology,
not just a shortcoming of this model. The same is true for actinic keratosis and
squamous cell carcinoma, which sit on the same disease progression.

**3. Small dataset.**
Even after adding more real images, the model was trained on roughly 1,300 images
total across 9 classes — small by deep learning standards. A production-grade
medical tool would need thousands of expert-verified images per condition, ideally
reviewed by dermatologists, plus formal clinical validation before ever being used
to inform real decisions.

---

## 🩹 Disclaimer

This project was built for learning and portfolio purposes. It is **not a
certified medical device**, has **not undergone clinical validation**, and should
**never be used to diagnose yourself or anyone else**. If you're concerned about a
skin lesion, please see a licensed dermatologist or doctor.

---

## 🙏 Credits

- Base dataset: Split_smol skin lesion dataset
- Additional training images: [ISIC Archive](https://www.isic-archive.com/)
- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (classification mode)
- Dataset management: [Roboflow](https://roboflow.com/)
- Web app: [Streamlit](https://streamlit.io/)
