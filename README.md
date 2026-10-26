# Crop Pest Detection Using YOLOv8
## Project Overview
A deep learning project that detects 12 types of crop pests using YOLOv8. A Streamlit web application lets users upload images and view detection results.
## Technologies
- Python
- YOLOv8
- Ultralytics
- PyTorch
- Streamlit
- OpenCV
## Pest Classes
Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, and Weevils.
## Model Evaluation Results
- Precision: 59.36%
- Recall: 55.61%
- mAP@50: 58.28%
- mAP@50-95: 29.52%
## How to Run
1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   `python -m pip install -r requirements.txt`
4. Configure the dataset and trained model weights locally.
5. Start the application:
   `streamlit run app.py`
## Future Improvements
- Improve detection accuracy.
- Expand the dataset.
- Optimize inference speed.
- Test on additional real-world crop images.
