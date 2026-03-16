# Vision Muscle Valuation & Autonomous Robotics ðŸ¤–

Research repository focused on Computer Vision applications in healthcare (muscle assessment) and defense (autonomous target detection).

## ðŸ”¬ Research Focus
1. **Real-time Muscle Assessment:** Using CNNs to value muscle health from image data.
2. **Autonomous Target Detection:** Edge AI models for robotic systems to identify and track targets in dynamic environments.

## ðŸ§¬ Implementation

### `muscle_inference.py`
```python
import tensorflow as tf
import cv2

def predict_muscle_health(image_path):
    model = tf.keras.models.load_model("models/muscle_net_v1.h5")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    
    score = model.predict(img.reshape(1, 224, 224, 3))
    return f"Health Valuation: {score[0][0]:.2f}"
```

## ðŸ“š Publications
- *Autonomous Robotic Navigation in Dynamic Environments (IEEE)*
- *Computer Vision based Muscle Valuation Systems*
