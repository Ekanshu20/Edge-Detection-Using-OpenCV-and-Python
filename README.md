# 🖼️ Sobel Edge Detection using OpenCV

This project demonstrates **edge detection** on images using the **Sobel operator** in Python via OpenCV. It focuses on detecting vertical and horizontal edges by computing intensity gradients in both the X and Y directions, helping to identify the outlines and structures in an image.

---

## 📑 Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Edge Detection Technique](#edge-detection-technique)
- [Evaluation and Output](#evaluation-and-output)
- [Author](#author)

---

## 🚀 Getting Started

This section will guide you through setting up and running the project on your local machine.

---

### 📋 Prerequisites

- Python 3.10.14 (which I used)
- Required libraries:
  - OpenCV (`cv2`)
  - NumPy
  - Matplotlib

---

## ⚙️ Installation

1. Download data required for performing edge detection using sobel operator.
You can downnload data from this site https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500

2. Install Dependencies 
```python
pip install opencv-python numpy matplotlib
```
---

## 🧪 Usage
Open the Jupyter Notebook:
1. Follow the steps inside the notebook:

Step 1: Load and Preprocess Image
Convert to grayscale:
```python
image = cv2.imread('your_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Step 2: Apply Sobel Filters
Detect edges in horizontal and vertical directions:

```python
grad_x = cv2.filter2D(gray, cv2.CV_32F, sobel_x)
grad_y = cv2.filter2D(gray, cv2.CV_32F, sobel_y)
```

Step 3: Calculate magnitude of gradients
Combine gradient magnitudes to create final edge map
```python
magnitude = np.sqrt(grad_x**2 + grad_y**2)
```
Normalize to 0-255 range
```python
edge_map = np.uint8(255 * magnitude / np.max(magnitude))
```

Step 4: Display Results
Visualize using Matplotlib:
```python
plt.imshow(grad_x, cmap='gray')
plt.title('Sobel X (Vertical Edges)')
```
--- 

## 🧠 Edge Detection Technique
Sobel Operator
```python
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
```

1. The Sobel operator is a discrete differentiation operator that computes an approximation of the gradient of image intensity. It highlights edges by emphasizing regions with high spatial derivatives.

2. Sobel X → detects vertical edges (intensity changes in the X direction).

3. Sobel Y → detects horizontal edges (intensity changes in the Y direction).

These filters help identify the contours and structural outlines in the image.

---

## 🎯 Evaluation and Output
The notebook will display:

Original image

Grayscale version

Sobel X (vertical edges)

Sobel Y (horizontal edges)

Final edge map image

You can try different kernel sizes and images to see how edge detection varies with changes.

The .ipynb file along with zipped file which contains images of Final Edge Map has been attached for complete walkthrough and visualization.

---

## 👨‍💻 Author
**Ekanshu Agrawal**

📧 ekanshu20agrawal@gmail.com

🔗 [LinkedIn] (https://www.linkedin.com/in/ekanshu20)
