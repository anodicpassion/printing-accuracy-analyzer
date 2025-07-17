# Print Quality Analysis Tool

This project provides a web-based application for analyzing the print quality of scanned documents by comparing them against a template. It identifies issues such as misalignment and blur, providing visual feedback and quantitative metrics.
---

https://github.com/user-attachments/assets/d488136b-b1d7-48fe-bcc9-ffd6febb1ca0


---


## Technologies Used
This project leverages a combination of Python libraries and a web framework to deliver its functionality:
Python: The core programming language used for the application logic.
- Flask: A lightweight and flexible micro web framework for Python. Flask is used to handle web requests, route URLs to Python functions, render HTML templates, and manage the overall web application structure. Its simplicity allows for rapid development and gives developers full control over the application's components.
- OpenCV (Open Source Computer Vision Library): A powerful library for computer vision and image processing tasks. OpenCV is central to this project, enabling:
   - Image Loading and Manipulation: Reading and processing image files (cv2.imread, cv2.cvtColor).
   - Feature Detection and Matching: Using ORB (Oriented FAST and Rotated BRIEF) algorithm (cv2.ORB_create, orb.detectAndCompute, cv2.DescriptorMatcher_create, matcher.match) to find key points and descriptors for image alignment.
   - Image Alignment (Homography): Calculating a homography matrix (cv2.findHomography) to align the scanned image with the template, correcting for perspective and rotation differences.
   - Image Differencing: Calculating the absolute difference between images (cv2.absdiff) to identify variations.
Image Thresholding and Morphological Operations: Applying thresholding (cv2.threshold) and dilation (cv2.dilate) to enhance differences and prepare for contour detection.
- Contour Detection: Finding contours (cv2.findContours) to identify misaligned regions.
- Drawing on Images: Drawing bounding boxes (cv2.rectangle) to visually highlight detected issues.
Blur Detection: Utilizing the Laplacian variance method (cv2.Laplacian, .var()) to assess image sharpness.
- NumPy: The fundamental package for scientific computing with Python. NumPy provides support for large, multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays. In this project, OpenCV images are represented as NumPy arrays, and NumPy is used for array manipulations and calculations, such as in the homography estimation and image differencing.
- os Module: Python's built-in module for interacting with the operating system. It's used for tasks like creating directories (os.makedirs), joining file paths (os.path.join), and deleting files (os.remove).
- uuid Module: Python's built-in module for generating Universally Unique Identifiers (UUIDs). It's used to create unique filenames for uploaded images, preventing naming conflicts and ensuring secure storage.
- Werkzeug: A comprehensive WSGI utility library that Flask is built upon. It handles fundamental web aspects like requests, responses, and routing behind the scenes, providing the underlying structure for Flask's operations.



---

## Installation 
- Clone the repo


```shell
git clone https://github.com/anodicpassion/printing-accuracy-analyzer.git
```

- Change to project dir


```shell
cd printing-accuracy-analyzer
```

- Create a virtual environment and activate it


```shell
python3 -m venv .venv
source .venv/bin/activate
```


- Kick-off the webserver


```shell
python3 app.py
```


--- 


## License

This project is licensed under the GNU General Public License v3.0.
See the LICENSE file for more details.
