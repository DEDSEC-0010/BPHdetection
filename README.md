<h1>Brown Plant Hopper Detection</h1>
This is a Python code for a Brown Plant Hopper Detection application using computer vision and machine learning techniques. The code allows you to detect and classify the presence of brown plant hoppers in real-time video streams or uploaded images.

Dependencies
Python 3.x
OpenCV (cv2)
NumPy (numpy)
Tkinter (tk)
PIL (PIL)
TensorFlow (tensorflow)
Install the required dependencies using the following command:

bash
Copy code
pip install opencv-python numpy tkinter pillow tensorflow
Make sure you have the necessary models and images in the specified paths before running the code.

Usage
Import the required libraries:
python
Copy code
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
Load the pre-trained model:
python
Copy code
model = tf.keras.models.load_model("D:\Mini Project\ImageClassification\imageclassifier.h5")
Define the class labels for classification:
python
Copy code
class_labels = ["Infected", "Not Infected"]
Set up the video capture:
python
Copy code
cap = cv2.VideoCapture(0)
Set the threshold value for image binarization:
python
Copy code
threshold = 128  # Specify your desired threshold value
Set up the data augmentation generator:
python
Copy code
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)
Create the main window and set its properties:
python
Copy code
window = tk.Tk()
window.title("Brown Plant Hopper Detection")
window.geometry("640x480")
Load and display the background image:
python
Copy code
background_image = ImageTk.PhotoImage(
    Image.open("D:\DLoads\hand-drawn-minimal-background\\5586989.jpg")
)
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
Create labels for displaying video frames and uploaded images:
python
Copy code
video_label = tk.Label(window)
video_label.pack()

uploaded_image_label = tk.Label(window)
uploaded_image_label.pack()
Load icons for buttons:
python
Copy code
start_icon = ImageTk.PhotoImage(Image.open("D:\DLoads\icons8-start-50 (1).png"))
stop_icon = ImageTk.PhotoImage(Image.open("D:\DLoads\icons8-stop-circled-50.png"))
browse_icon = ImageTk.PhotoImage(Image.open("D:\DLoads\icons8-browse-50.png"))
back_icon = ImageTk.PhotoImage(Image.open("D:\DLoads\icons8-go-back-50.png"))
Define the function for processing video frames:
python
Copy code
def process_video():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip horizontally
    resized = cv2.resize(frame, (256, 256))
    normalized = resized / 255.0
    augmented_frame = datagen.random_transform(normalized)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    prediction = model.predict(np.expand_dims(augmented_frame, axis=0))[0]
    predicted_class = class_labels[int(np.round(prediction))]

    cv2.putText(
        frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    video_label.img_tk = img_tk
    video_label.configure(image=img_tk)

    video_label.after(10, process_video)
Define functions for button actions:
python
Copy code
def start_video():
    process_video()
    start_button.pack_forget()  # Hide the start button
    back_button.pack(side="left")  # Show the back button

def stop_video():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        try:
            image = cv2.imread(file_path)
            resized = cv2.resize(image, (256, 256))
            normalized = resized / 255.0
            augmented_image = datagen.random_transform(normalized)

            prediction = model.predict(np.expand_dims(augmented_image, axis=0))[0]
            predicted_class = class_labels[int(np.round(prediction))]
            messagebox.showinfo(
                "Prediction", f"The predicted class is: {predicted_class}"
            )

            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            uploaded_image_label.img_tk = img_tk
            uploaded_image_label.configure(image=img_tk)

        except Exception as e:
            messagebox.showerror("Error", str(e))

def go_back():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()
    # Call your menu function or display the menu again here
Create buttons and pack them in the window:
python
Copy code
start_button = tk.Button(window, image=start_icon, command=start_video)
start_button.pack(side="left")

stop_button = tk.Button(window, image=stop_icon, command=stop_video)
stop_button.pack(side="left")

browse_button = tk.Button(window, image=browse_icon, command=browse_image)
browse_button.pack(side="left")

back_button = tk.Button(window, image=back_icon, command=go_back)
Run the Tkinter event loop:
python
Copy code
window.mainloop()
Running the Code
To run the code, execute the Python script. The main window will appear with the buttons for starting and stopping the video, browsing an image, and going back. Click the corresponding buttons to perform the desired actions.

The "Start" button initiates the video stream and starts the brown plant hopper detection process.
The "Stop" button stops the video stream and closes the application.
The "Browse" button allows you to select an image file to perform the detection on.
The "Back" button is used to go back or return to the previous menu or interface.
When the video stream or image is processed, the predicted class ("Infected" or "Not Infected") will be displayed on the video frame or in a message box.

Note: Make sure the necessary models and images are available in the specified paths before running the code.

Feel free to modify the code according to your specific requirements and paths.
