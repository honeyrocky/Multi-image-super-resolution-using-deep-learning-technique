from tkinter import *
import tkinter as tk
import cv2
from tkinter import filedialog
from glob import glob
import numpy as np

from PIL import ImageFile                            

from PIL import Image
from PIL import ImageTk

import time
from PIL import ImageTk, Image

from numpy import load
from numpy import save

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image, ImageTk

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from math import floor

from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from math import floor



import cv2
def load_and_preprocess_image(filepath, ratio, resize_height, resize_width):
    # Load the image from the file
    img = image.load_img(filepath)
    # Convert image to a numpy array
    img = image.img_to_array(img)
    # Resize the image
    img = tf.image.resize(img, [resize_height, resize_width], 'bicubic', antialias=True)
    
    # Get dimensions and resize to create a low-resolution version
    height, width, _ = img.shape
    low_res_img = tf.image.resize(img, [height // ratio, width // ratio], 'bicubic', antialias=True)
    
    # Normalize the images to the range [0, 1]
    return low_res_img / 255.0, img / 255.0

# Parameters
#epochs = 10
batch_size = 16
#lr = 1e-3
alpha = 0.1
resize_height = 300
resize_width = 300
ratio = 4
train_val_split_perc = 0.9
val_test_split_perc = 0.5

# Generator class
class Generator(keras.utils.Sequence):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__()-1:
                self.on_epoch_end()

# Preprocessing function
def preprocessing(path, ratio, resize_height, resize_width):
    y = tf.keras.utils.load_img(path)
    y = tf.keras.utils.img_to_array(y)
    y = tf.image.resize(y, [resize_height, resize_width], 'bicubic', antialias=True)
    height, width, _ = y.shape
    x = tf.image.resize(y, [height // ratio, width // ratio], 'bicubic', antialias=True)
    return x / 255.0, y / 255.0

# Load image paths
img_paths = []
for dirname, _, filenames in os.walk('./dataset/train'):
    for filename in filenames:
        img_paths.append(os.path.join(dirname, filename))
val_img_paths = img_paths[floor(len(img_paths) * train_val_split_perc):]
img_paths = img_paths[:floor(len(img_paths) * train_val_split_perc)]

# Load or preprocess tensors
def load_or_preprocess(img_paths, tensor_x_path, tensor_y_path, is_validation=False):
    if not (os.path.exists(tensor_x_path) and os.path.exists(tensor_y_path)):
        img_lr, img_hr = [], []
        for path in tqdm(img_paths):
            x, y = preprocessing(path, ratio, resize_height * (2 if is_validation else 1), resize_width * (2 if is_validation else 1))
            img_lr.append(x)
            img_hr.append(y)
        tensor_x = np.array(img_lr)
        tensor_y = np.array(img_hr)
        np.save(tensor_x_path, tensor_x)
        np.save(tensor_y_path, tensor_y)
    else:
        tensor_x = np.load(tensor_x_path)
        tensor_y = np.load(tensor_y_path)
    return tensor_x, tensor_y

img_lr, img_hr = load_or_preprocess(img_paths, './tensor_x.npy', './tensor_y.npy')
val_img_lr, val_img_hr = load_or_preprocess(val_img_paths, './val_tensor_x.npy', './val_tensor_y.npy', is_validation=True)

# Create data generators
train_generator = Generator(img_lr, img_hr)
val_generator = Generator(val_img_lr[:floor(val_img_lr.shape[0] * val_test_split_perc)], val_img_hr[:floor(val_img_lr.shape[0] * val_test_split_perc)])
test_generator = Generator(val_img_lr[floor(val_img_lr.shape[0] * val_test_split_perc):], val_img_hr[floor(val_img_lr.shape[0] * val_test_split_perc):])

train_dataset = tf.data.Dataset.from_generator(train_generator, output_types=(img_lr.dtype, img_hr.dtype)).batch(batch_size).cache().shuffle(len(train_generator) + 1).prefetch(4)
val_dataset = tf.data.Dataset.from_generator(val_generator, output_types=(val_img_lr.dtype, val_img_hr.dtype)).batch(batch_size).cache().prefetch(4)
test_dataset = tf.data.Dataset.from_generator(test_generator, output_types=(val_img_lr.dtype, val_img_hr.dtype)).batch(batch_size).cache().prefetch(4)

# Custom loss functions
@tf.function
def MeanGradientError(targets, outputs):
    filter_x = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=outputs.dtype)
    filter_x = tf.tile(filter_x[..., None, None], [1, 1, outputs.shape[-1], 1])
    filter_y = tf.transpose(filter_x, [1, 0, 2, 3])

    output_grad_x = tf.square(tf.nn.conv2d(outputs, filter_x, strides=1, padding='SAME'))
    output_grad_y = tf.square(tf.nn.conv2d(outputs, filter_y, strides=1, padding='SAME'))
    target_grad_x = tf.square(tf.nn.conv2d(targets, filter_x, strides=1, padding='SAME'))
    target_grad_y = tf.square(tf.nn.conv2d(targets, filter_y, strides=1, padding='SAME'))

    output_grad = output_grad_x + output_grad_y
    target_grad = target_grad_x + target_grad_y
    return tf.keras.metrics.mean_absolute_error(output_grad, target_grad)

@tf.function
def overall_loss_func(y_true, y_pred):
    mae_loss = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mge_loss = MeanGradientError(y_true, y_pred)
    return mae_loss + alpha * mge_loss

net = tf.keras.models.load_model('./super_resolution_model.h5', custom_objects={'MeanGradientError': MeanGradientError, 'overall_loss_func': overall_loss_func})

# Predict and visualize
outputs = net.predict(test_dataset)

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.pack(fill=BOTH, expand=1)  # Fill the entire window

        # Load the background image
        image_path = "log.jpg"  # Replace this with the path to your image
        img = Image.open(image_path)
        img = img.resize((1600, 800))  # Resize the image to fit the window size
        self.background_image = ImageTk.PhotoImage(img)

        # Create a label with the background image
        self.background_label = Label(self, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Place the label to cover the entire window

        # Changing the title of our master widget      
        self.master.title("Window with Background Image")
        w = tk.Label(self.master, 
		 #text="skindisease detection ",
		 fg = "light green",
		 bg = "brown",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=350, y=0)
        # creating a button instance
        quitButton = Button(self,command=self.query,text="upload",fg="blue",bg="red",width=20)
        quitButton.place(x=50, y=100)
        quitButton = Button(self,text="Enhanced Image",fg="blue",bg="red",width=20)
        quitButton.place(x=530, y=100)


        quitButton = Button(self,text="SuperResolution Image",fg="blue",bg="red",width=20)
        quitButton.place(x=930, y=100)
        
        load = Image.open("logo.jfif")
        render = ImageTk.PhotoImage(load)

        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=400, width=300)
        image1.image = render
        image1.place(x=50, y=200)

        

        image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=400, width=300)
        image1.image = render  # Store the reference to avoid garbage collection
        image1.place(x=500, y=200)

        

        image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=400, width=300)
        image1.image = render  # Store the reference to avoid garbage collection
        image1.place(x=900, y=200)




        


    def query(self, event=None):
        global T, rep
        rep = filedialog.askopenfilenames()

        img = cv2.imread(rep[0])  # Take the first image path from the tuple

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (450, 450))

        Input_img = img.copy()
        print(rep[0])  # Print the path of the selected image
        self.from_array = Image.fromarray(cv2.resize(img, (450, 450)))  # Create an Image object from the array

        render = ImageTk.PhotoImage(self.from_array.resize((450, 450)))  # Resize and convert to Tkinter-compatible format

        image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=400, width=300)
        image1.image = render  # Store the reference to avoid garbage collection
        image1.place(x=50, y=200)

        # Extract the file path correctly
        filepath = rep[0]  # Get the first file path from the tuple

        # Ensure you pass the correct parameters to `load_and_preprocess_image`
        x, y = load_and_preprocess_image(filepath, ratio, resize_height, resize_width)

        # Predict the high-resolution image using your trained model (e.g., net)
        output = net.predict(np.expand_dims(x, axis=0))[0]  # Add batch dimension and predict

        # Convert TensorFlow tensors to NumPy arrays
        x_bgr = cv2.cvtColor((x.numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  # Convert x to NumPy array
        y_bgr = cv2.cvtColor((y.numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  # Convert y to NumPy array
        output_bgr = cv2.cvtColor((output * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  # Convert output to NumPy array
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #img = cv2.cvtColor(y_bgr, cv2.COLOR_BGR2RGB)
        sharpened_output = cv2.filter2D(img, -1, kernel)
        #cv2.imwrite("./enhanced/enhanced image.png",sharpened_output)

        self.from_array = Image.fromarray(cv2.resize(sharpened_output, (450, 450)))  # Create Image object for sharpened output
        # Convert the PIL Image to a NumPy array
        #sharpened_output_array = np.array(self.from_array)
        self.from_array.save("./enhanced/enhanced2.png")

        # Save the image using OpenCV
        #cv2.imwrite("./enhanced/enhanced_image2.png", sharpened_output_array)
        render = ImageTk.PhotoImage(self.from_array.resize((450, 450)))  # Convert to Tkinter-compatible format
        #cv2.imwrite("./enhanced/enhanced image1.png",render)

        image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=400, width=300)
        image1.image = render  # Store the reference to avoid garbage collection
        image1.place(x=900, y=200)

        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_output = cv2.filter2D(img, -1, kernel)
        #cv2.imwrite("./superresolution/superresolution image.png",sharpened_output)

        self.from_array = Image.fromarray(cv2.resize(sharpened_output, (450, 450)))  # Create Image object for sharpened output
        render = ImageTk.PhotoImage(self.from_array.resize((450, 450)))
        self.from_array.save("./superresolution/superresolution1.png")
        #cv2.imwrite("./superresolution/superresolution image1.png",render)

        

        image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=400, width=300)
        image1.image = render  # Store the reference to avoid garbage collection
        image1.place(x=500, y=200)

        x_bgr_scaled = (x_bgr/255.0)*1
        y_bgr_scaled = (y_bgr/255.0)*1

        print(x_bgr_scaled)
        
        print(y_bgr_scaled)



from tkinter import messagebox
from PIL import Image, ImageTk

class LoginWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.config(bg='white')
        self.master.title("Login")

        # Load background image
        bg_image = Image.open("log.jpg")
        bg_render = ImageTk.PhotoImage(bg_image)

        # Create a label to hold the background image
        self.background_label = Label(self, image=bg_render)
        self.background_label.image = bg_render
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.pack(fill=BOTH, expand=1)
        
        w = Label(self, text="Multi Image SuperResolution Login Page", fg="#e6f2ff", bg="black", font="Helvetica 20 bold italic")
        w.pack()
        w.place(x=450, y=200)
        
        self.username_label = Label(self, text="Username:")
        self.username_label.place(x=480, y=300)
        
        self.username_entry = Entry(self)
        self.username_entry.place(x=550, y=300)
        
        self.password_label = Label(self, text="Password:")
        self.password_label.place(x=480, y=350)
        
        self.password_entry = Entry(self, show="*")
        self.password_entry.place(x=550, y=350)
        
        self.login_button = Button(self, text="Login", command=self.login)
        self.login_button.place(x=550, y=400)
    
    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        
        if username == "jai" and password == "chandru":
            self.master.switch_frame(Window)

        elif username == "me" and password == "gha":
             self.master.switch_frame(Window)

        elif username == "shr" and password == "uthi":
             self.master.switch_frame(Window)    
             
        else:
            messagebox.showerror("Error", "Invalid username or password")    
             
        


class MainApplication(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        
        self.title("Multi Image SuperResolution")
        self.geometry("1400x720")
        self.current_frame = None
        self.switch_frame(LoginWindow)
        
    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack()

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()



  
  
