def query(self, event=None):
    global T, rep
    rep = filedialog.askopenfilenames()

    # Load the selected image
    img = cv2.imread(rep[0])  # Take the first image path from the tuple
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (450, 450))
    Input_img = img.copy()

    # Display the input image
    self.from_array = Image.fromarray(cv2.resize(img, (450, 450)))
    render = ImageTk.PhotoImage(self.from_array.resize((450, 450)))
    image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=400, width=300)
    image1.image = render
    image1.place(x=50, y=200)

    # Preprocess and predict
    filepath = rep[0]
    x, y = load_and_preprocess_image(filepath, ratio, resize_height, resize_width)
    output = net.predict(np.expand_dims(x, axis=0))[0]

    # Convert predictions to displayable images
    x_bgr = cv2.cvtColor((x.numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    y_bgr = cv2.cvtColor((y.numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    output_bgr = cv2.cvtColor((output * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Enhance the image (apply sharpening)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_output = cv2.filter2D(output_bgr, -1, kernel)

    # Display enhanced image
    self.from_array = Image.fromarray(cv2.resize(enhanced_output, (450, 450)))
    render = ImageTk.PhotoImage(self.from_array.resize((450, 450)))
    image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=400, width=300)
    image1.image = render
    image1.place(x=500, y=200)

    # Display super-resolution image
    self.from_array = Image.fromarray(cv2.resize(output_bgr, (450, 450)))
    render = ImageTk.PhotoImage(self.from_array.resize((450, 450)))
    image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=400, width=300)
    image1.image = render
    image1.place(x=900, y=200)

    # Save functionality
    def save_image(image, default_name):
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg", initialfile=default_name, filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if filepath:
            cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Success", f"Image saved as {filepath}")

    save_enhanced_button = Button(self, text="Save Enhanced Image", command=lambda: save_image(enhanced_output, "enhanced_image.jpg"))
    save_enhanced_button.place(x=500, y=650)

    save_superres_button = Button(self, text="Save SuperResolution Image", command=lambda: save_image(output_bgr, "super_resolution_image.jpg"))
    save_superres_button.place(x=900, y=650)
