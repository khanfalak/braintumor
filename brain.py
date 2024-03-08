import os
import cv2
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Function to load and preprocess images
def load_images_and_preprocess(path, classes):
    X = []
    Y = []
    for cls in classes:
        pth = os.path.join(path, cls)
        for j in os.listdir(pth):
            img = cv2.imread(os.path.join(pth, j), 0)
            img = cv2.resize(img, (200, 200))
            X.append(img)
            Y.append(classes[cls])
    X = np.array(X)
    Y = np.array(Y)
    X_updated = X.reshape(len(X), -1)
    return X_updated, Y

# Load training data
path = "C:\\Users\\mkash\\Desktop\\dataset\\Training"
classes = {'notumor': 0, 'pituitary': 1}
X, Y = load_images_and_preprocess(path, classes)

# Split data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=10, test_size=.20)

# Perform PCA
pca = PCA(.98)
pca_train = pca.fit_transform(xtrain)
pca_test = pca.transform(xtest)

# Train models
lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)
sv = SVC()
sv.fit(pca_train, ytrain)

# Define class labels
dec = {0: 'No Tumor', 1: 'Positive Tumor'}

# Streamlit UI
st.title('Tumor Classification')

st.write('## Sample images from the "No Tumor" class:')
st.write('Note: Images classified as "No Tumor" are shown here.')

# Display sample images for "No Tumor" class
notumor_dir = "C:\\Users\\mkash\\Desktop\\dataset\\Testing\\notumor"
notumor_images = os.listdir(notumor_dir)[:9]
for i, img_name in enumerate(notumor_images):
    img_path = os.path.join(notumor_dir, img_name)
    img = cv2.imread(img_path, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1) / 255
    p = sv.predict(img1)
    st.image(img, caption=dec[p[0]], use_column_width=True)

st.write('## Sample images from the "Positive Tumor" class:')
st.write('Note: Images classified as "Positive Tumor" are shown here.')

# Display sample images for "Positive Tumor" class
pituitary_dir = "C:\\Users\\mkash\\Desktop\\dataset\\Testing\\pituitary"
pituitary_images = os.listdir(pituitary_dir)[:16]
for i, img_name in enumerate(pituitary_images):
    img_path = os.path.join(pituitary_dir, img_name)
    img = cv2.imread(img_path, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1) / 255
    p = sv.predict(img1)
    st.image(img, caption=dec[p[0]], use_column_width=True)
