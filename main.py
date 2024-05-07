from fastapi import FastAPI , Request
from generator import generator
from keras.models import load_model
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

# Read image
image_pil = Image.open('0_114_test.png')

if image_pil.mode != 'L':
    image_pil = image_pil.convert('L')

# Convert image to NumPy array
image_np = np.array(image_pil)
print (image_np.shape)
# image_np=np.expand_dims(a=image_np,axis=2)

app = FastAPI()

model = load_model("classification_model.keras")
generator.load_weights("generator.weights.h5")


def predict(image, size=224):
    c_img = tf.cast(image, tf.float32)
    c_img = tf.image.resize(c_img, [size,size])
    c_img = tf.expand_dims(c_img, axis=0)
    print(c_img.shape)
    pred = model.predict(c_img)
    return pred



def preprocess_dicom(dcm):
    image = dcm.pixel_array
    
    image_win = apply_voi_lut(image, dcm)
    
    if np.sum(image_win) == 0:
        image_win = image

    pixels = image_win - np.min(image_win)
    pixels = pixels / np.max(pixels)
    image = (pixels * 255).astype(np.uint8)

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        inverted_image = 255 - image
        image = tf.cast(inverted_image, tf.uint8)

    image = np.stack([image] * 3, axis=-1)

    h,w=image.shape[0],image.shape[1]
    max_dir=max(w,h)
    padded_image = np.zeros((max_dir, max_dir,3), dtype=image.dtype)

    left_zeros = np.sum(image[:,:image.shape[1]//2] == 0)
    right_zeros = np.sum(image[:,image.shape[1]//2:] == 0)

    if left_zeros >= right_zeros:
        padded_image[0:image.shape[0]:, max_dir - image.shape[1]:] = image[:,:,:]  # Right lateral, pad on the left side
    else:
        padded_image[0:image.shape[0], :image.shape[1],:] = image[:,:,:]  # Left lateral, pad on the right side

    image_resized = cv2.resize(padded_image, (512, 512))
    
    print(image_resized.shape)
    
    return image_resized 

def pad_to_square(image, laterality):
    """
    Pad mammogram image to make it square while considering laterality (right vs left).
    
    Parameters:
    - image: Input mammogram image as a numpy array.
    - laterality: Laterality of the image ('R' for right, 'L' for left).
    
    Returns:
    - max is the maximum height or width
    - Padded square image as a numpy array of size (max,max).
    """
    # Create empty square canvas with the target size
    w,h=image.shape[0],image.shape[1]
    max_dir=max(w,h)
    padded_image = np.zeros((max_dir, max_dir), dtype=image.dtype)
    
    # Calculate padding dimensions based on laterality
    if laterality == 'R':
        padded_image[0:image.shape[0]:, max_dir - image.shape[1]:] = image[:,:]  # Right lateral, pad on the left side
    elif laterality == 'L':
        padded_image[0:image.shape[0], :image.shape[1]] = image[:,:]  # Left lateral, pad on the right side
    else:
        raise ValueError("Invalid laterality. Expected 'R' or 'L'.")
    
    return padded_image 

def preprocess_numpy(dcm,lat):
    # apply lut
    image = dcm.pixel_array
    
    image_win = apply_voi_lut(image, dcm)
    
    if np.sum(image_win) == 0:
        image_win = image

    pixels = image_win - np.min(image_win)
    pixels = pixels / np.max(pixels)
    image = (pixels * 255).astype(np.uint8)
    print(image.shape)
    # padding
    padded=pad_to_square(image,lat)
    resized=cv2.resize(padded, (256, 256))
    image=np.expand_dims(a=resized,axis=2)
    image=tf.convert_to_tensor(image)
    g_img = tf.cast(image, tf.float32)
    g_img = tf.image.resize(g_img, [256, 256])
    g_img= (g_img / 127.5 ) -1
    g_img = tf.expand_dims(g_img, axis=0)
    generated = generator(g_img, training=False)
    generated = (generated[0]* 127.5 + 127.5)
    cv2.imwrite('generat.jpg',generated.numpy() )
    print(generated)
    return generated.numpy()

def create_dcm_pxlarray(pixel,dcm):
    dcm.pixeldata=pixel[:,:,0].tobytes()
    dcm.Rows,dcm.Columns=pixel.shape[0],pixel.shape[1]
    dcm.save_as('modified.dcm')


@app.post("/instanceid")
async def root(request: Request):
    body = await request.json()
    instanceid=body["instanceid"]
    response=requests.get(f"http://localhost:8042/instances/{instanceid}/file")
    image_array_response= requests.get(f"http://localhost:8042/instances/{instanceid}/numpy")
    print(image_array_response)
    array=np.load(BytesIO(image_array_response.content))
    print(array.max())
    print(array.shape)
    dcm = pydicom.dcmread(BytesIO(response.content))
    laterality=dcm.ImageLaterality
    generator_img=preprocess_numpy(dcm,laterality)
    create_dcm_pxlarray(generator_img,dcm)
    image = preprocess_dicom(dcm)
    # print (dcm.)
    pred=predict(image)
    if pred >= 0.5:
        pred=True
    else:
        pred=False

    return {"classification":pred, "laterality": dcm.ImageLaterality }