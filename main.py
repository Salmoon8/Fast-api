from fastapi import FastAPI , Request
from generator import generator
from model import model
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO


# Read image
image_pil = Image.open('0_114_test.png')

if image_pil.mode != 'L':
    image_pil = image_pil.convert('L')

# Convert image to NumPy array
image_np = np.array(image_pil)
print (image_np.shape)
# image_np=np.expand_dims(a=image_np,axis=2)

app = FastAPI()

model.load_weights("model.keras")
generator.load_weights("generator_weights.h5")

def predict(image):
    image=tf.convert_to_tensor(image)
    c_img = tf.cast(image, tf.float32)
    c_img = tf.image.resize(c_img, [512,512])
    c_img = (c_img / 255.0)
    pred = model.predict(c_img)
    return pred

def generate(image):
    image=np.expand_dims(a=image,axis=2)
    image=tf.convert_to_tensor(image)
    g_img = tf.cast(image, tf.float32)
    g_img = tf.image.resize(g_img, [256, 256])
    g_img = (g_img / 127.5) - 1.0
    g_img = tf.expand_dims(g_img, axis=0)
    
    generated = generator(g_img, training=False)[0].numpy()
    generated = (generated * 127.5 + 127.5).astype(np.uint8)
    return generated





@app.post("/instanceid")
async def root(request: Request):
    body = await request.json()
    instanceid=body["instanceid"]
    response=requests.get(f"http://localhost:8042/instances/{instanceid}/numpy")
    array=np.load(BytesIO(response.content))
    print (body,array.shape)
    pred=predict(array)
    if pred >= 0.5:
        pred=True
    else:
        pred=False

    return {"classification":pred }