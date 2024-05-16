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
import numpy as np
from pydicom.dataset import Dataset
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import requests
from io import BytesIO
from pydicom import dcmread, dcmwrite
from pydicom.filebase import DicomFileLike

from pydicom.dataset import  FileMetaDataset
from pydicom.sequence import Sequence
import threading


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

def send_to_orthanc(dataset):
    # create a buffer
    with BytesIO() as buffer:
        # create a DicomFileLike object that has some properties of DataSet
        memory_dataset = DicomFileLike(buffer)
        # write the dataset to the DicomFileLike object
        dcmwrite(memory_dataset, dataset)
        # to read from the object, you have to rewind it
        memory_dataset.seek(0)
        # read the contents as bytes
        binary_data=  memory_dataset.read()
    porthanc=requests.post("http://localhost:8042/instances",data=binary_data,headers={"content-type":'application/dicom'})
    print(porthanc.status_code)
    print(porthanc.text)
    return porthanc.status_code


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
    
    img = g_img[0] * 127.5 + 127.5
    cv2.imwrite('image.jpg',img.numpy())
    cv2.imwrite('generated.jpg',generated.numpy())
    #print(generated)
    return generated.numpy()

def create_dcm_pxlarray(pixel,dcm):
    meta = pydicom.Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.DigitalMammographyXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  

    ds = Dataset()
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.DigitalMammographyXRayImageStorageForPresentation
    ds.PatientName = dcm.PatientName
    ds.PatientID = dcm.PatientID

    ds.Modality = "MG"
    ds.SeriesInstanceUID = dcm.SeriesInstanceUID
    ds.StudyInstanceUID = dcm.StudyInstanceUID
    ds.SOPInstanceUID=pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15
    ds.ImagesInAcquisition = "1"
    ds.Rows = pixel.shape[0]
    ds.Columns = pixel.shape[1]
    ds.InstanceNumber = 1
    ds.ImagePositionPatient = r"0\0\1"
    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.ImageType = r"DERIVED\SECONDARY"

    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = r"1\1"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    print("Setting pixel data...")
    ds.PixelData = pixel[:,:,0].astype(np.uint16).tobytes()
    print("tags",ds.PatientID,ds.StudyInstanceUID,ds.SeriesInstanceUID)
    # send to orthanc
    ds.save_as("outputfile.dcm", write_like_original=False)
    dataset=pydicom.dcmread('outputfile.dcm',force=True)
    try: 
        request_thread = threading.Thread(target=send_to_orthanc, args=(ds,))
        request_thread.start()
    except Exception as e:
        print (e)
    
    return "awaiting generation"

    

@app.post("/instanceid")
async def root(request: Request):
    body = await request.json()
    instanceid=body["instanceid"]
    response=requests.get(f"http://localhost:8042/instances/{instanceid}/file")
    dcm = pydicom.dcmread(BytesIO(response.content))
    laterality=dcm.ImageLaterality
    image = preprocess_dicom(dcm)
    # print (dcm.)
    pred=predict(image)
    if pred >= 0.5:
        pred=True
    else:
        pred=False

    return {"classification":pred, "laterality": dcm.ImageLaterality }

@app.post("/generate")
async def root(request: Request):
    body = await request.json()
    instanceid=body["instanceid"]
    response=requests.get(f"http://localhost:8042/instances/{instanceid}/file")
    dcm = pydicom.dcmread(BytesIO(response.content))
    laterality=dcm.ImageLaterality
    image_array_response= requests.get(f"http://localhost:8042/instances/{instanceid}/numpy")
    array=np.load(BytesIO(image_array_response.content))
    generator_img=preprocess_numpy(dcm,laterality)
    code= create_dcm_pxlarray(generator_img,dcm)
    return {"generation_code":code }