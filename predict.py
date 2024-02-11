try:
    import os
    import base64
    import tempfile
    import numpy as np
    import tensorflow as tf
    import pydicom
    import cv2
    from PIL import Image as img
    import matplotlib.pyplot as plt
    import scipy
    from tensorflow.keras.models import Model
    from io import BytesIO
    import json
    import httplib2
    import boto3 as boto
    import contextlib
    import sys
    from pydicom.uid import generate_uid
    import pymysql
    import os
    import sys
    # import mysql.connector
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.layers import LeakyReLU
    from sklearn.neighbors import KernelDensity
    import joblib
    from datetime import datetime
    from dotenv import load_dotenv
    import scipy
    from pydicom.uid import generate_uid
    from PIL import Image    
    print("Successfully Imported Libraries News")
    

except ImportError as ie:
    print("Failed to import modules")
    print("Details: {}".format(ie))
    exit(1)
    
#get current time at prediction
now=datetime.now()
date = now.strftime("%Y%d%m")
print(date)

load_dotenv()
print("Reading Environmental Variables")

aws_access_key=os.getenv("AWS_S3_PACS_ACCESS_KEY")
aws_secret_key=os.getenv("AWS_S3_PACS_SECRET_KEY")
pacs_url=os.getenv("URL")


print(pacs_url)
if (aws_access_key == None) or (aws_secret_key == None):
    print("Failed to access AWS ACCESS KEYS Terminating Program")
    exit(1)
elif pacs_url == None:
    print("Failed to fetch PACS end point, Terminating Program")
    exit(1)
else:
    print("Successfully Retrieved AWS Access Key and PACS end point")

 

#connect to s3 for simpler uploading
@contextlib.contextmanager
def s3connection(key, secret):
    print("Connecting to S3")
    
    #
    if (key == None) or (key == None):
        print("Failed to access AWS ACCESS KEYS Terminating Program")
        exit(1)
    else:
        print("Successfully Retrieved AWS Access Key")
        
    
    try:
        session=boto.session.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        print("Successfully created session")
        print("Established Connection")
        yield session
    except Exception as e:
        print("Error creating session")
        print("Error Details: {}".format(e))
        exit(1)
        
    # s3 = boto3.client('s3', aws_access_key_id=key, aws_secret_access_key=secret)
    
    # try:
    #     yield session
    # except Exception as e:
    #     print("Error occurred: ".format(e))
    #     sys.exit(1)
    # # except NoCredentialsError:
    #     print("Credentials not available")
    #     sys.exit(1)

    finally:
        print("Keeping Connection")
        

with s3connection(aws_access_key, aws_secret_key) as s3:
    print("success")
    
#load model pretrained model
print("Retriving model")
def get_model():
    #     print("Accessing model")
    #     s3.download_file(Bucket='diagnosoft-model',key='BaseU.hdf5',Filename='BaseU.hdf5')
    try:
        print("Fetchong models from bucket")
        download_session = s3.client('s3')
        
        download_session.download_file('diagnetv-v24-models','Diagnet24V1-1-m.hdf5','models/Diagnet24V1-1-m.hdf5')
        download_session.download_file('diagnetv-v24-models','Diagnet24V1-1-g.hdf5','models/Diagnet24V1-1-g.hdf5')
        print("Successfully fetched model...")
        print("Accessing models...")
        model=tf.keras.models.load_model("models/Diagnet24V1-1-m.hdf5")
        # loaded_model = joblib.load(model_filename)
        # pred_model=Model(inputs=model.input,outputs=(model.layers[-4].output,model.layers[-1].output))
        pred_model=tf.keras.models.load_model("models/Diagnet24V1-1-g.hdf5")
        # last_conv_output,pred_out=pred_model.predict(y)
        # last_conv_out=np.squeeze(last_conv_output)
        # last_layer_weights=model.layers[-1].get_weights()[0]
        
        print("Successfully Loaded model")
        return model,pred_model
        
    except FileNotFoundError as fnf:
        print("Error fetching model: ")
        print("Model path doesn't exist")
        exit(1)
    except Exception as e:
        print("Error fetching model: ")
        print("Details: {}".format(e))
        exit(1)
    
      

def IsJson(content):
    try:
        if (sys.version_info >= (3, 0)):
            json.loads(content.decode())
            return True
        else:
            json.loads(content)
            return True
    except Exception as e:
        print(e)
        return False

def img_to_base64_str(img):
    buffered=BytesIO()
    img.save(buffered,format="PNG")
    buffered.seek(0)
    img_bytes=buffered.getvalue()
    img_str="data:image/png;base64,"+base64.b64encode(img_bytes).decode()
    
    return img_str

# try cropping the image
def center_crop(img, new_width=None, new_height=None):        

    width = img.size[1]
    height = img.size[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.size) == 2:
        # Crop the center of the image
        center_cropped_img = img.crop((left, top, right, bottom))
        # center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


# take the image and convert into Grayscale first
def clean_pipeline(scan_file,width=2400,height=2000,scale="RGB"):
    #open file
    scan=img.open(scan_file).convert(scale)
    # print(scan.size)
    cropped_scan=scan.resize((width,height))
    # cropped_scan=center_crop(scan,width,height)
    
    return cropped_scan

def scan_to_numpy(img_path,img_width=512,img_height=512,grayscale=False):
    # transformed_img=[]
    
    if grayscale:
        image=clean_pipeline(img_path,img_width,img_height,scale="L")
        
    else:
        image=clean_pipeline(img_path,img_width,img_height,scale="RGB")
    
    prep_scan = np.array(image)
    # transformed_img.append(prep_scan)
    
    return prep_scan

def preprocess_image(file_path,p_name="None",patient_id=1,inspection_code=2,file_type="DCM"):
    if file_type.lower() == "dcm":
        print("Beginning image preprocessing")
        dicom_file_folder="scans/dicom_files/"
        processed_folder="downloads/processed/"
        
        if not os.path.exists(dicom_file_folder):
            os.makedirs(dicom_file_folder)
            print("Dicom Directory created successfully.")
        else:
            print("Directory already exists.")
            
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
            print("Processed Directory created successfully.")
        else:
            print("Directory already exists.")
        
        new_filepath="downloads/processed/{}.DCM".format(p_name)
        
        test_image=[]
        print("Reading DICOM image")
        image=pydicom.dcmread(file_path)
        # image.StudyInstanceUID='1.3.6.1.4.1.14519.5.2.1.7009.2404.315222469415623828162094912079'
        # image.SeriesInstanceUID='1.3.6.1.4.1.14519.5.2.1.7009.2404.309111601391566216060061725328'
        # image.SOPInstanceUID='1.3.6.1.4.1.14519.5.2.1.7009.2404.170420932819315718805816034120'
        # image.StudyDate=date
        image.StudyInstanceUID=generate_uid()
        image.SeriesInstanceUID=generate_uid()
        image.SOPInstanceUID=generate_uid()
        image.PatientName=p_name
        image.PatientID=str(inspection_code)
        pydicom.dcmwrite(new_filepath,image)
        print("Successfully Input patient detail")
        # print(image)
        # print(image.pixel_array)
        image=image.pixel_array
        print(image)
        cv2.imwrite('scans/dicom_files/{}.png'.format(p_name),image)
        image=cv2.imread('scans/dicom_files/{}.png'.format(p_name))
        print(image)
        image=Image.fromarray(image,'RGB')
        image=image.resize((512,512))
        test_image.append(np.array(image))  
        test_image=np.array(test_image)
        test_image=test_image/255
        print("Done image prep ")
        return test_image
        
    elif file_type.lower() != "dcm":
        print("Processing Different image type")
        scan=scan_to_numpy(file_path)
        scan_np=np.array(scan)
        scan_file=np.expand_dims(scan_np,axis=0)
        print("Done")
        
        return scan_file        
    

def upload_image(path,inspection_code = int("1")):
    
    f=open(path,'rb')
    content=f.read()
    f.close()    
    
    print("Importing file: ")
    
    if IsJson(content):
        print("Ignored JSON file")
        json_count+=1
        return
    try:
        h=httplib2.Http()
        headers={'content-type':'application/dicom'}
        username='orthanc'
        password='orthanc'
        creds_str=username+':'+password
        creds_str_bytes=creds_str.encode('ascii')
        creds_str_bytes_b64=b'Basic '+base64.b64encode(creds_str_bytes)
        headers['authorization']=creds_str_bytes_b64.decode('ascii')
        resp,content=h.request(pacs_url,'POST',body=content,headers=headers)
        
        
        if resp.status == 200:
            
            # query='''
            # UPDATE diagnosoft_rds_ris.ExaminationRequest
            # SET uploaded=1
            # WHERE Inspection_code = %s
            
            # '''
            # cursor.execute(query,(inspection_code,))
            # connection.commit()
            print("Updated database record")
            print("Successfully uploaded file: ")
            
        else:
            print(resp.status)
            print("Error uploading file: ")
            print("Is it a dicom file?")
            
    except Exception as e:
        type,value,traceback=sys.exc_info()
        print(str(value))
        print("Unable to connect to Orthanc Server")
        print(e)
   

def prediction_scan(img,p_name="None"):
    #adding the prediction model
    #adding 
    # pred_model=Model(inputs=(512,512,3),outputs=(model.layers[-4].output,model.layers[-1].output)) 
    
    #expand dimensions
    #predict
    model,pred_model=get_model()
    scan_image=img
    img=np.expand_dims(img,axis=0)
    print("Spooling up model")
    
    last_conv_output,pred_out=pred_model.predict(img)
    last_conv_out=np.squeeze(last_conv_output)
    last_layer_weights=model.layers[-1].get_weights()[0]
    
    pred_t=np.argmax(pred_out)
    if pred_t == 0:
        pred_class = "TB Detected"
        text="Tuberculosis Detected in Scan"
        print(img.shape[1])
        
        print(text)
        # spline interpolation to resize each filtered image to size of original image 
        h = int(img.shape[1]/last_conv_out.shape[0])
        # h=512
        # w=512
        w = int(img.shape[2]/last_conv_out.shape[1])
        print("Generating heatmap")
        upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_out, (h, w, 1), order=1) # dim: 224 x 224 x 2048

        print("Getting last weights for prediction")
        #Get the weights from the last layer for the prediction class
        last_layer_weights_for_pred = last_layer_weights[:, pred_t] # dim: (2048,) 

        #To generate the final heat map. 
        #Reshape the upsampled last conv. output to n x filters and multiply (dot product) 
        # with the last layer weigths for the prediction. 
        # Reshape back to the image size for easy overlay onto the original image. 
        print("Stamping heatmap")
        heat_map = np.dot(upsampled_last_conv_output.reshape((512*512, 2048)), 
                        last_layer_weights_for_pred).reshape(512,512) # dim: 224 x 224

        #We can fetch the actual class name for the prediction so we can add it as
        # a title to our plot.
        # with open('imagenet_classes.txt') as imagenet_classes_file:
        #         imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())
        # predicted_class = imagenet_classes_dict[pred]   

        #saving heatmap to s3 bucket
        # plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
        # plt.imshow(heat_map, cmap='jet', alpha=0.30)
        
        print("Uploading heatmap to s3 bucket")
        plt.imshow(heat_map, cmap='jet', alpha=0.30)
        plt.savefig('scans/heatmaps/{}.png'.format(p_name))
        uploading_session = s3.client('s3')
        inspection_code="45"
        uploading_session.upload_file(Filename="scans/heatmaps/{}.png".format(p_name),Bucket="diagnetv24-heatmaps",Key=p_name+".png")

        #Plot original image with heatmap overlaid.
        print("Displaying output") 
        fig, ax = plt.subplots()
        ax.imshow(scan_image)
        ax.imshow(heat_map, cmap='jet', alpha=0.30)
        # ax.set_title(predicted_class)
        print("Done") 
        
        image=cv2.imread('scans/heatmaps/{}.png'.format(p_name))
        image=Image.fromarray(image,'RGB')
        image=image.resize((512,512))
        print("Calculating prediction")
        # plt.show()
        return [{'text':text,'class':pred_class,'image':img_to_base64_str(image)}]
    else:
        text='No pneumothorax detected with confidence probability of '
        pred_class="No TB"
        print("No TB detected")
        return [{'text':text,'class':pred_class,'image':img_to_base64_str(image)}]
    
def predict_base64_image(name,patient_name,inspection_code,contents,file_type):
    print("receiving image")
    fd,file_path=tempfile.mkstemp()
    with open(fd,'wb') as f:
        f.write(base64.b64decode(contents))
    print("Stored Dicom file")
    image=preprocess_image(file_path,patient_name,inspection_code=inspection_code)
    # print(image)
    if file_type.lower() == "dcm":
        upload_image(file_path,inspection_code)
    # anomaly_image=preprocess_image(file_path)
    classes=prediction_scan(image[0],patient_name)
    # classes=prediction_scan(image)
    # anomaly_analysis=check_anomaly(anomaly_image)
    os.remove(file_path)
    return {name: classes}

if __name__ == '__main__':
    
    file_path='downloads/Final.DCM'
    scan_path="scans/dicom_files/Blaise_Papa.png"
    file_type="dcm"
    
    image=preprocess_image(scan_path,'Blaise_Papa',file_type="png")
    if file_type.lower() == "dcm":
        upload_image(file_path,12)
    # anomaly_image=preprocess_image(file_path)
    # print(image)
    # upload_image(file_path,982)
    # print(check_anomaly(anomaly_image))
    # prediction_scan(image[0],"Blaise_Papa")
    # print(classes)
    print("done")
