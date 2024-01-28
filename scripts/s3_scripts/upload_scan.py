# This script uplaods a scan file to the s3 bucket
try:
    import boto3 as boto
    import pandas as pd
    from dotenv import load_dotenv
    import os
    print("Successfully loaded Libraries")
except ModuleNotFoundError as mnf:
    print("Error Importing module: {}".fromat(mnf.name))
    print("Details: {}".fromat(mnf))
    exit(1)
    



load_dotenv()
print("Reading Environmental Variables")
aws_access_key=os.getenv("AWS_S3_PACS_ACCESS_KEY")
aws_secret_key=os.getenv("AWS_S3_PACS_SECRET_KEY")
if (aws_access_key == None) or (aws_secret_key == None):
    print("Failed to access AWS ACCESS KEYS Terminating Program")
    exit(1)
else:
    print("Successfully Retrieved AWS Access Key")

#create a boto session
print("Creating a boto session")
try:
    session=boto.session.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    print("Successfully created session")
except Exception as e:
    print("Error creating session")
    print("Error Details: {}".format(e))
    exit(1)

s3=session.client("s3")
#uploading a file
def upload_scan(file_name: str="../../data/testtext.txt",bucket_destination:str = "diagnosoft-pacs-scans" ):
    #first check to confirm that file extension is dcom
    if not file_name.lower().endswith('.dcom'):
        try:
            print("Uploading scan to bucket: %s" % bucket_destination)
            s3.upload_file(
                    Filename=file_name,
                    Bucket=bucket_destination,
                    Key="testdata2.txt")
            print("Successfully uploaded file: {}".format())
        except Exception as ie:
            print("Error while Uploading file data: ")
            print("Details: {}".format(ie))
            exit(1)
    else:
        print("Uploading file data failed")
        print("Details: The file isn't supported. Only supported documents are DCOM")
        exit(1)
        
        
def download_scan(bucket_name,file_path,local_path):
    # session=boto.cient('s3')
    try:
        s3.download_file(bucket_name,file_path,local_path)
        print(f"Image downloaded successfully to {local_path}")
    except Exception as e:
        print("Uploading file data failed")
        print("Details: {}".format(e))
        print("Details: The file isn't supported. Only supported documents are DCOM")
        exit(1)

download_scan("diagnosoft-pacs-scans","Final.DCM","../../downloads/Final.DCM")
        
        
# upload_scan()   

