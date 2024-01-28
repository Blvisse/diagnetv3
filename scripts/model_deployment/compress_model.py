try:    
    import zipfile
    import zlib
    import os
    print("Successfully Imported Libraries")
    

except ImportError as ie:
    print("Failed to import modules")
    print("Details: {}".format(ie))
    exit(1)
    
# def compress_file(input_file,output_zip):
#     with zipfile.ZipFile(output_zip,'w',zipfile.ZIP_DEFLATED) as zipf:
#         zipf.write(input_file)
        
# input_file="../../modelz/best.hdf5"
# output_file='../../modelz/compressedbest.zip'

# compress_file(input_file, output_file)


def compress_to_target_size(input_file, output_file, target_size_ratio):
    with open(input_file, 'rb') as f:
        data = f.read()

    original_size = len(data)
    target_size = original_size * target_size_ratio

    compression_level = 1
    compressed_data = zlib.compress(data, compression_level)

    while len(compressed_data) > target_size:
        compression_level += 1
        compressed_data = zlib.compress(data, compression_level)

    with open(output_file, 'wb') as f:
        f.write(compressed_data)

    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {len(compressed_data)} bytes")
    print(f"Compression level: {compression_level}")

# Example usage:
input_file="../../modelz/best.hdf5"
output_file='../../modelz/compressedbest.zip'
target_size_ratio = 1/2  # Compress to one-third of the original size
compress_to_target_size(input_file, output_file, target_size_ratio)

