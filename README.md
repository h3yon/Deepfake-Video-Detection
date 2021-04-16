# Deepfake Video Detection

We used DFDC, FaceForensics++, Celeb-DF datasets.
In this project, the performance on detection was compared and analyzed using color channels.
Used Models are AlexNet, DenseNet, InceptionResNetV2, MesoNet, ResNet152, VGG, Xception.

### Progress

![Progress](./progress.png)

### How to start?

1. Download Datasets

   ```
   1. celeb-DF: http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html
   2. DFDC: https://dfdc.ai/
   3. FaceForensics++: https://github.com/ondyari/FaceForensics
   ```

2. Download python code for converting videos to images

- We used some codes in ![Debanik/Detecting-Deepfakes](https://github.com/Debanik/Detecting-Deepfakes)
- To extract frames, used python codes are
  **extract_faces*.py, res10_300x300_ssd_iter_140000.caffemodel, search_videos_in_directory*.py**.
  Additionaly, We changed code search\*.py.

3. Download Requirements.txt and download requirements.

   ```
   pip install -r requirements.txt
   ```

4. Make up Directories Structures whatever you want.
   ```
   ㄴCD_RGB
   ㄴCD_Gray
   ㄴDFDC_RGB
   ㄴDFDC_Gray
   ㄴFF_RGB
    ㄴFF_Gray
    ㄴCeleb-DF-frames
   ㄴFaceForensics++-frames
   ㄴDFDC-frames
   ```
   <div></div>

### Preprocessing

```
deploy.prototxt.txt           real_test1
real_train                    extract_faces_keep_in_mem.py
real_train1                   extract_faces.py
requirements.txt              fake_test
res10_300x300_ssd_iter_140000.caffemodel
fake_test1                    fake_train
search_videos_in_directory_keep_in_mem.py
fake_train1                   search_videos_in_directory.py
real_test
```

extract*.py, res*.coffeemodel, search\_\*.py, deploy.proto\*.txt are from [Debanik/Detecting-Deepfakes](https://github.com/Debanik/Detecting-Deepfakes)

we changed search_videos\*.py
