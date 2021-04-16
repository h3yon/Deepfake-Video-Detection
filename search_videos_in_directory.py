
import os
import extract_faces


def search_videos(input_dir=os.path.dirname(os.path.realpath(__file__)), output_dir='~/'):
    try:
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/fake')
        os.mkdir(output_dir+'/real')
    except FileExistsError:
        pass
    threshold = int(input("Enter the number of images to be generated per video file: "))
    i = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
          try:
            if file.endswith('.mp4'):
                a = (root + '/' + str(file)).split('/')

                if 'Deepfakes' in a or 'FaceSwap' in a or 'Face2Face' in a or 'fake_train' in a or 'fake_test' or 'fake' or 'tr_fake' in a:
                    extract_faces.frame_capture(path=(root+'/'+str(file)), output_path=output_dir+'/fake', label='FAKE',
                                                start_counter=i, threshold=threshold)
                if 'real_train' in a or 'real_test' in a or 'tr_real' in a:
                    extract_faces.frame_capture(path=(root + '/' + str(file)), output_path=output_dir+'/real', label='REAL',
                                                start_counter=i, threshold=threshold)

                else:
                    extract_faces.frame_capture(path=(root + '/' + str(file)), output_path=output_dir+'/real', label='REAL',
                                                start_counter=i, threshold=threshold)

                i += threshold
          except:
                print('error') 


if __name__ == '__main__':
    input_direc = input("Enter the absolute path of the input directory: ")
    output_direc = input("Enter the absolute path of the output directory: ")
    search_videos(input_direc, output_direc)


