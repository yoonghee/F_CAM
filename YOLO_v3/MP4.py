import numpy as np
import cv2 
import os 

def frames_to_video(inputpath,outputpath,fps):
    image_array = []
    files = [f for f in os.listdir(inputpath) if isfile(join(inputpath, f))]
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        img = cv2.imread(inputpath + files[i])
        size =  (img.shape[1],img.shape[0])
        img = cv2.resize(img,size)
        image_array.append(img)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(outputpath,fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()

        if success:
            if count%76 != 0: # 19, 76
                pass
            cv2.imwrite(os.path.join(path_output_dir, '%d.jpg') % (count/76+1089), image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

video_to_frames('nayeon.mp4', 'content/sample_data_out/')

''' ~480 481~
inputpath = 'content/sample_data_frames/'
outpath =  'content/sample_data_frames/video.mp4'
fps = 24
frames_to_video(inputpath,outpath,fps)
'''

