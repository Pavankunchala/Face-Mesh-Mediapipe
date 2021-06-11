import streamlit as st

import mediapipe as mp
import cv2


import numpy as np
import tempfile
from PIL import Image

#Lets try to integrate streamlit and mediapipe

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = 'demo.mp4'
OUTM = 'output.mp4'
DEMO_IMAGE = 'demo.jpg'

def main():

    st.title('Face Mesh Application using MediaPipe')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


    st.sidebar.title('Face Mesh Application using MediaPipe')
    st.sidebar.subheader('Parameters')

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    #max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces',value =1,min_value= 1)

    st.markdown(' ## Output')
    stframe = st.empty()

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])

    

    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tfflie.name = DEMO_VIDEO
    
    else:
        tfflie.write(video_file_buffer.read())


    vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','8')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))


    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    

    with mp_face_mesh.FaceMesh(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence , 
    max_num_faces = max_faces) as face_mesh:




        while vid.isOpened():

            ret, frame = vid.read()

            if not ret:
                continue
                
            

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            

            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)


            out.write(frame)    
            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)

            #frame = image_resize(image = frame, width = 640)
            
            stframe.image(frame,channels = 'BGR',use_column_width=True)

    

    st.text('Video Processed')

    output_video = open('output1.webm','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)


        





    vid.release()
    out.release()




if __name__ == '__main__':
    main()
