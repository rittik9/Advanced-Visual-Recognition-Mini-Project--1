import cv2
import numpy as np

# Load the pre-trained face and nose detection models (Haar Cascade)
face_cascade = cv2.CascadeClassifier('C:/Users/prith/PycharmProjects/Mini_Project_Task2/haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('C:/Users/prith/PycharmProjects/Mini_Project_Task2/haarcascade_mcs_nose.xml')

# Load the nose ring image (make sure it doesn't have an alpha channel)
nose_ring_img = cv2.imread('C:/Users/prith/PycharmProjects/Mini_Project_Task2/nose.png')

# Define the resizing factor (adjust as needed)
resize_factor = 1

# Open the video capture (you can replace 0 with the path to your video file)
cap = cv2.VideoCapture(0)

# Get the video's frames per second (fps) and frame dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()  # Read a frame from the video

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Region of interest (ROI) for face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect noses in the face region
        noses = nose_cascade.detectMultiScale(roi_gray)

        # If at least one nose is detected, take the first one and break the loop
        if len(noses) > 0:
            nx, ny, nw, nh = noses[0]
            nw = int(0.5 * nw)
            nh = int(0.5 * nh)
            nx = nx + 35
            ny = ny + 4
            # Resize the nose ring image to fit the detected nose (smaller size)
            nose_ring_resized = cv2.resize(nose_ring_img, (int(nw * resize_factor), int(nh * resize_factor)))

            # Create a mask for the white pixels in the nose ring image
            white_mask = np.all(nose_ring_resized == [255, 255, 255], axis=-1)

            # Invert the white mask to make white pixels transparent
            transparent_mask = np.logical_not(white_mask)

            # Calculate the position for placing the nose ring within the face ROI
            nose_ring_x = x + nx
            nose_ring_y = y + ny

            # Combine the nose ring with the frame, making white pixels transparent
            frame[nose_ring_y:nose_ring_y + nh, nose_ring_x:nose_ring_x + nw][transparent_mask] = nose_ring_resized[transparent_mask]

    # Write the frame with the detected faces and noses to the output video
    out.write(frame)

    cv2.imshow('Face and Nose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
