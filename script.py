import cv2
import dlib

# Initialize dlib's face detector and load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the dog tongue image
dog_tongue_img = cv2.imread("dog_tongue.png", -1) # -1 for alpha channel (transparency)

# Function to overlay the tongue on the image
def overlay_tongue(image, tongue, position, width, height):
    tongue = cv2.resize(tongue, (width, height))
    for i in range(height):
        for j in range(width):
            if tongue[i, j][3] != 0: # Checking the alpha channel
                image[position[1] + i, position[0] + j] = tongue[i, j][:3]
    return image

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Assuming the mouth is around landmarks 48 to 68
        mouth_center = (landmarks.part(66).x, landmarks.part(66).y)
        tongue_width = landmarks.part(54).x - landmarks.part(48).x
        tongue_height = tongue_width

        frame = overlay_tongue(frame, dog_tongue_img, mouth_center, tongue_width, tongue_height)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()
