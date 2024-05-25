import cv2
import time
import os
import Rastreamento as htm

# Function to set dynamic resolution
def set_dynamic_resolution(cap):
    # Get the default camera resolution
    wCam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hCam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the current window size
    window_name = 'Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, wCam, hCam)

    # Get the new size of the window
    window_width = cv2.getWindowImageRect(window_name)[2]
    window_height = cv2.getWindowImageRect(window_name)[3]

    # Set the new resolution to the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

    return window_name

# Função para aplicar filtros
def apply_filter(img, totalFingers):
    if totalFingers == 1:
        # Escala de cinza
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif totalFingers == 2:
        # Escala de vermelho
        img[:, :, 1] = 0
        img[:, :, 2] = 0
    elif totalFingers == 3:
        # Escala de verde
        img[:, :, 0] = 0
        img[:, :, 2] = 0
    elif totalFingers == 4:
        # Escala de azul
        img[:, :, 0] = 0
        img[:, :, 1] = 0
    elif totalFingers == 5:
        # Inversão de cores
        img = cv2.bitwise_not(img)
    return img

# Carregar imagens de sobreposição e redimensioná-las
folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # Resize the overlay images to fit a small part of the window
    image_resized = cv2.resize(image, (100, 100))  # Adjust size as needed
    overlayList.append(image_resized)
print(len(overlayList))

# Configurações iniciais da câmera
cap = cv2.VideoCapture(0)
window_name = set_dynamic_resolution(cap)

# Inicialização do detector de mãos
pTime = 0
detector = htm.HandDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    side = detector.handSide()

    if len(lmList) != 0:
        fingers = []
        # Polegar
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] and side[0] == 'Right':
            fingers.append(1)
        elif lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] and side[0] == 'Left':
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 dedos
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers, side)

        # Aplicar filtro conforme o número de dedos levantados
        img = apply_filter(img, totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape
        overlay_img_resized = overlayList[totalFingers - 1]
        img[0:h, 0:w] = overlay_img_resized
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow(window_name, img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
