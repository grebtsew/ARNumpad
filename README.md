# AR numpad
An Augumented Reality Numpad created to test the capabilities of MediaPipe library i python.

The idea for the project came after using the Hololens 2 device. Interactions are natural and really feels as a good AR implementation. This is a one-hand-at-a-time implementation. But can be further developed for handling several hands per camera. The interaction and AR technique works fine and we can really see how this could be useful in the future.

# Demonstration

![demo](./images/demo.gif)

# Getting Started

```
# Clone repo
git clone git@github.com:grebtsew/ARNumpad.git

# Install libraries
pip install -r requirements.txt

# Start implementation
python3 main.py
# or
python3 main_record.py
# or
python3 main_show_seperate_record.py
```

## Explaination
By using the MediaPipe Hands library for python3 we detect hands in realtime video streams. Landmarks on the hand acts as buttons for the numpad (1-9, 0-del-enter). Interact with the numpad ny pressing buttons with your thumb. The result is visualized on screen.

![landmarks](./images/hand_landmarks.png)

## Features
Trying to solve some main issues with datastreams from Mediapipe i have implemented:
* Hand open or close detection
* Right or Left hand detection
* Middlepoint of hand and size of hand
* Kalman filter to smooth transactions of landmarks and remove noice.
* Hand 3d angle detection
* Automatic calibration mode, for handling lights and angles
* Recording mode

## MediaPipe Hands
Mediapipe hands is a great library supporting many languages and gives the developer tools to create fast AR implementations with great accuracy and speed. Do checkout it out!

## Requirements
Webcam and install all python packages!

## License
[MIT](./LICENSE) License 