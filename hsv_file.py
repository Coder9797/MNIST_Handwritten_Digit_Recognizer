{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3a194ce2-e062-497e-b939-66d5f7c3c674",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-02-16 22:36:51.021 Python[78612:7508259] +[IMKClient subclass]: chose IMKClient_Modern\n",
      "2025-02-16 22:36:51.021 Python[78612:7508259] +[IMKInputSession subclass]: chose IMKInputSession_Modern\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[35, 50, 50], [85, 255, 255]]\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "# Set video feed window size\n",
    "cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)\n",
    "cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)\n",
    "\n",
    "def nothing(x):\n",
    "    pass\n",
    "\n",
    "# Create trackbar window\n",
    "cv2.namedWindow('Trackbars')\n",
    "cv2.createTrackbar(\"L - H\", \"Trackbars\", 35, 179, nothing)  # Lower Hue for green\n",
    "cv2.createTrackbar(\"L - S\", \"Trackbars\", 50, 255, nothing)  # Lower Saturation\n",
    "cv2.createTrackbar(\"L - V\", \"Trackbars\", 50, 255, nothing)  # Lower Value\n",
    "cv2.createTrackbar(\"U - H\", \"Trackbars\", 85, 179, nothing)  # Upper Hue for green\n",
    "cv2.createTrackbar(\"U - S\", \"Trackbars\", 255, 255, nothing) # Upper Saturation\n",
    "cv2.createTrackbar(\"U - V\", \"Trackbars\", 255, 255, nothing) # Upper Value\n",
    "\n",
    "while True:\n",
    "    ret, frame = cap.read()\n",
    "    if not ret:\n",
    "        break\n",
    "\n",
    "    frame = cv2.flip(frame, 1)\n",
    "    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)\n",
    "\n",
    "    # Get trackbar positions\n",
    "    l_h = cv2.getTrackbarPos(\"L - H\", \"Trackbars\")\n",
    "    l_s = cv2.getTrackbarPos(\"L - S\", \"Trackbars\")\n",
    "    l_v = cv2.getTrackbarPos(\"L - V\", \"Trackbars\")\n",
    "    u_h = cv2.getTrackbarPos(\"U - H\", \"Trackbars\")\n",
    "    u_s = cv2.getTrackbarPos(\"U - S\", \"Trackbars\")\n",
    "    u_v = cv2.getTrackbarPos(\"U - V\", \"Trackbars\")\n",
    "\n",
    "    lower_range = np.array([l_h, l_s, l_v])\n",
    "    upper_range = np.array([u_h, u_s, u_v])\n",
    "\n",
    "    # Create mask for detecting green\n",
    "    mask = cv2.inRange(hsv, lower_range, upper_range)\n",
    "    res = cv2.bitwise_and(frame, frame, mask=mask)\n",
    "\n",
    "    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)\n",
    "    stacked = np.hstack((mask_3, frame, res))\n",
    "\n",
    "    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.6, fy=0.6))\n",
    "\n",
    "    key = cv2.waitKey(1)\n",
    "    if key == 27:  # Press 'ESC' to exit\n",
    "        break\n",
    "\n",
    "    if key == ord('s'):  # Press 's' to save the HSV values\n",
    "        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]\n",
    "        print(thearray)\n",
    "        np.save('hsv_value.npy', thearray)\n",
    "        break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "33ba929a-8680-4b93-aa15-f79c7a3ef51b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
