{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9723fc01-e296-45cb-af31-e9234ba7e40f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-02-16 22:44:15.542 Python[78692:7512717] +[IMKClient subclass]: chose IMKClient_Modern\n",
      "2025-02-16 22:44:15.543 Python[78692:7512717] +[IMKInputSession subclass]: chose IMKInputSession_Modern\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Digit: 3\n",
      "Predicted Digit: 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-02-16 22:44:45.688 Python[78692:7512717] _TIPropertyValueIsValid called with 16 on nil context!\n",
      "2025-02-16 22:44:45.688 Python[78692:7512717] imkxpc_getApplicationProperty:reply: called with incorrect property value 16, bailing.\n",
      "2025-02-16 22:44:45.688 Python[78692:7512717] Text input context does not respond to _valueForTIProperty:\n",
      "2025-02-16 22:46:09.396 Python[78692:7512717] _TIPropertyValueIsValid called with 16 on nil context!\n",
      "2025-02-16 22:46:09.396 Python[78692:7512717] imkxpc_getApplicationProperty:reply: called with incorrect property value 16, bailing.\n",
      "2025-02-16 22:46:09.396 Python[78692:7512717] Text input context does not respond to _valueForTIProperty:\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Digit: 7\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import import_ipynb\n",
    "from Model import DigitCNN\n",
    "\n",
    "# Load the trained model\n",
    "model = DigitCNN()\n",
    "model.load_state_dict(torch.load(\"cnn_model_weights.pth\", map_location=torch.device('cpu')))\n",
    "model.eval()\n",
    "\n",
    "# OpenCV setup\n",
    "cap = cv2.VideoCapture(0)\n",
    "cap.set(3, 1280)\n",
    "cap.set(4, 720)\n",
    "canvas = np.zeros((720, 1280, 3), dtype=np.uint8)\n",
    "kernel = np.ones((5, 5), np.uint8)\n",
    "x1, y1 = 0, 0\n",
    "noise_thresh = 800\n",
    "\n",
    "while True:\n",
    "    _, frame = cap.read()\n",
    "    frame = cv2.flip(frame, 1)\n",
    "    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)\n",
    "    \n",
    "    # Load color range (assumes hsv_value.npy exists)\n",
    "    hsv_value = np.load('hsv_values.npy')\n",
    "    lower_range, upper_range = hsv_value[0], hsv_value[1]\n",
    "    \n",
    "    # Masking\n",
    "    mask = cv2.inRange(hsv, lower_range, upper_range)\n",
    "    mask = cv2.erode(mask, kernel, iterations=1)\n",
    "    mask = cv2.dilate(mask, kernel, iterations=2)\n",
    "    \n",
    "    # Contours detection\n",
    "    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
    "    \n",
    "    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:\n",
    "        c = max(contours, key=cv2.contourArea)\n",
    "        x2, y2, w, h = cv2.boundingRect(c)\n",
    "        if x1 == 0 and y1 == 0:\n",
    "            x1, y1 = x2, y2\n",
    "        else:\n",
    "            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 8)\n",
    "        x1, y1 = x2, y2\n",
    "    else:\n",
    "        x1, y1 = 0, 0\n",
    "    \n",
    "    # Add canvas to frame\n",
    "    frame = cv2.add(frame, canvas)\n",
    "    stacked = np.hstack((canvas, frame))\n",
    "    cv2.imshow('Live Writing', cv2.resize(stacked, None, fx=0.6, fy=0.6))\n",
    "    \n",
    "    # Key press actions\n",
    "    key = cv2.waitKey(1) & 0xFF\n",
    "    if key == 27:\n",
    "        break\n",
    "    elif key == ord('c'):\n",
    "        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)\n",
    "    elif key == ord('p'):\n",
    "        # Crop and preprocess digit\n",
    "        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)\n",
    "        _, thresh = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY)\n",
    "        x, y, w, h = cv2.boundingRect(thresh)\n",
    "        digit = thresh[y:y+h, x:x+w]\n",
    "        \n",
    "        # Center and resize the digit\n",
    "        h, w = digit.shape\n",
    "        pad_size = max(h, w)\n",
    "        padded_digit = np.zeros((pad_size, pad_size), dtype=np.uint8)\n",
    "        x_offset = (pad_size - w) // 2\n",
    "        y_offset = (pad_size - h) // 2\n",
    "        padded_digit[y_offset:y_offset+h, x_offset:x_offset+w] = digit\n",
    "        digit = cv2.resize(padded_digit, (28, 28))\n",
    "        \n",
    "        # Show the preprocessed input image\n",
    "        cv2.imshow('Model Input', digit)\n",
    "        cv2.waitKey(500)  # Display for 500ms\n",
    "        \n",
    "        # Normalize input\n",
    "        digit = digit.astype(np.float32) / 255.0\n",
    "        digit = torch.tensor(digit).unsqueeze(0).unsqueeze(0)\n",
    "        \n",
    "        # Predict digit\n",
    "        with torch.no_grad():\n",
    "            output = model(digit)\n",
    "            prediction = torch.argmax(output, dim=1).item()\n",
    "        \n",
    "        print(f\"Predicted Digit: {prediction}\")\n",
    "        cv2.putText(frame, f\"Predicted: {prediction}\", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
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
