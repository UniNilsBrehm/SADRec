# Soundcard ADC Recoder (SADRec)

The  **SADRec** is a lightweight Python app for viewing and recording any signals connected to your soundcard microphone input. 
It is based on the following Python Packages:

 - PyAudio (https://people.csail.mit.edu/hubert/pyaudio/)
 - PyQt6 (https://pypi.org/project/PyQt6/)
 - PyQtGraph (https://www.pyqtgraph.org/)
 - Numpy
 - SciPy


## Installation

If you are using Anaconda, it is recommended to first create a new environment.

    conda create --name sadrec

Then install all packages:

    pip install numpy scipy pyaudio PyQt6 pyqtgraph

Finally, copy this repository to your hard drive. Navigate to the directory using your favorite terminal and run:

    python sadrec.py

## Features
- Live View - Audio Monitor  
- Store recording in a wav file  
- Low and High Pass Filter  
- Navigate the Viewer with shortcuts  
- Simple Sine Wave Stimulator (via Speaker)  
- Open and view wav files (e.g. your recordings)

## Navigation
There are several handy keyboard shortcuts for navigation:

Press "M" to mute and unmute audio monitor  
Press "R" to start and stop recording
Press "B" to reset the view
Press "C" to center the view (y-axis)
Press "T" and "Shift+T" to zoom the x-axis
Press "X" and "Shift+X" to zoom the y-axis
Press "Left" and "Right" to move along the x-axis
Press "Up" and "Down" to move along the y-axis


----------
Nils Brehm - 2024