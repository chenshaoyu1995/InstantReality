# Instant Reality: Gaze-Contingent Perceptual Optimization for 3D Virtual Reality Streaming

This branch demonstrates the neural acceleration under an edge-cloud setting.

This content of this repository is structured as:
- ``Glitter``: the C++ client
- ``NeuralAccelerator``: training and server code for the neural accelerator

## Getting Started

- Download the texture asset at https://www.turbosquid.com/3d-models/quantum-background-skybox-x/678381
- Copy the ``QuantumHighlands_Diffuse_E.bmp``, ``QuantumHighlands_Displacement_E.tif``, ``QuantumHighlands_Normal_E.bmp`` and all the sky textures to ``Glitter/Assets/terrain``
- Convert the ``QuantumHighlands_Displacement_E.tif`` to png format
- Install ``requirements.txt`` in ``NeuralAccelerator`` for the Python server
- Use cmake and Visual Studio 2019 to compile the client
- Start the Python server by running ``NeuralAccelerator/server.py``
- Set Glitter as the startup project in VS2019 and run the client
- Put on your HTC Vive Pro Eye headset and press key Q to start the streaming
