# Live2d face (Inspired by Facerig)

## Example
Myself trying to speak somthing.

https://youtu.be/UC2fJBYEUnI

## Device
- Somehow good computer
- Camera/Webcam (better webcam help)

## Installation
1. Download the Live2d native SDK from https://www.live2d.com/en/download/cubism-sdk/
2. Build the MSVC project followed the instruction for DX11 https://docs.live2d.com/cubism-sdk-tutorials/sample-build-directx11/
3. Download library `dlib` and `opencv (3.4.0)` then include the header, library, into the MSVC project (if not sure, please take a look on internet how to include those library into your project)
3. Replace/add the source code with the code in `src/` then try to build the solution. To gain performance, please change to `Release` mode
