# TTNet-Tensorflow
Unofficial Tensorflow Implementation of TTNet: Real-time temporal and spatial video analysis of table tennis

Does not work?? Looked over research paper, other implementations and the model doesn't predict well at all? Maybe someone could point this out idk.

Work in progress, things that need to be done:
- Custom training loop (Complete)
- Custom metrics (Started)
- Dataset creation (Complete)

- Testing Procedure (Started)
- Fix the Events Predictions
- Convert Pixel Locations to useable numbers
- Live Capture and Utilization of Model
- GUI??

Notes:
- When ball leaves the frame retain previous prediction. Do not swtich to (0, 0) prediction.

