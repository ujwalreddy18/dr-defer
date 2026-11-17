# DR-Defer web app (3-model ensemble, 512 px)

Folder layout (create the `static` folder yourself):

```
dr-defer-app/
  best_512_s0.pt  best_512_s1.pt  best_512_s2.pt   <- trained weights
  engine.py
  server.py
  requirements.txt
  static/
    index.html
```

## Run
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python server.py
```
Your browser opens http://127.0.0.1:8000 by itself. Stop with Ctrl+C.

## Settings (top of engine.py)
- TEMPS: per-model temperatures fitted on EyePACS validation
- TE: extra temperature for the ensemble
- TAU = 0.5013 (defer when confidence < 0.4987, chosen on validation for 80% coverage)

Images are processed in memory and never saved.
For research use only. Not a medical device. It grades retinopathy severity in patients who
already have diabetes; it does not detect diabetes.
