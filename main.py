<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>COVID X-ray Classification (MobileNetV2)</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f6f7fb; margin:0; }
    .wrap { max-width: 980px; margin: 24px auto; padding: 0 16px; }
    .card { background:#fff; border:1px solid #e6e8ef; border-radius:14px; padding:18px; box-shadow:0 8px 20px rgba(0,0,0,.05); }
    .header { display:flex; align-items:center; justify-content:space-between; gap:14px; border-bottom:1px solid #eee; padding-bottom:14px; margin-bottom:14px; }
    .left { text-align:left; font-size:12px; line-height:1.35; }
    .mid { text-align:center; }
    .mid img { height:70px; }
    .right { text-align:right; font-size:12px; line-height:1.35; }
    h1 { margin:10px 0 0; font-size:20px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
    .btn { background:#111; color:#fff; border:0; padding:10px 14px; border-radius:10px; cursor:pointer; }
    .btn2 { background:#2d6cdf; }
    .box { border:1px dashed #cfd6ea; padding:12px; border-radius:12px; background:#fbfcff; }
    img.preview { width:100%; max-height:420px; object-fit:contain; border-radius:12px; border:1px solid #e6e8ef; background:#fff; }
    .kv { font-size:14px; line-height:1.6; }
    .muted { color:#666; font-size:12px; }
    @media (max-width: 860px){ .grid{ grid-template-columns:1fr; } .right,.left{ display:none; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <div class="left">
          <div>Republic of Iraq</div>
          <div>Ministry of Higher Education and Scientific Research</div>
          <div>University of Alkafeel</div>
          <div>College of Health & Medical Technology</div>
          <div>Dept. of Radiology Techniques</div>
        </div>

        <div class="mid">
          <img src="/static/logo.png" alt="Logo">
          <h1>COVID X-ray Classification (MobileNetV2)</h1>
          <div class="muted">Upload an X-ray image and get prediction + PDF report.</div>
        </div>

        <div class="right">
          <div><b>Stage:</b> Fourth Year</div>
          <div><b>Academic Year:</b> 2025–2026</div>
        </div>
      </div>

      <div class="grid">
        <div class="box">
          <form action="/ui/predict" method="post" enctype="multipart/form-data">
            <div class="muted">X-ray image</div>
            <input type="file" name="file" accept="image/*" required />
            <div style="height:10px"></div>

            <div class="muted">Threshold (default 0.5)</div>
            <input type="number" step="0.01" min="0" max="1" name="threshold" value="{{ threshold if threshold is not none else 0.5 }}" />

            <div style="height:14px"></div>
            <button class="btn" type="submit">Diagnose</button>
          </form>

          <div style="height:12px"></div>

          <form action="/report" method="post" enctype="multipart/form-data">
            <div class="muted">Download PDF report (re-upload same image)</div>
            <input type="file" name="file" accept="image/*" required />
            <input type="hidden" name="threshold" value="{{ threshold if threshold is not none else 0.5 }}">
            <div style="height:10px"></div>
            <button class="btn btn2" type="submit">Download PDF</button>
          </form>
        </div>

        <div class="box">
          {% if img_preview %}
            <div class="muted">Uploaded image preview</div>
            <div style="height:8px"></div>
            <img class="preview" src="{{ img_preview }}" />
          {% else %}
            <div class="muted">Image preview will appear here after diagnosis.</div>
          {% endif %}

          {% if result %}
            <div style="height:12px"></div>
            <div class="kv">
              <div><b>Prediction:</b> {{ result.prediction }}</div>
              <div><b>Confidence:</b> {{ result.confidence }}%</div>
              <div><b>Raw Sigmoid P(label=1):</b> {{ result.prob_label1 }}</div>
              <div><b>Threshold:</b> {{ result.threshold }}</div>
              <div class="muted">label0={{ result.label0 }}, label1={{ result.label1 }}</div>
            </div>
          {% endif %}
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="muted">
        Disclaimer: For educational/research use only. Not a standalone medical diagnosis.
      </div>
    </div>
  </div>
</body>
</html>
