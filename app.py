from flask import Flask, render_template_string, request, redirect, url_for, session
import os, traceback, datetime

app = Flask(__name__)
app.secret_key = "deepfake_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Safe import of explainability modules (dummy if missing)
# -----------------------------
def _dummy(module_name):
    def _f(path, **kwargs):
        return f"⚠ Module `{module_name}` not installed. Placeholder result for `{os.path.basename(path)}`.\n"
    return _f

try:
    from explainability.eye_blink_mismatch import main as analyze_eye
except Exception:
    analyze_eye = _dummy("eye_blink_detector")

try:
    from explainability.iris_alignment import main as analyze_iris
except Exception:
    analyze_iris = _dummy("iris_alignment")

try:
    from explainability.eyebrow_mismatch import main as analyze_eyebrow
except Exception:
    analyze_eyebrow = _dummy("eyebrow_mismatch")

try:
    from explainability.texture_analyzer import main as analyze_texture
except Exception:
    analyze_texture = _dummy("texture_analyzer")

try:
    from explainability.flicker_detection import main as analyze_flicker
except Exception:
    analyze_flicker = _dummy("flicker_detection")

try:
    from explainability.lip_sync_module import main as analyze_lip
except Exception:
    analyze_lip = _dummy("lip_sync_mismatch")

# -----------------------------
# In-memory user store and upload tracking
# -----------------------------
users = {}  # username -> {password, email}
uploads_per_user = {}  # username -> {date_str: count}

# -----------------------------
# Base CSS Styling
# -----------------------------
base_css = """
:root{
  --bg:#0b0c1a; --panel:#141522; --accent:#00eaff; --accent-2:#00b2cc; --muted:#a0a6b0;
  --glass:rgba(255,255,255,0.04); --radius:12px; --card-shadow:0 10px 30px rgba(0,0,0,0.6);
  --font-heading:"Orbitron", Arial, sans-serif; --font-body:"Roboto", Arial, sans-serif;
}
*{box-sizing:border-box} 
html,body{height:100%;margin:0} 
body{
  font-family:var(--font-body); 
  background: linear-gradient(180deg, #071021 0%, var(--bg) 100%);
  color:#fff; line-height:1.5;
}
.header{
  width:100%; background:var(--panel); padding:14px 20px; 
  display:flex; align-items:center; gap:20px;
  box-shadow:0 6px 18px rgba(0,0,0,0.6); 
  position:sticky; top:0; z-index:50;
}
.brand{
  font-size:26px; font-weight:700; 
  text-shadow:0 0 10px var(--accent); color:#fff;
  text-decoration:none;
}
.brand .oo{ color:var(--accent); }
.nav-links{
  display:flex; gap:20px; align-items:center;
}
.nav-links a{
  color:var(--muted); text-decoration:none; font-size:16px;
  transition: color 0.3s ease;
}
.nav-links a:hover{
  color:var(--accent);
}
.container{max-width:1100px; margin:36px auto; padding:0 20px;}
.center-card{
  background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: var(--radius); padding:28px; 
  box-shadow: var(--card-shadow);
}
.h1{
  font-size:38px; color:var(--accent); margin:0 0 16px 0; 
  text-shadow:0 0 10px rgba(0,234,255,0.06);
}
.lead{font-size:18px; color:var(--muted); margin-bottom:20px;}
input[type="text"], input[type="email"], input[type="password"], input[type="file"], select, textarea{
  width:100%; padding:14px 16px; margin:8px 0; border-radius:10px; 
  border:1px solid rgba(255,255,255,0.04);
  background: var(--glass); color:#fff; font-size:16px;
}
input[type="file"]{
  padding:10px 16px;
}
label{display:block;margin-top:12px;font-size:15px;color:var(--muted)}
button.primary, .btn-primary{
  display:inline-block; padding:14px 20px; background:var(--accent); 
  border:none; border-radius:12px; color:black; font-weight:700; 
  cursor:pointer; box-shadow:0 10px 30px rgba(0,234,255,0.06);
  text-decoration:none; font-size:16px;
}
button.primary:hover, .btn-primary:hover{background:var(--accent-2);}
.btn-secondary{
  display:inline-block; padding:14px 20px; 
  background:transparent; border:1px solid rgba(255,255,255,0.04);
  border-radius:12px; color:var(--accent); font-weight:700; 
  cursor:pointer; text-decoration:none; font-size:16px;
}
.btn-secondary:hover{
  background:rgba(0,234,255,0.1);
}
.upload-grid{display:grid; grid-template-columns:1fr 300px; gap:20px; align-items:start;}
@media(max-width:900px){.upload-grid{grid-template-columns:1fr}}
.analysis-card{ 
  background: rgba(255,255,255,0.02); padding:14px; 
  border-radius:10px; margin-bottom:12px;
}
.analysis-card h4{margin:0 0 6px 0; color:var(--accent); font-size:16px}
.result-pre{ 
  background: #07101a; border-radius:8px; padding:16px; 
  color:#e6f7ff; font-family:monospace; white-space:pre-wrap;
  max-height:520px; overflow:auto; 
  border:1px solid rgba(0,234,255,0.06)
}
.show-pass{
  margin-top:6px;font-size:14px;color:var(--muted); 
  cursor:pointer; user-select:none;
}
.social-btn{
  display:inline-block;margin:6px 6px 0 0;padding:10px 14px;
  border-radius:8px;font-weight:700;color:#fff;cursor:pointer;
}
.social-google{background:#db4437;} 
.social-facebook{background:#4267B2;}
.btn-group{
  display:flex; gap:16px; flex-wrap:wrap;
}
.logout-btn{
  color:var(--muted); text-decoration:none; font-size:14px;
  padding:8px 12px; border-radius:6px;
  background:rgba(255,255,255,0.04);
}
.logout-btn:hover{
  color:var(--accent); background:rgba(255,255,255,0.08);
}
"""

# -----------------------------
# HTML Templates
# -----------------------------

# Home Page Template
index_html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PROOF - Deepfake Detection System</title>
    <style>{{base_css}}</style>
</head>
<body>
    <div class="header">
        <a href="{{ url_for('index') }}" class="brand">Pr<span class="oo">oo</span>f</a>
        <div style="flex:1"></div>
        <div class="nav-links">
            <a href="{{ url_for('how_it_works') }}">How It Works</a>
            {% if session.user %}
                <a href="{{ url_for('upload') }}">Upload</a>
                <a href="{{ url_for('logout') }}" class="logout-btn">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('signup') }}">Sign Up</a>
            {% endif %}
        </div>
    </div>
    <div class="container">
        <div class="center-card">
            <h1 class="h1">Advanced Deepfake Detection</h1>
            <p class="lead">Protect yourself from AI-generated fake media. Our advanced system analyzes videos and images using multiple detection algorithms including eye blink patterns, facial alignment, texture analysis, and temporal inconsistencies.</p>
            
            <div class="analysis-card" style="margin:20px 0;">
                <h4>Why PROOF?</h4>
                <p style="color:var(--muted);font-size:16px;">
                    • <strong>Multi-Modal Analysis:</strong> 6 different detection algorithms<br>
                    • <strong>Real-time Processing:</strong> Get results in minutes<br>
                    • <strong>Explainable AI:</strong> Understand how decisions are made<br>
                    • <strong>High Accuracy:</strong> State-of-the-art detection methods
                </p>
            </div>

            {% if session.user %}
                <div class="btn-group">
                    <a href="{{ url_for('upload') }}" class="btn-primary">Start Analysis</a>
                </div>
                <p style="margin-top:12px;color:var(--muted);">Welcome back, {{ session.user }}!</p>
            {% else %}
                <div class="btn-group">
                    <a href="{{ url_for('signup') }}" class="btn-primary">Get Started</a>
                    <a href="{{ url_for('login') }}" class="btn-secondary">Login</a>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# Login Page Template
login_html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PROOF - Login</title>
    <style>{{base_css}}</style>
    <script>
    function togglePass(id){ 
        var x=document.getElementById(id); 
        x.type=(x.type==='password')?'text':'password'; 
    }
    </script>
</head>
<body>
    <div class="header">
        <a href="{{ url_for('index') }}" class="brand">Pr<span class="oo">oo</span>f</a>
        <div style="flex:1"></div>
        <div class="nav-links">
            <a href="{{ url_for('how_it_works') }}">How It Works</a>
            <a href="{{ url_for('signup') }}">Sign Up</a>
        </div>
    </div>
    <div class="container">
        <div class="center-card" style="max-width:420px;margin:0 auto">
            <h2 class="h1" style="font-size:26px">Welcome Back</h2>
            <p class="lead">Sign in to your account</p>
            <form method="post">
                <label>Username</label>
                <input type="text" name="username" required placeholder="Enter your username">
                
                <label>Password</label>
                <input type="password" name="password" id="pass" required placeholder="Enter your password">
                <div class="show-pass" onclick="togglePass('pass')">Show/Hide Password</div>
                
                <div style="margin-top:20px">
                    <button class="primary" type="submit">Sign In</button>
                </div>
            </form>
            
            <p style="margin-top:16px;font-size:15px;text-align:center;">
                Don't have an account? <a href="{{ url_for('signup') }}" style="color:var(--accent);text-decoration:none;">Create one here</a>
            </p>
            
            {% if error %}
                <div style="background:rgba(255,139,139,0.1);border:1px solid #ff8b8b;color:#ff8b8b;padding:12px;border-radius:8px;margin-top:16px;">
                    {{ error }}
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# Signup Page Template
signup_html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PROOF - Create Account</title>
    <style>{{base_css}}</style>
    <script>
    function togglePass(id){ 
        var x=document.getElementById(id); 
        x.type=(x.type==='password')?'text':'password'; 
    }
    </script>
</head>
<body>
    <div class="header">
        <a href="{{ url_for('index') }}" class="brand">Pr<span class="oo">oo</span>f</a>
        <div style="flex:1"></div>
        <div class="nav-links">
            <a href="{{ url_for('how_it_works') }}">How It Works</a>
            <a href="{{ url_for('login') }}">Login</a>
        </div>
    </div>
    <div class="container">
        <div class="center-card" style="max-width:480px;margin:0 auto">
            <h2 class="h1" style="font-size:26px">Create Your Account</h2>
            <p class="lead">Join PROOF to start detecting deepfakes</p>
            
            <form method="post">
                <label>Username</label>
                <input type="text" name="username" required placeholder="Choose a username">
                
                <label>Email Address</label>
                <input type="email" name="email" required placeholder="Enter your email">
                
                <label>Password</label>
                <input type="password" id="pass1" name="password" required placeholder="Create a password">
                <div class="show-pass" onclick="togglePass('pass1')">Show/Hide Password</div>
                
                <label>Confirm Password</label>
                <input type="password" id="pass2" name="confirm_password" required placeholder="Confirm your password">
                <div class="show-pass" onclick="togglePass('pass2')">Show/Hide Password</div>
                
                <div style="margin-top:20px">
                    <button class="primary" type="submit">Create Account</button>
                </div>
            </form>
            
            <div style="margin-top:20px;text-align:center;">
                <p style="font-size:15px;color:var(--muted);">Or sign up with:</p>
                <div>
                    <div class="social-btn social-google">Google</div>
                    <div class="social-btn social-facebook">Facebook</div>
                </div>
            </div>
            
            <p style="margin-top:16px;font-size:15px;text-align:center;">
                Already have an account? <a href="{{ url_for('login') }}" style="color:var(--accent);text-decoration:none;">Sign in here</a>
            </p>
            
            {% if error %}
                <div style="background:rgba(255,139,139,0.1);border:1px solid #ff8b8b;color:#ff8b8b;padding:12px;border-radius:8px;margin-top:16px;">
                    {{ error }}
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# How It Works Page Template
how_it_works_html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>How PROOF Works - Deepfake Detection</title>
    <style>{{base_css}}</style>
</head>
<body>
    <div class="header">
        <a href="{{ url_for('index') }}" class="brand">Pr<span class="oo">oo</span>f</a>
        <div style="flex:1"></div>
        <div class="nav-links">
            <a href="{{ url_for('index') }}">Home</a>
            {% if session.user %}
                <a href="{{ url_for('upload') }}">Upload</a>
                <a href="{{ url_for('logout') }}" class="logout-btn">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('signup') }}">Sign Up</a>
            {% endif %}
        </div>
    </div>
    <div class="container">
        <div class="center-card">
            <h2 class="h1">How PROOF Detects Deepfakes</h2>
            <p class="lead">Our multi-modal approach combines six advanced detection algorithms to identify AI-generated content with high accuracy.</p>
            
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-top:32px">
                <div class="analysis-card">
                    <h4>1. Upload Media</h4>
                    <p style="color:var(--muted);">Upload your suspicious video or image file. We support most common formats including MP4, AVI, JPG, PNG.</p>
                </div>
                
                <div class="analysis-card">
                    <h4>2. Preprocessing</h4>
                    <p style="color:var(--muted);">Extract frames, detect faces, and identify key facial landmarks for detailed analysis.</p>
                </div>
                
                <div class="analysis-card">
                    <h4>3. Eye Analysis</h4>
                    <p style="color:var(--muted);">Detect unnatural blinking patterns and iris alignment inconsistencies common in deepfakes.</p>
                </div>
                
                <div class="analysis-card">
                    <h4>4. Facial Features</h4>
                    <p style="color:var(--muted);">Analyze eyebrow movements, facial symmetry, and micro-expression authenticity.</p>
                </div>
                
                <div class="analysis-card">
                    <h4>5. Texture & Temporal</h4>
                    <p style="color:var(--muted);">Examine skin texture consistency and detect temporal flickering artifacts.</p>
                </div>
                
                <div class="analysis-card">
                    <h4>6. Audio Sync</h4>
                    <p style="color:var(--muted);">Verify lip-sync accuracy and detect audio-visual mismatches in video content.</p>
                </div>
            </div>
            
            <div class="analysis-card" style="margin-top:24px;">
                <h4>Comprehensive Report</h4>
                <p style="color:var(--muted);">Get detailed results from each detection module with explanations of suspicious patterns found. Our explainable AI approach helps you understand the reasoning behind each detection.</p>
            </div>
            
            <div style="margin-top:24px;text-align:center">
                {% if session.user %}
                    <a href="{{ url_for('upload') }}" class="btn-primary">Try Detection Now</a>
                {% else %}
                    <a href="{{ url_for('signup') }}" class="btn-primary">Get Started</a>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
"""

# Upload Page Template (After Login)
upload_html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PROOF - Upload & Analyze</title>
    <style>{{base_css}}</style>
</head>
<body>
    <div class="header">
        <a href="{{ url_for('index') }}" class="brand">Pr<span class="oo">oo</span>f</a>
        <div style="flex:1"></div>
        <div class="nav-links">
            <a href="{{ url_for('how_it_works') }}">How It Works</a>
            <a href="{{ url_for('logout') }}" class="logout-btn">Logout ({{ session.user }})</a>
        </div>
    </div>
    <div class="container">
        <div class="center-card">
            <h2 class="h1">Upload & Analyze Media</h2>
            <p class="lead">Upload a video or image file to run our comprehensive deepfake detection analysis.</p>
            
            <form method="post" enctype="multipart/form-data" class="upload-grid">
                <div>
                    <label>Select Media File</label>
                    <input type="file" name="file" accept="video/*,image/*" required>
                    
                    <div style="background:rgba(0,234,255,0.05);border:1px solid rgba(0,234,255,0.2);padding:12px;border-radius:8px;margin:12px 0;font-size:14px;">
                        <strong>Supported formats:</strong><br>
                        • Videos: MP4, AVI, MOV, MKV<br>
                        • Images: JPG, PNG, JPEG, BMP
                    </div>
                    
                    <div style="margin-top:20px">
                        <button class="primary" type="submit">Start Analysis</button>
                    </div>
                </div>
                
                <aside>
                    <div class="analysis-card">
                        <h4>Your Usage</h4>
                        <p style="color:var(--muted);font-size:14px">
                            <strong>User:</strong> {{ session['user'] }}<br>
                            <strong>Files analyzed today:</strong> {{ uploads_today }}<br>
                            <strong>Status:</strong> Active
                        </p>
                    </div>
                    
                    <div class="analysis-card">
                        <h4>Analysis Info</h4>
                        <p style="color:var(--muted);font-size:14px">
                            • Processing may take 1-5 minutes<br>
                            • All 6 detection modules will run<br>
                            • Results include detailed explanations<br>
                            • Files are processed securely
                        </p>
                    </div>
                    
                    <div class="analysis-card">
                        <h4>Privacy</h4>
                        <p style="color:var(--muted);font-size:14px">
                            Your uploaded files are processed locally and not stored permanently on our servers.
                        </p>
                    </div>
                </aside>
            </form>
        </div>
    </div>
</body>
</html>
"""

# Results Page Template
result_html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PROOF - Analysis Results</title>
    <style>{{base_css}}</style>
</head>
<body>
    <div class="header">
        <a href="{{ url_for('index') }}" class="brand">Pr<span class="oo">oo</span>f</a>
        <div style="flex:1"></div>
        <div class="nav-links">
            <a href="{{ url_for('upload') }}">Upload Another</a>
            <a href="{{ url_for('logout') }}" class="logout-btn">Logout</a>
        </div>
    </div>
    <div class="container">
        <div class="center-card" style="max-width:900px;margin:0 auto">
            <h2 class="h1">Analysis Results</h2>
            <p class="lead">Comprehensive deepfake detection analysis from all modules:</p>
            
            {% for module_name,text in outputs %}
            <div style="margin-bottom:24px">
                <div class="analysis-card">
                    <h4>{{ module_name }}</h4>
                    <div class="result-pre">{{ text }}</div>
                </div>
            </div>
            {% endfor %}
            
            <div style="margin-top:32px;text-align:center;">
                <a href="{{ url_for('upload') }}" class="btn-primary">Analyze Another File</a>
                <a href="{{ url_for('index') }}" class="btn-secondary">Back to Home</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

# -----------------------------
# Flask Routes
# -----------------------------

@app.route("/")
def index():
    return render_template_string(index_html, base_css=base_css)

@app.route("/login", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if username in users and users[username]["password"] == password:
            session["user"] = username
            return redirect(url_for("upload"))
        else:
            error = "Invalid username or password"
    
    return render_template_string(login_html, base_css=base_css, error=error)

@app.route("/signup", methods=["GET","POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        if not username or not email:
            error = "Username and email are required"
        elif password != confirm_password:
            error = "Passwords do not match"
        elif username in users:
            error = "Username already exists"
        elif len(password) < 6:
            error = "Password must be at least 6 characters"
        else:
            users[username] = {"password": password, "email": email}
            session["user"] = username
            return redirect(url_for("upload"))
    
    return render_template_string(signup_html, base_css=base_css, error=error)

@app.route("/how_it_works")
def how_it_works():
    return render_template_string(how_it_works_html, base_css=base_css)

@app.route("/upload", methods=["GET","POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))
    
    user = session["user"]
    today = datetime.date.today().isoformat()
    uploads_today = uploads_per_user.get(user, {}).get(today, 0)

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file selected", 400
        
        # Save uploaded file
        safe_name = file.filename.replace("/", "_").replace("\\", "_")
        path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(path)

        # Track uploads
        uploads_per_user.setdefault(user, {})
        uploads_per_user[user][today] = uploads_today + 1

        # Run all explainability modules
        modules = [
            ("Eye Blink Detection", analyze_eye),
            ("Iris Alignment Analysis", analyze_iris),
            ("Eyebrow Movement Analysis", analyze_eyebrow),
            ("Skin Texture Analysis", analyze_texture),
            ("Temporal Flicker Detection", analyze_flicker),
            ("Lip-Sync Analysis", analyze_lip)
        ]
        
        outputs = []
        for name, func in modules:
            try:
                result = func(path)
                if not isinstance(result, str):
                    result = str(result)
                outputs.append((name, result))
            except Exception as e:
                tb = traceback.format_exc()
                outputs.append((name, f"Analysis failed: {str(e)}\n\nTraceback:\n{tb}"))

        return render_template_string(result_html, base_css=base_css, outputs=outputs)

    return render_template_string(upload_html, base_css=base_css, uploads_today=uploads_today)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

# -----------------------------
# Run the Flask application
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
