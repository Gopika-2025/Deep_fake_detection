from flask import render_template, request, redirect, url_for

def init_routes(app):

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            # dummy login validation
            if username == "admin" and password == "admin":
                return redirect(url_for('upload'))
            else:
                error = "Invalid username or password"
        return render_template("login.html", error=error)

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        error = None
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            # implement your DB logic here
            return redirect(url_for('login'))
        return render_template("signup.html", error=error)

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        result = None
        if request.method == "POST":
            file = request.files.get("file")
            analysis_type = request.form.get("analysis_type")
            # here you can call your deepfake detection function based on analysis_type
            result = f"Analysis done for {analysis_type}"
        return render_template("upload.html", result=result)

    @app.route("/how_it_works")
    def how_it_works():
        return render_template("how_it_works.html")

    @app.route("/feedback", methods=["GET", "POST"])
    def feedback():
        if request.method == "POST":
            name = request.form.get("name")
            message = request.form.get("message")
            # store feedback in DB or file
            return "Feedback submitted. Thank you!"
        return render_template("feedback.html")

    @app.route("/result")
    def result():
        return render_template("result.html")
