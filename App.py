from flask import Flask, render_template, send_from_directory

app = Flask(__name__, template_folder = "templates")

@app.route('/assets/<path:filename>')
def serve_image(filename):
    return send_from_directory("assets", filename)

@app.route('/')
def home(): 
    return render_template('home.html')

@app.route('/tool')
def tool():
    return render_template('tool.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/profile')
def account():
    return render_template('account.html')

if __name__ == "__main__":
    app.run(debug=True)