from flask import Flask, render_template, send_from_directory

app = Flask(__name__, template_folder = "templates")

@app.route('/assets/<path:filename>')
def serve_image(filename):
    return send_from_directory("assets", filename)


@app.route('/')
def home(): 
    image_url = "/assets/base_zipcode.png"
    return render_template('index.html', image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)