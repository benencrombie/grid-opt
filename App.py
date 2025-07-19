from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from main.Numericals import SMROptimizer

app = Flask(__name__, template_folder = "templates")
# SocketIO used to route and update matplotlib figures
socketio = SocketIO(app)

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

@socketio.on('start_run')
def handle_start_algo(data): 
    # Specify method chosen. Default to gradient descent if selection is invalid
    method = data.get('method', 'GD')
    optimizer = SMROptimizer(method = method)
    # The algorithms will yield the iteration and the filepath.
    for i, path_name in optimizer.gradientDescent():
        socketio.emit('update_plot', {'filename': '/' + path_name})

if __name__ == "__main__":
    app.run(debug=True)