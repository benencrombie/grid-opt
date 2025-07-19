from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from main.Numericals import SMROptimizer
from threading import Thread
import os

app = Flask(__name__, template_folder = "templates")
# SocketIO used to route and update matplotlib figures
socketio = SocketIO(app)

@app.route('/outputs/<path:filename>')
def serve_image(filename):
    return send_from_directory("outputs", filename)

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
def handle_start_run(data): 
    print('Started Handler')
    # Specify method chosen. Default to gradient descent if selection is invalid
    method = data.get('method', 'GD')
    max_iter = data.get('iterations', 100)
    optimizer = SMROptimizer(method = method, max_iter = max_iter)
    # Define internal method to run the optizer
    def run_optimizer():
        print('Started run')
        # The algorithms will yield the iteration and the filepath.
        for i, path_name in optimizer.gradientDescent():
            socketio.emit('update_plot', {'filename': '/outputs/' + path_name})
        print('Run done')
    Thread(target=run_optimizer).start()


if __name__ == "__main__":
    # Make outputs folder if not exist
    os.makedirs("outputs", exist_ok=True)
    # Run using socketio for dynamic plotting.
    socketio.run(app, debug=True)