import os
from process_music import main
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Define the folder where uploaded music files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # Check if the file has a filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Check if the file extension is allowed (e.g., mp3, wav)
        allowed_extensions = {'mp3', 'wav', 'ogg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file extension'})

        # Save the uploaded file to the uploads folder
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Process the uploaded music file (replace with your processing logic)
        processing_result = main(filename)

        os.remove(filename)

        # Return the processing result as JSON
        return jsonify({'result': processing_result})

    return render_template('index.html')

# def process_music_file(filename):
#     # Replace this function with your actual processing logic
#     # For example, you can use a music processing library to analyze or transform the music file
#     # Return the processing result
#     return 'Music file processed successfully'

if __name__ == '__main__':
    app.run(debug=True)