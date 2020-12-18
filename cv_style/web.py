from flask import Flask, render_template
from flask import request
import os
from bs4 import BeautifulSoup as Soup
import datetime
import sys
sys.path.append("/VL/space/zhan1624/pytorch-CycleGAN-and-pix2pix")
from interface import cycle_GAN_interface

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './'

@app.route('/')
def hello_world():
    return render_template('example.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        style = request.form.get('styles')
        input_filename, output_filename = save_file(f, style)
        process_image(input_filename, output_filename, style)
        now = datetime.datetime.now()
        timeString = now.strftime("%Y-%m-%d %H:%M")
        templateData = {
        'time': timeString ,
        'input': input_filename,
        'output': output_filename,
        }
        return render_template('uploader.html', **templateData)
    return render_template('example.html')

def process_image(filename, output_filename, style):
    # read 'input.png' and save 'output.png'
    cycle_GAN_interface(filename, output_filename, style)
    return

def save_file(f, style):
    splits = f.filename.split('.')
    name = splits[0]
    suffix = splits[-1]
    assert suffix in ['jpg', 'png']
    filename = 'static/images/input_%s_%s.png' % (name, style)
    f.save(filename)
    output_filename = 'static/images/output_%s_%s.png' % (name, style)
    return filename, output_filename

if __name__ =='__main__':
    app.run()

