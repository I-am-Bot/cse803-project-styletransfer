
from PIL import Image
import torchvision.transforms as transforms
from options.test_options import TestOptions
from models import create_model
from data import base_dataset
from util import util
import os


#model1:
opt = TestOptions().parse()  # get test options
opt.name = 'horse2zebra_pretrained'
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model1 = create_model(opt)      # create a model given opt.model and other options
model1.setup(opt) 

#model2:
#opt = TestOptions().parse()  # get test options
opt.name = 'monet2photo_pretrained'
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model2 = create_model(opt)      # create a model given opt.model and other options
model2.setup(opt) 



def cycle_GAN_interface(image_path, output_path, style):
    print(style)
    name = image_path.split('/')[-1][:-4] #output image name
    input_image = Image.open(image_path).convert('RGB') 
    input_nc = 3
    
    data = {}
    transform = base_dataset.get_transform(opt, grayscale=(input_nc == 1))
    visual_data = transform(input_image)
    visual_data = visual_data.unsqueeze(0)
    data['A'] = visual_data
    data['A_paths'] = image_path
    if style == 'horse2zebra':
        model = model1
    elif style == 'monet2photo':
        model = model2


    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    img_path = model.get_image_paths()     # get image paths
    
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        #save_path = os.path.join(output_image_dir, image_name)
        util.save_image(im, output_path)


if __name__ == "__main__":
   input_path = "input_images/00010.jpg"
   output_dir = "output_images"
   style = 'horse2zebra'
   cycle_GAN_interface(input_path, output_dir, style)
   print('aa')