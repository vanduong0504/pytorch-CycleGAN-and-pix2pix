import os
from options.test_options import TestOptions
from data import create_dataset, create_dataloader
from models import create_model
from util.visualizer import save_images
from util import html
import tqdm
import copy
import torch.nn as nn
import torch 
from torchprofile import profile_macs
import time
from metric import create_metric_models, get_fid
import numpy as np

def profile(model, config=None, verbose=False):
    netG = model.netG
    if isinstance(netG, nn.DataParallel):
        netG = netG.module
    if config is not None:
        netG.configs = config
    with torch.no_grad():
        macs = profile_macs(netG, (model.real[:1],))
    params = 0
    for p in netG.parameters():
        params += p.numel()
    if verbose:
        print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)
    return macs, params

def test(model, config=None):
    with torch.no_grad():
        model.forward(config)

def evaluate(model,dataloader):
    fakes = []
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        model.set_input(data)
        if i == 0:
            macs, params = profile(model)
        model.test()
        visuals = model.get_current_visuals()
        generated = visuals['fake'].cpu()
        fakes.append(generated)
    # Test the latency of the model
    for i in range(100):
        model.test()
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(100):
        model.test()
        torch.cuda.synchronize()
    cost = time.time()-start_time
    latency = cost/100
    return macs, params, latency, fakes

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataloader = create_dataloader(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    macs_full, params_full, latency_full, fakes_full = evaluate(model,dataloader)
    print('Full: %.3fG MACs\t%.3fM Params\t%.5fs Latency' % 
    (macs_full/1e9, params_full/1e6, latency_full))

    device = copy.deepcopy(model.device)
    del model
    torch.cuda.empty_cache()
    inception_model = create_metric_models(opt, device)
    if inception_model is not None:
      npz = np.load(opt.real_stat_path)
      fid = get_fid(fakes_full, inception_model, npz, device, opt.batch_size)
      print('fid score: %.2f' % fid, flush=True)

    
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # if opt.eval:
    #     model.eval()
    # for i, data in enumerate(dataset):
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.test()           # run inference
    #     visuals = model.get_current_visuals()  # get image results
    #     img_path = model.get_image_paths()     # get image paths
    #     if i % 5 == 0:  # save images to an HTML file
    #         print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()  # save the HTML
