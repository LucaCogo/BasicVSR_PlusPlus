# Precompute Optical Flow Maps

import sys
import os

sys.path.append(os.getcwd)

import argparse
import glob
from PIL import Image
from mmedit.models.RAFT.raft import RAFT

from mmedit.models.RAFT.utils import InputPadder


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def compute_flows(args):

	model = torch.nn.DataParallel(RAFT(args))
	model.load_state_dict(torch.load(args.model, map_location= torch.device(DEVICE)))

	model = model.module()
	model.eval()

	input_folder = "lq_sequences_train"
	output_folder = "lq_sequences_train_of"

	imagespath = os.path.join(args.path, input_folder)
	targetpath = os.path.join(args.path, output_folder)

	if not os.path.exists(targetpath):
		os.mkdir(targetpath)

	with torch.no_grad():
		for d in os.listdir(imagespath):
			sequencepath = os.path.join(imagespath, d)
			print(f"Computing optical flow for sequence: {d}")

			image_filenames = glob.glob(os.path.join(sequencepath, '*.png')) + \
                 glob.glob(os.path.join(sequencepath, '*.jpg'))

			image_filenames = sorted(image_filenames)

            
			frames = []
			for imfile in image_filenames:
			        frames.append(load_image(imfile))
			frames = torch.cat(frames).to(DEVICE)

			padder = InputPadder(frames.shape)
			frames = padder.pad(frames)[0]

			frames_batch1 = frames[:-1]
			frames_batch2 = frames[1:]

			_, flow_out = model(frames_batch1, frames_batch2, iters=32, test_mode = True)


			filename = os.path.join(targetpath,d) + ".flo"
			fileObj = open(filename, 'wb')
			flow_out.cpu().numpy().tofile(fileObj)

			fileObj.close()




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help="restore checkpoint")
	parser.add_argument('--path', help="dataset for flow computation")
	parser.add_argument('--small', action='store_true', help='use small model')
	parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
	parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
	
	
	compute_flows(args)
	