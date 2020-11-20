
import os
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model import BinClassifier

def train():
	device = ['cpu', 'cuda'][torch.cuda.is_available()]
	print(device)
	log_fn = f'data_and_model/train.log'
	model_fn = f'data_and_model/best.pt'
	if os.path.isfile(log_fn) or os.path.isfile(model_fn):
		input('Model file and/or log file exist. Press Enter to overwrite or Ctrl-C to abort...')
	log_file = open(log_fn, 'w')
	batchsize = 256
	max_epoch = 1000
	N_BINS = 100
	angle_res = 2 * np.pi / 100

	data = np.load('data_and_model/data.npz')['data'].astype('float32')[:, 0:19]
	X = data[:, :18]
	y = data[:, 18]
	y = y / 180 * np.pi
	y = ((y + np.pi) / angle_res).astype('int')
	ss = StandardScaler()
	X = ss.fit_transform(X)
	MEAN = ss.mean_
	STD = ss.scale_
	log_str = f'mean {list(MEAN.flat)}\nstandard deviation {list(STD.flat)}'
	print(log_str)
	log_file.write(log_str + '\n')
	data[:, :18] = X
	data[:, 18] = y
	np.random.shuffle(data)
	num_train = int(len(data) * 0.9)
	train = data[:num_train]
	test = data[num_train:]
	train_loader = DataLoader(train, batch_size=batchsize, shuffle=True)
	test_loader = DataLoader(test, batch_size=batchsize, shuffle=True)
	
	model = BinClassifier(18, N_BINS)
	model.to(device)
	loss = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters())
	scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True)
	best_acc = None

	for e in range(max_epoch):
		for d in tqdm(train_loader, ncols=78):
			X_batch = d[:, :18].to(device)
			y_batch = d[:, 18].long().to(device)
			optimizer.zero_grad()
			pred_logits = model(X_batch)
			loss_val = loss(pred_logits, y_batch)
			loss_val.backward()
			optimizer.step()
		with torch.no_grad():
			total_correct = 0
			total_num = 0
			for d in test_loader:
				X_batch = d[:, :18].to(device)
				y_batch = d[:, 18].numpy()
				pred_logits = model(X_batch).cpu().numpy()
				pred = pred_logits.argmax(axis=1)
				total_correct += (pred == y_batch).sum()
				total_num += len(d)
		test_acc = total_correct / total_num
		scheduler.step(test_acc)
		log_str = f'epoch: {e + 1}, test accuracy: {test_acc}'
		print(log_str)
		log_file.write(log_str + '\n')
		if best_acc is None or best_acc < test_acc:
			log_str = 'saved best model'
			print(log_str)
			log_file.write(log_str + '\n')
			best_acc = test_acc
			torch.save({'model': model.state_dict(), 'MEAN': MEAN, 'STD': STD, 'N_BINS': N_BINS}, model_fn)
		log_file.flush()

if __name__ == '__main__':
	train()
