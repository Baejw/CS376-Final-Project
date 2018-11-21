import numpy as np 
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Model
from auxf import read_data, get_train_test

saved_training_loss = []
saved_validation_loss = []
val_performance = []
train_performance = []


def train(model, device, X, y, optimizer, criterion, epoch):
	model.train()
	len_train = X.shape[0]
	batch_size = 64
	total_loss = 0
	tmp_performance = 0
	for idx in range(0, len_train, batch_size):
		optimizer.zero_grad()
		X_batch = X[idx: idx + batch_size]
		y_batch = y[idx: idx + batch_size]
		y_, _ = model(X_batch)
		loss = criterion(y_, y_batch)
		tmp_performance += torch.sum(torch.abs((y_batch - y_) / y_batch)).cpu().detach().numpy()
		total_loss += loss
		loss.backward()
		optimizer.step()
		if idx % 5000 == 0:
			print("Training epoch {} [{}, {}) Loss: {:.3}".format(epoch, idx, idx + batch_size, loss/batch_size))
	global saved_training_loss, train_performance
	total_loss = total_loss.cpu().detach().numpy()
	saved_training_loss.append(total_loss/len_train)
	p = 1 - (tmp_performance / len_train)
	print("Training Performance: {:.3}\n".format(p))
	train_performance.append(p)


def test(model, device, X, y, criterion, epoch):
	model.eval()
	len_train = X.shape[0]
	batch_size = 128
	total_loss = 0
	tmp_performance = 0
	with torch.no_grad():
		for idx in range(0, len_train, batch_size):
			X_batch = X[idx: idx + batch_size]
			y_batch = y[idx: idx + batch_size]
			y_, _ = model(X_batch)
			loss = criterion(y_, y_batch)
			tmp_performance += torch.sum(torch.abs((y_batch - y_) / y_batch)).cpu().numpy()
			total_loss += loss
			if (idx + batch_size >= len_train):
				print("\nValidation epoch {} [{}, {}) Loss: {:.3}".format(epoch, idx, idx + batch_size, loss/batch_size))
	global saved_validation_loss, val_performance
	total_loss = total_loss.cpu().numpy()
	saved_validation_loss.append(total_loss/len_train)
	p = 1 - (tmp_performance / len_train)
	print("Validation Performance: {:.3}\n".format(p))
	val_performance.append(p)



def main():
	data = read_data()
	X_train, X_test, y_train, y_test = get_train_test(data)
	X_train = X_train[:10000, :]
	y_train = y_train[:10000, :]
	if torch.cuda.is_available():
		print("USING CUDA")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Model().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0, amsgrad=True)
	optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1)
	criterion = nn.MSELoss().to(device)
	X_train = torch.from_numpy(X_train).float().to(device)
	X_test = torch.from_numpy(X_test).float().to(device)
	y_train = torch.from_numpy(y_train).float().to(device)
	y_test = torch.from_numpy(y_test).float().to(device)
	global saved_training_loss, saved_validation_loss, val_performance, train_performance
	for epoch in range(50):
		train(model, device, X_train, y_train, optimizer, criterion, epoch+1)
		test(model, device, X_test, y_test, criterion, epoch+1)
		optim_scheduler.step()
	plt.plot(saved_training_loss/max(saved_training_loss), label="Training Loss")
	plt.plot(saved_validation_loss/max(saved_validation_loss), label="Validation Loss")
	plt.plot(val_performance, label="Validation Performance")
	plt.plot(train_performance, label="Training Performance")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()
