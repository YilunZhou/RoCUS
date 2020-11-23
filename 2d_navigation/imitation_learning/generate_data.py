
import os, sys
from tqdm import trange
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(dir_path, '..'))
sys.path.append(root_dir)
from environment import RBFArena
from rapidly_exploring_random_tree.rrt import RRT

def smooth(arena, xs, ys, niter=1):
	xs = list(np.array(xs).flat)
	ys = list(np.array(ys).flat)
	N = len(xs)
	for _ in range(niter):
		for i in range(1, N-1):
			xp = (xs[i - 1] + xs[i + 1]) / 2
			yp = (ys[i - 1] + ys[i + 1]) / 2
			if arena.gamma([xp, yp]) > 1.2:
				xs[i] = xp
				ys[i] = yp
	return xs, ys

def generate_single(arena, rrt):
	xys = None
	while xys is None:
		arena.reset()
		xys = rrt.get_path(arena)
	xys = rrt.increase_resolution(xys)
	xs, ys = zip(*xys)
	xs, ys = smooth(arena, xs, ys, niter=20)
	return xs, ys

def generate_data(fn, N):
	arena = RBFArena()
	rrt = RRT(control_res=0.2)
	if os.path.isfile(fn):
		input(f'Data file {fn} already exists! Press Enter to overwrite, or Ctrl-C to abort... ')
	all_data = []
	for _ in trange(N):
		xs, ys = generate_single(arena, rrt)
		dir_x = np.array(xs[1:]) - xs[:-1]
		dir_y = np.array(ys[1:]) - ys[:-1]
		angle = np.arctan2(dir_y, dir_x) * 180 / np.pi
		lidar = np.array([arena.lidar(pos) for pos in zip(xs[:-1], ys[:-1])])
		for x, y, lid, ang in zip(xs[:-1], ys[:-1], lidar, angle):
			data = np.concatenate([[x, y], lid, [ang], arena.points.flatten()])
			all_data.append(data)
	all_data = np.array(all_data).astype('float32')
	np.savez_compressed(fn, data=all_data)

if __name__ == '__main__':
	generate_data('data_and_model/data.npz', 50000)
