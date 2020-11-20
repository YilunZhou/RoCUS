from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pybulletgym_rocus'))

VERSION = 0.1

setup_py_dir = os.path.dirname(os.path.realpath(__file__))

need_files = []
datadir = "pybulletgym_rocus/envs/assets"

hh = setup_py_dir + "/" + datadir

for root, dirs, files in os.walk(hh):
    for fn in files:
        ext = os.path.splitext(fn)[1][1:]
        if ext and ext in 'png gif jpg urdf sdf obj mtl dae off stl STL xml '.split():
            fn = root + "/" + fn
            need_files.append(fn[1+len(hh):])

setup(name='pybulletgym_rocus',
      version=VERSION,
      packages=[package for package in find_packages()
                if package.startswith('pybulletgym_rocus')],
      zip_safe=False,
      install_requires=[
          'pybullet>=1.7.8',
      ],
      package_data={'pybulletgym_rocus': need_files},
)
