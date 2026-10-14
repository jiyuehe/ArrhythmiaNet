tmux  
==============================  
It manages multiple terminal sessions and keeps them running after SSH disconnects.  
To start it: tmux 
To view it: tmux attach -t 0  

To run the U-Net  
==============================  
```bash
docker run --gpus all -it \
    -v /home/j/Desktop/ssd/git:/workspace \
    -v /home/j/Desktop/hdd/share_folder:/home/j/Desktop/hdd/share_folder \
    -v /home/j/Documents/box/carto3_files/data_npz:/home/j/Documents/box/carto3_files/data_npz \
    -p 8888:8888 \
    my_minkowski_image:latest /bin/bash
```
```bash
python3 ArrhythmiaNet/main_u_net.py
```

To exit docker, command line
```bash
exit
```

Install Minkowski Engine Docker  
==============================  
[Install standard Docker Engine in Ubuntu]  
sudo apt update  
sudo apt install -y docker.io  
sudo systemctl enable docker --now  
sudo usermod -aG docker $USER  

Restart Ubuntu  

[Install NVIDIA Container Toolkit]  
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

[Install Minkowski Engine]  
docker pull chrischoy/minkowski_engine:latest

[Run the Docker container (and install Python libraries as needed)]  
docker run --gpus all -it \
    -v /home/j/Desktop/ssd/git/heart_project:/workspace \
    -v /home/j/Desktop/hdd:/data \
    -p 8888:8888 \
    chrischoy/minkowski_engine:latest /bin/bash

pip install matplotlib

apt update
apt install -y graphviz
pip install graphviz

[Save the installed libraries in the Docker container]  
Run this outside the container (in another terminal):
docker commit d6da8c95d86f my_minkowski_image
d6da8c95d86f is from the root@d6da8c95d86f:/workspace#
it's different each time

Misc notes  
==============================  
The minkowski engine docker container is Ubuntu 18.  

3D data dimensions (t, depth, height, width).  

If docker is running, command line to kill all docker containers:  
docker stop $(docker ps -q)  

