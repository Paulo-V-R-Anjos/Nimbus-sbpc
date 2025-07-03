# Create & activate conda env
conda create -n human_detection python=3.9 -y
conda activate human_detection

# Install PyTorch (with CUDA if available; adjust cudatoolkit version to match your GPU/driver)
conda install -c pytorch pytorch torchvision torchaudio cudatoolkit=11.7 -y

# Install other dependencies
conda install -c conda-forge opencv pillow numpy schedule -y

# Install YOLO package
pip install ultralytics
