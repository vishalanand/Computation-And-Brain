#First time execution start
python3 -m venv env
pip install --upgrade pip
source env/bin/activate
pip install -r requirements.txt
deactivate
#First time execution end

#Each time code is revisited start
source activate env/bin
#conda install pytorch torchvision -c pytorch
#pip3 install torch torchvision
pip install -r requirements.txt
deactivate
#Each time code is revisited end

