#conda create -n searchr1_test python=3.9 -y
#conda init
#conda activate searchr1_test
#pip3 install vllm==0.6.3 
#pip install -e . 
#pip3 install flash-attn --no-build-isolation 
#pip install wandb 
#pip install nvidia-cublas-cu12==12.3.4.1



#conda create -n retriever python=3.10 -y
#conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
#pip uninstall -y scipy numpy scikit-learn
#pip install --upgrade pip
#pip install numpy scipy scikit-learn
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
pip install uvicorn fastapi
bash retrieval_launch.sh

