# Federated-Blockchain
This repository contains the complete experimental setup used in the paper
“A Federated Blockchain Security for MEC-enabled IoT Networks in Industrial 5.0”
including hardware configuration, software stack, dataset, blockchain network, and training/evaluation scripts that reproduce the results in Section IV (Experimental Setup) and Section V (Results).
# Hardware Testbed

1. IIoT End Devices ---> Raspberry Pi 4 Model B (4 GB RAM),12,Sensors (IIoT nodes)
2. Malicious IIoT Node (Attacker)---> Raspberry Pi 4 Model B,1,Spoofing & anomalous key requests
3. MEC Server ---> "Intel NUC11 i7-1185G7, 32 GB RAM, 1 TB NVMe SSD",1,Runs Hyperledger Fabric + DNN model
4. External Domain Node --->  Standard PC (connected via WAN),1,Simulates cross-domain communication

# Software Stack
1. Operating System ---> Ubuntu 22.04 LTS (all nodes)
2. Container Runtime ---> Docker + Docker Compose,24.x
3. Blockchain ---> Hyperledger Fabric (consortium mode),2.5.4
4. Smart Contracts ---> Chaincode written in Go,1.20
5. DNN Framework ---> PyTorch,2.0+
6. ML Feature Extraction ---> CICFlowMeter-v4 (for generating flow features
7. Python Environment---> Python 3.9+ (venv or conda)

# How to Deploy (Updated for Python Chaincode)
# 1. Package the chaincode
cd blockchain/chaincode

tar cfz code.tar.gz fbs_iot.py
tar cfz fbs_iot.tgz metadata.json code.tar.gz

# 2. Install on peer
peer chaincode install -n fbsiot -v 1.0 -p fbs_iot.tgz -l python

# 3. Instantiate
peer chaincode instantiate -n fbsiot -v 1.0 -C mychannel \
    -c '{"Args":[]}' --collections-config ./collections.json

# How to use the code:
1. Train the ML model using the New_Dataset and the
2. Save the ML trained code as s Pickle file (. pkl) and deploy it to the server
3. verify the performance using the test dataset extracted from the New_Dataset.csv file.
4. Setup the Hyperledger fibric and run the
5. Setup a rest api for the ML and the hyperledger
6. Deploy Experiment_sim.py file to the IoT device and Run the simulation  


