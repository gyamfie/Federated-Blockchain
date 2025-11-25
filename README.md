# Federated-Blockchain
This repository contains the complete experimental setup used in the paper
“A Federated Blockchain Security for MEC-enabled IoT Networks in Industrial 5.0”
including hardware configuration, software stack, dataset, blockchain network, and training/evaluation scripts that reproduce the results in Section IV (Experimental Setup) and Section V (Results).
# Hardware Testbed (Real Deployment – Figure 9 of the paper)

IIoT End Devices ---> Raspberry Pi 4 Model B (4 GB RAM),12,Sensors (IIoT nodes)
Malicious IIoT Node (Attacker)---> Raspberry Pi 4 Model B,1,Spoofing & anomalous key requests
MEC Server ---> "Intel NUC11 i7-1185G7, 32 GB RAM, 1 TB NVMe SSD",1,Runs Hyperledger Fabric + DNN model
External Domain Node --->  Standard PC (connected via WAN),1,Simulates cross-domain communication

# Software Stack
Operating System ---> Ubuntu 22.04 LTS (all nodes)
Container Runtime ---> Docker + Docker Compose,24.x
Blockchain ---> Hyperledger Fabric (consortium mode),2.5.4
Smart Contracts ---> Chaincode written in Go,1.20
DNN Framework ---> PyTorch,2.0+
ML Feature Extraction ---> CICFlowMeter-v4 (for generating flow features
Python Environment---> Python 3.9+ (venv or conda)

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

# Citation
@ARTICLE{gyamfi2025federated,
  author    = {Eric Gyamfi and James Adu Ansere and Mohsin Kamal and ...},
  journal   = {IEEE Transactions on Industrial Informatics},
  title     = {A Federated Blockchain Security for MEC-enabled IoT Networks in Industrial 5.0},
  year      = {2025},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {},
  ISSN      = {},
  month     = {}
}
