# chaincode/fbs_iot.py
# Python Hyperledger Fabric Chaincode
# Implements Federated Blockchain Security (FBS) for MEC-enabled IIoT
# Fully compatible with paper's testbed (Raspberry Pi + MEC + Hyperledger Fabric)

import time
import json
import uuid
from typing import Dict, Any
from fabric_chaincode_shim import ChaincodeStub, Chaincode

class FBSSmartContract(Chaincode):
    # Constants from the paper
    TRUST_THRESHOLD = 0.85  # DNN score ≥ 0.85 → approve key request

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # SubmitKeyRequest – Called by IIoT device or MEC gateway
    # ------------------------------------------------------------------
    def submit_key_request(self, stub: ChaincodeStub, args: list) -> Dict[str, Any]:
        if len(args) != 4:
            return self._error_response("Incorrect number of arguments. Expecting 4")

        device_id = args[0]
        public_key_pem = args[1]
        cluster_id = args[2]
        try:
            dnn_score = float(args[3])
        except ValueError:
            return self._error_response("DNN score must be a float")

        if not (0.0 <= dnn_score <= 1.0):
            return self._error_response("DNN score must be between 0.0 and 1.0")

        # Round to 4 decimals (as in paper)
        dnn_score = round(dnn_score, 4)
        approved = dnn_score >= self.TRUST_THRESHOLD

        request_id = f"REQ-{int(time.time() * 1000000)}-{uuid.uuid4().hex[:8]}"
        timestamp = int(time.time())

        key_request = {
            "requestID": request_id,
            "deviceID": device_id,
            "publicKey": public_key_pem,
            "clusterID": cluster_id,
            "dnnScore": dnn_score,
            "timestamp": timestamp,
            "approved": approved,
            "approver": "MEC-server-01",
            "txID": stub.get_tx_id(),
        }

        # Save key request
        stub.put_state(request_id, json.dumps(key_request).encode('utf-8'))

        # Emit event
        event_name = "KeyApproved" if approved else "KeyRejected"
        event_payload = json.dumps({
            "requestID": request_id,
            "deviceID": device_id,
            "approved": approved,
            "dnnScore": dnn_score
        }).encode('utf-8')
        stub.set_event(event_name, event_payload)

        # If approved → update device record (onboarding or key rotation)
        if approved:
            self._update_device_after_approval(stub, device_id, public_key_pem, dnn_score, cluster_id)

        return self._success_response({
            "requestID": request_id,
            "approved": approved,
            "dnnScore": dnn_score,
            "message": "Key request processed successfully"
        })

    # ------------------------------------------------------------------
    # Helper: Update device state after successful DNN verification
    # ------------------------------------------------------------------
    def _update_device_after_approval(self, stub: ChaincodeStub, device_id: str,
                                      public_key: str, trust_score: float, cluster_id: str):
        device_key = f"DEVICE-{device_id}"
        device_data = stub.get_state(device_key)

        if device_data:
            device = json.loads(device_data)
            device["publicKey"] = public_key
            device["keyVersion"] += 1
            device["trustScore"] = trust_score
            device["status"] = "ACTIVE"
        else:
            # New device onboarding
            device = {
                "deviceID": device_id,
                "publicKey": public_key,
                "keyVersion": 1,
                "trustScore": trust_score,
                "status": "ACTIVE",
                "clusterID": cluster_id,
                "lastVerified": int(time.time())
            }

        device["lastVerified"] = int(time.time())
        stub.put_state(device_key, json.dumps(device).encode('utf-8'))

    # ------------------------------------------------------------------
    # GetDevice – Query current device state
    # ------------------------------------------------------------------
    def get_device(self, stub: ChaincodeStub, args: list) -> Dict[str, Any]:
        if len(args) != 1:
            return self._error_response("Expecting deviceID")

        device_id = args[0]
        device_key = f"DEVICE-{device_id}"
        data = stub.get_state(device_key)

        if not data:
            return self._error_response(f"Device {device_id} not found")

        return self._success_response(json.loads(data))

    # ------------------------------------------------------------------
    # GetRequestHistory – Get full history of a key request
    # ------------------------------------------------------------------
    def get_request_history(self, stub: ChaincodeStub, args: list) -> Dict[str, Any]:
        if len(args) != 1:
            return self._error_response("Expecting requestID")

        request_id = args[0]
        history_iter = stub.get_history_for_key(request_id)
        history = []

        for record in history_iter:
            try:
                value = json.loads(record.value.decode('utf-8'))
                value["txID"] = record.tx_id
                value["timestamp"] = record.timestamp.seconds
                value["isDelete"] = record.is_delete
                history.append(value)
            except:
                continue

        return self._success_response({"history": history})

    # ------------------------------------------------------------------
    # Chaincode entry points
    # ------------------------------------------------------------------
    def Invoke(self, stub: ChaincodeStub):
        function, args = stub.get_function_and_params()

        if function == "submitKeyRequest":
            return self.submit_key_request(stub, args)
        elif function == "getDevice":
            return self.get_device(stub, args)
        elif function == "getRequestHistory":
            return self.get_request_history(stub, args)

        return self._error_response(f"Invalid function: {function}")

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------
    def _success_response(self, data: dict):
        return {"status": "success", "data": data}

    def _error_response(self, message: str):
        return {"status": "error", "message": message}


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    server = FBSSmartContract()
    server.start()
