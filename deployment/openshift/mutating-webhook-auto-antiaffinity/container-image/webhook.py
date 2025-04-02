import json
import base64
from flask import Flask, request, jsonify
import ssl

app = Flask(__name__)

def create_antiaffinity_patch(pod):
    annotations = pod.get('metadata', {}).get('annotations', {})
    if 'scheduler.alpha.kubernetes.io/antiaffinity' in annotations:
        return {
            "op": "add",
            "path": "/spec/affinity",
            "value": {
                "podAntiAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "labelSelector": {
                                "matchExpressions": [
                                    {
                                        "key": "app",
                                        "operator": "In",
                                        "values": [pod['metadata'].get('labels', {}).get('app', '')]
                                    }
                                ]
                            },
                            "topologyKey": "kubernetes.io/hostname"
                        }
                    ]
                }
            }
        }
    return None

@app.route('/mutate', methods=['POST'])
def mutate_pod():
    admission_review = request.get_json()
    pod = admission_review['request']['object']
    
    patch = create_antiaffinity_patch(pod)
    
    # Debugging: Print incoming admission review and patch
    print(f"Received AdmissionReview: {json.dumps(admission_review, indent=2)}")
    print(f"Generated patch: {json.dumps(patch, indent=2)}")

    if patch:
        response = {
            "apiVersion": "admission.k8s.io/v1",  # Make sure apiVersion is correct
            "kind": "AdmissionReview",  # Kind must be AdmissionReview
            "response": {
                "uid": admission_review['request']['uid'],
                "allowed": True,
                "patchType": "JSONPatch",
                "patch": base64.b64encode(json.dumps([patch]).encode()).decode()
            }
        }
    else:
        response = {
            "apiVersion": "admission.k8s.io/v1",  # Ensure apiVersion is set
            "kind": "AdmissionReview",  # Ensure kind is set to AdmissionReview
            "response": {
                "uid": admission_review['request']['uid'],
                "allowed": True
            }
        }

    # Debugging: Print outgoing response
    print(f"Sending AdmissionReview response: {json.dumps(response, indent=2)}")
    
    return jsonify(response)

if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('webhook.crt', 'webhook.key')
    app.run(host='0.0.0.0', port=8443, ssl_context=context)
