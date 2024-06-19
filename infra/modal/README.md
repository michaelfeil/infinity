Deployment via Modal.

The serverless deployment is now live at https://infinity.modal.michaelfeil.eu

Deployed via:
```bash
pip install modal
# modal setup
cd infra/modal
modal serve --env main webserver.py
```