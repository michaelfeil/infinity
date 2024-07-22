Deployment via [Modal](https://modal.com).

The serverless deployment is now live at https://infinity.modal.michaelfeil.eu. The free Nvidia L4/A100 deployment is sponsored by Modal.com, feel free to use as long as the endpoint is reachable.

### Local
Run the following sequence of commands to deploy:

```bash
git clone https://github.com/michaelfeil/infinity
cd infra/modal
pip install modal
modal deploy --env main modal.infra.webserver
```

### Github Actions
For automatic integration via Github Actions, follow this Pipeline:
https://github.com/michaelfeil/infinity/blob/main/.github/workflows/release_modal_com.yaml
