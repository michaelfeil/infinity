Deployment via [Modal](https://modal.com).

The serverless deployment is now live at https://infinity.modal.michaelfeil.eu. The Nvidia L4/A100 deployment is hosted via Modal.com, and can be hosted free of charge.

### Local
Run the following sequence of commands to deploy:

```bash
git clone https://github.com/michaelfeil/infinity
pip install modal==0.66.0
modal deploy --env main infra.modal.webserver
```

### Github Actions
For automatic integration via Github Actions, follow this Pipeline:
https://github.com/michaelfeil/infinity/blob/main/.github/workflows/release_modal_com.yaml
