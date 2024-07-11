# Generate Portable Executable

This section describe the steps to prepare a single executable for infinity.
It is setup using pyinstaller. The setup step has been verified on Windows 11.
However, the compilation procedure should be the same on other platforms.

## Windows (CPU)
1. Create a python virtual environment.
2. `pip install infinity-emb[all]`.
3. Install pyinstaller. `pip install pyinstaller`.
4. Compile executable. `pyinstaller .\infinity_server.spec`