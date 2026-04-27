# Use your conda / venv / uv env
conda activate vllm-xformers-env
pip install pandas matplotlib

# Install AMD smi python bindings
# Both MI210 / MI300 at the directory
cp -r /opt/rocm/share/amd_smi ./amd_smi_build
cd amd_smi_build
pip install .
cd ..
