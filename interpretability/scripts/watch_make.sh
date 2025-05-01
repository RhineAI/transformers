export PYTHONPATH=$(pwd)/src
watchmedo shell-command \
  --patterns="*.py" \
  --recursive \
  --command='python utils/check_modular_conversion.py --fix_and_overwrite --num_workers 32' \
  src/transformers/models/qwen3/
