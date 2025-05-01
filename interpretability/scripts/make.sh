cd /data/disk1/guohaoran/transformers
export PYTHONPATH=$(pwd)/src
python utils/check_modular_conversion.py --fix_and_overwrite --num_workers 32
